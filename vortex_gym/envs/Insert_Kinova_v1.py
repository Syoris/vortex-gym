import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pyvortex.vortex_env import VortexEnv
from spatialmath import SE3
from vortex_gym.robot.kinova_gen_2 import KinovaGen2, KinovaVxIn, KinovaVxOut
from pyvortex.vortex_classes import VortexInterface

from vortex_gym import ASSETS_DIR


class SceneVxIn(VortexInterface):
    socket_pose: str = 'socket_pose'


class InsertKinovaV1(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 100}

    # --------------------------------------------------------------------------------------------
    # MARK: Initialization
    # --------------------------------------------------------------------------------------------
    def __init__(
        self,
        render_mode=None,
        sim_time_step=0.01,
        insertion_time=2.5,
        z_insertion=0.07,
        speed_misaligment_range=(0.0, 0.0),
        socket_x_range=(0.55, 0.55),  # Range of the socket x position [m] (0.529, 0.529)
        socket_x_offset=0.005,  # Offset of the socket x position [m]
        eval_mode=False,
        viewpoint=None,
        ctrl_freq=100,
        reward_weight=1,
        action_coeff=1,
    ):
        print('[InsertKinovaV1.__init__] Initializing InsertKinovaV1 gym environment')
        # Task parameters
        self.sim_time_step = sim_time_step  # Simulation time step [sec]
        self.ctrl_freq = ctrl_freq  # Control frequency [Hz]
        self._n_sim_steps = int((1 / self.sim_time_step) / self.ctrl_freq)  # Number of steps per control frequency

        self.insertion_time = insertion_time  # Time to insert the peg in the hole [sec]
        self.z_insertion = z_insertion  # Depth of the peg insertion [m]
        self.z_insertion_speed = self.z_insertion / self.insertion_time  # Speed of the peg insertion [m/s]

        self.speed_misalignment = 0.0  # Misalignment of the peg insertion. Changes the angle of the velocity [deg]
        self.speed_misalignment_range: tuple = speed_misaligment_range  # Range of misalignment [deg]

        self.socket_x = 0.550  # X position of the socket [m]
        self.socket_x_range: tuple = socket_x_range
        self.socket_x_offset = socket_x_offset  # Offset of the socket x position [m]
        self.socket_default_pose = np.array([[1, 0, 0, 0.550], [0, 1, 0, -0.007], [0, 0, 1, 0.0], [0, 0, 0, 1.0]])
        self.randomization_start = 20  # Number of episodes before randomizing the socket position

        self.eval_mode = eval_mode

        # Vortex environment
        self._assets_dir = ASSETS_DIR
        if not self.eval_mode:
            self._config_file = 'config.vxc'
            self._content_file = 'Kinova Gen2 Unjamming/Scenes/kinova_peg-in-hole.vxscene'
        else:
            self._config_file = 'config.vxc'
            self._content_file = 'Kinova Gen2 Unjamming/Scenes/kinova_peg-in-hole_eval.vxscene'

        self._kinova_vx_in = KinovaVxIn()
        self._kinova_vx_out = KinovaVxOut()
        self._scene_vx_in = SceneVxIn()

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        assert viewpoint is None or viewpoint in [
            'Global',
            'Perspective',
        ], 'Invalid viewpoint. Use "Global" or "Perspective"'
        if viewpoint is None:
            viewpoints = ['Perspective']
        else:
            viewpoints = [viewpoint]

        self.vortex_env = VortexEnv(
            assets_dir=ASSETS_DIR,
            h=self.sim_time_step,
            config_file=self._config_file,
            content_file=self._content_file,
            viewpoints=viewpoints,  # ['Global', 'Perspective'],
            render=True if render_mode == 'human' else False,
        )

        self.robot = KinovaGen2(self.vortex_env)
        self._J = None  # Current Jacobian matrix

        # RL Variables and Hyperparameters
        self.n_action = 3
        self.action = np.zeros(self.n_action)  # Last action taken by the agent
        self.command = np.zeros(3)  # Command sent to the robot [j2, j4, j6]
        self.ik_joints_vels = np.zeros(3)  # Joint velocities computed by the IK
        self.ee_vel_ctrl = np.zeros(3)  # Desired ee vel in task space, output of the controller
        self.ee_vel_aug = np.zeros(3)  # Desired ee vel in task space, augmented

        # Init observation and action spaces
        self._init_spaces()

        self.obs = None  # observation dict from the last step, updated in `step` method
        self.obs_normalized = None  # observation dict from the last step, normalized
        self.info = None  # info dict from the last step, updated in `step` method
        self.ep_completed = False  # Flag indicating if the simulation is completed

        self.step_count = 0  # Number of steps taken in the current episode
        self.episode_count = 0  # Number of episodes taken in the current training session
        self.max_step_per_ep = int(insertion_time * self.ctrl_freq)  # Maximum number of steps per episode

        # Scene Parameters
        self.socket_pose = [0, 0, 0]

        # RL HP
        self.action_coeff = action_coeff
        self.reward_weight = reward_weight
        self.reward_clipping = 10

        # Initialize robot
        self.robot.go_home()
        self.robot.set_joints_vels(self.command)
        self.vortex_env.step()
        self.vortex_env.save_current_frame()

        # Compute traj
        traj_joint_angles, self.joints_vel_traj = self.robot.compute_joint_vels_traj(
            -self.z_insertion, -self.z_insertion_speed, self.max_step_per_ep
        )

        self.reset()
        self.episode_count = 0  # Number of episodes taken in the current training session
        print('[InsertKinovaV1.__init__] InsertKinovaV1 environment initialized')

    def _init_spaces(self):
        # Observation space
        self.observation_space = spaces.Dict(
            {
                'joint_angles': self.robot.joints_angles_obs_space,
                'joint_vels': self.robot.joints_vels_obs_space,
                'joint_torques': self.robot.joints_torques_obs_space,
                'joint_target_vels': self.robot.joints_vels_obs_space,
            }
        )

        # Action space
        # Actuator bounds - Torques
        self._actuator_low_bound = self.robot.joints_torques_obs_space.low[[0, 2]]
        self._actuator_high_bound = self.robot.joints_torques_obs_space.high[[0, 2]]
        self._joint_max_torque = 30.5  # Maximum

        # Actuator bounds - Velocities
        self._joint_max_speed = (
            30.0  # Maximum joint speed [deg/s] (Manually set for now, all joints set to the same speed)
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_action,), dtype=np.float32)

    # --------------------------------------------------------------------------------------------
    # MARK: Gym methods
    # --------------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Reset vortex
        self.vortex_env.reset_saved_frame()
        # self.vortex_env.pause_sim(True)

        self.speed_misalignment = np.random.uniform(self.speed_misalignment_range[0], self.speed_misalignment_range[1])

        # Get observation and info
        self.obs, self.obs_normalized = self._get_obs()
        self.ik_joints_vels = self._get_ik_vels(self.obs['joint_angles'], desired_vel=np.zeros(3))
        info = self._get_info()

        if (self.episode_count > self.randomization_start) or self.eval_mode:
            self.socket_x = np.random.uniform(self.socket_x_range[0], self.socket_x_range[1])
            socket_x_offset = np.random.uniform(-self.socket_x_offset, self.socket_x_offset)
            self.socket_x += socket_x_offset

            new_socket_pose = self.socket_default_pose.copy()
            new_socket_pose[0, 3] = self.socket_x
            self.vortex_env.set_input(self._scene_vx_in.socket_pose, new_socket_pose)

        self.episode_count += 1
        self.ep_completed = False

        self.render()
        self.vortex_env.step()

        # self.vortex_env.pause_sim(False)

        return self.obs, info

    def step(self, action):
        """Take one step. This is the main function of the environment.

        The state of the robot is defined as all the measurements that the physical robot could obtain from sensors:
        - position
        - vel
        - ideal vel
        - torque

        The info returned is the other information that might be useful for analysis, but not for learning:
        - command
        - plug force
        - plug torque

        Args:
            action (np.Array): The action to take. Defined as a correction to the desired task space velocity

        Returns:
            obs (np.Array): The observation of the environment after taking the step
            reward (float): The reward obtained after taking the step
            sim_completed (bool): Flag indicating if the simulation is completed
            done (bool): Flag indicating if the episode is done
            info (dict): Additional information about the step
        """
        terminated = False

        x_vel_ctrl = self.z_insertion_speed * np.sin(np.deg2rad(self.speed_misalignment))
        z_vel_ctrl = self.z_insertion_speed * np.cos(np.deg2rad(self.speed_misalignment))
        rot_vel_ctrl = 0.0
        self.ee_vel_ctrl = np.array([x_vel_ctrl, -z_vel_ctrl, rot_vel_ctrl])

        self.action = action

        self.ee_vel_aug = self.ee_vel_ctrl + self.action_coeff * self.action

        self.ik_joints_vels = self._get_ik_vels(self.obs['joint_angles'], desired_vel=self.ee_vel_aug)

        # Apply actions
        self.command = np.array([self.ik_joints_vels[0], self.ik_joints_vels[1], self.ik_joints_vels[2]])
        self.robot.set_joints_vels(self.command)

        # Step the simulation
        for _ in range(self._n_sim_steps):
            self.vortex_env.step()

        # Observations
        self.obs, self.obs_normalized = self._get_obs()

        # Info
        self.info = self._get_info()  # plug force and torque

        # Reward
        reward = self._compute_reward()

        # Done flag
        self.step_count += 1
        if self.step_count >= self.max_step_per_ep:
            self.ep_completed = True

            # Check if it is a success
            self.info['is_success'] = self._is_success()

        return self.obs_normalized, reward, self.ep_completed, terminated, self.info

    def render(self):
        """Render the environment.

        Gymnasium docs: https://gymnasium.farama.org/api/env/#gymnasium.Env.render
        """
        if self.render_mode == 'human':
            active = True
        else:
            active = False

        self.vortex_env.render(active=active)

    def close(self): ...

    # --------------------------------------------------------------------------------------------
    # MARK: Utilities
    # --------------------------------------------------------------------------------------------
    def _get_obs(self) -> dict:
        """Observation of the environment.

        - joint_angles (np.array): Joint angles [j2, j4, j6] [deg]
        - joint_vels (np.array): Joint velocities [j2, j4, j6] [deg/s]
        - joint_torques (np.array): Joint torques [j2, j4, j6] [Nm]
        - joint_target_vels (np.array): Target joint velocities [j2, j4, j6] [deg/s]
        """
        joints_states = self.robot.joints

        joint_angles = np.array(
            [joints_states.angles[1], joints_states.angles[3], joints_states.angles[5]], dtype=np.float32
        )
        joint_vels = np.array([joints_states.vels[1], joints_states.vels[3], joints_states.vels[5]], dtype=np.float32)
        joint_torques = np.array(
            [joints_states.torques[1], joints_states.torques[3], joints_states.torques[5]], dtype=np.float32
        )
        joint_vels_cmd = np.array(
            [joints_states.vels_cmds[1], joints_states.vels_cmds[3], joints_states.vels_cmds[5]], dtype=np.float32
        )

        obs = {
            'joint_angles': joint_angles,
            'joint_vels': joint_vels,
            'joint_torques': joint_torques,
            'joint_target_vels': joint_vels_cmd,
        }
        # Normalize the observations

        vels_normalized = joint_vels / self._joint_max_speed
        vels_cmds_normalized = joint_vels_cmd / self._joint_max_speed
        torques_normalized = joint_torques / self._joint_max_torque
        angles_normalized = joint_angles / 180.0

        obs_normalized = {
            'joint_angles': angles_normalized,
            'joint_vels': vels_normalized,
            'joint_torques': torques_normalized,
            'joint_target_vels': vels_cmds_normalized,
        }

        # TODO: Add noise to the observations
        ...

        return obs, obs

    def _get_info(self) -> dict:
        """Get additional information about the environment.

        - action (np.array): The action taken by the agent [j2_aug, j6_aug]
        - command (np.array): The command sent to the robot [j2, j4, j6]
        - peg_force (np.array): The force applied to the peg [fx, fy, fz]
        - peg_torque (np.array): The torque applied to the peg [tx, ty, tz]
        - peg_pose ((np.array, np.array)): The pose of the tool ([x, y, z], [roll, pitch, yaw])
        - ee_pose ((np.array, np.array)): The pose of the end-effector ([x, y, z], [roll, pitch, yaw])
        # - insertion_depth (float): The depth of the peg in the hole
        - misaligment (float): Misaligment angle

        Returns:
            dict: _description_
        """
        ee_pose = self.robot.ee_pose
        peg_pose = self.robot.peg_pose
        ee_vel = self._J @ np.deg2rad(self.obs['joint_vels'])

        peg_force = self.robot.get_peg_force()
        peg_torque = self.robot.get_peg_torque()

        info_dict = {
            'action': self.action,
            'joint_cmd': self.command,  # Command sent to the robot [j2, j4, j6]
            'ik_joint_vels': self.joints_vel_traj[self.step_count],  # Expected joint velocities from IK trajectory
            'ee_vel_ctrl': self.ee_vel_ctrl,  # Desired ee vel in task space, output of the controller
            'ee_vel_aug': self.ee_vel_aug,  # Desired ee vel in task space, augmented
            'ee_vel': ee_vel,  # End-effector velocity [m/s, m/s, rad/s]
            'peg_force_x': peg_force[0],
            'peg_force_y': peg_force[1],
            'peg_force_z': peg_force[2],
            'peg_force_norm': np.linalg.norm(peg_force),
            'peg_torque_x': peg_torque[0],
            'peg_torque_y': peg_torque[1],
            'peg_torque_z': peg_torque[2],
            'peg_torque_norm': np.linalg.norm(peg_torque),
            'peg_pose': (peg_pose.t, peg_pose.rpy(order='xyz', unit='deg')),
            'peg_pose_z': peg_pose.t[2],
            'ee_pose': (ee_pose.t, ee_pose.rpy(order='xyz', unit='deg')),
            'speed_misaligment': self.speed_misalignment,
            'socket_x': self.socket_x,
            # 'insertion_depth': self.robot.get_insertion_depth(),
        }

        return info_dict

    def _compute_reward(self) -> float:
        obs = self.obs
        joint_vels = obs['joint_vels']
        # joint_id_vels = obs['target_vels']
        joint_ik_vels = self.joints_vel_traj[self.step_count]
        joint_torques = obs['joint_torques']

        reward = -self.reward_weight * np.sum(abs((joint_ik_vels - joint_vels) * joint_torques))

        # reward = np.clip(reward, -self.reward_clipping, self.reward_clipping) # TODO: Reward clipping

        return reward

    def _build_Jacobian(self, th_current: np.ndarray) -> np.ndarray:
        """Build the manipulator's Jacobian matrix.

        Args:
            th_current (np.ndarray): Current joint positions [j2, j4, j6] [deg]

        Returns:
            np.ndarray: Jacobian matrix
        """
        q2 = np.deg2rad(th_current[0])
        q4 = np.deg2rad(th_current[1])
        q6 = np.deg2rad(th_current[2])

        a_x = (
            -self.robot.L34 * np.cos(-q2)
            - self.robot.L56 * np.cos(-q2 + q4)
            - self.robot.L78 * np.cos(-q2 + q4 - q6)
            - self.robot.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)
        )
        b_x = (
            self.robot.L56 * np.cos(-q2 + q4)
            + self.robot.L78 * np.cos(-q2 + q4 - q6)
            + self.robot.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)
        )
        c_x = -self.robot.L78 * np.cos(-q2 + q4 - q6) - self.robot.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)

        a_z = (
            self.robot.L34 * np.sin(-q2)
            + self.robot.L56 * np.sin(-q2 + q4)
            + self.robot.L78 * np.sin(-q2 + q4 - q6)
            + self.robot.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)
        )
        b_z = (
            -self.robot.L56 * np.sin(-q2 + q4)
            - self.robot.L78 * np.sin(-q2 + q4 - q6)
            - self.robot.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)
        )
        c_z = self.robot.L78 * np.sin(-q2 + q4 - q6) + self.robot.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)

        J = [[a_x, b_x, c_x], [a_z, b_z, c_z], [-1.0, 1.0, -1.0]]

        return J

    def _get_ik_vels(self, q: np.ndarray, desired_vel: np.ndarray = None) -> np.ndarray:
        """Compute the joint velocities to go straight down with a misalignment.

        Args:
            q (np.ndarray): Current joint positions [j2, j4, j6] [deg]
            desired_vel (np.ndarray): Desired velocity [x, z, rot] [m/s, m/s, rad/s]

        Returns:
            np.ndarray: Desired joint velocities [j2, j4, j6] [deg/s]
        """
        # J = self._build_Jacobian(q)
        self._J = self.robot.compute_jacob0_3dof()

        Jinv = np.linalg.inv(self._J)
        q_vel = np.dot(Jinv, desired_vel)

        return np.rad2deg(q_vel)

    def _get_socket_pose(self) -> np.ndarray:
        """Return the world transformation matrix of the socket.

        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        return self.vortex_env.get_input(self._scene_vx_in.socket_pose)

    def _is_success(self) -> bool:
        """Check if the task was successful.

        Success is defined as the peg being inserted in the hole between:
            - z_peg:
            - x_peg:
        """
        x_range = 0.02  # 2 cm
        x_target = 0.55
        x_lims = [x_target - x_range, x_target + x_range]

        # z_range = 0.01  # 1 cm
        # z_target = 0.02
        # z_socket = 0.08 # Top of the socket
        z_lims = [0, 0.06]  # Sucess if 2cm in the hole

        peg_pose = self.info['peg_pose'][0]
        peg_pose_z = peg_pose[2]
        peg_pose_x = peg_pose[0]

        return (z_lims[0] <= peg_pose_z <= z_lims[1]) and (x_lims[0] <= peg_pose_x <= x_lims[1])
