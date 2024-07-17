import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pyvortex.vortex_env import VortexEnv
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
        max_epoch_time=3,  # Maximum time of the episode [sec]
        z_insertion=0.07,  # dz of the peg insertion [m]
        speed_misaligment_range=(0.0, 0.0),
        socket_x_range=(0.55, 0.55),  # Range of the socket x position [m] (0.529, 0.529)
        socket_x_offset=0.005,  # Offset of the socket x position [m]
        eval_mode=False,
        viewpoint=None,
        ctrl_freq=50,
        reward_weight=1,
        action_coeff=[1, 1, 1],
    ):
        print('[InsertKinovaV1.__init__] Initializing InsertKinovaV1 gym environment')
        # Task parameters
        self.sim_time_step = sim_time_step  # Simulation time step [sec]
        self.ctrl_freq = ctrl_freq  # Control frequency [Hz]
        self._n_sim_steps = int((1 / self.sim_time_step) / self.ctrl_freq)  # Number of steps per control frequency

        self.max_epoch_time = max_epoch_time  # Maximum time of the episode [sec]
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

        # Set viewpoint
        if viewpoint is None:
            viewpoints = ['Perspective']

        elif isinstance(viewpoint, str):
            assert viewpoint is None or viewpoint in [
                'Global',
                'Perspective',
            ], 'Invalid viewpoint. Use "Global" or "Perspective"'

            viewpoints = [viewpoint]

        elif isinstance(viewpoint, list):
            for each_viewpoint in viewpoint:
                assert each_viewpoint is None or each_viewpoint in [
                    'Global',
                    'Perspective',
                ], f'Invalid viewpoint [{each_viewpoint}]. Use "Global" or "Perspective"'
            viewpoints = viewpoint

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
        self._J_inv = None  # Current Inverse Jacobian matrix

        # RL Variables and Hyperparameters
        self.n_action = 3
        self.action = np.zeros(self.n_action)  # Last action taken by the agent
        self.joint_cmd = np.zeros(3)  # Command sent to the robot [j2, j4, j6]
        self.joint_vels_ideal = np.zeros(3)  # Ideal Joint velocities, from traj or controller
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
        self.max_step_per_ep = int(self.max_epoch_time * self.ctrl_freq)  # Maximum number of steps per episode

        # Scene Parameters
        self.socket_pose = [0, 0, 0]

        # RL HP
        self.action_coeff = np.array(action_coeff)
        self.reward_weight = reward_weight
        self.reward_clipping = 10

        # Initialize robot
        self.robot.go_home()
        self.robot.set_joints_vels(self.joint_cmd)
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
                # 'joint_angles': self.robot.joints_angles_obs_space,
                'joint_vels': self.robot.joints_vels_obs_space,
                'joint_torques': self.robot.joints_torques_obs_space,
                'joint_vels_ideal': self.robot.joints_vels_obs_space,
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
        self._update_jacobian()
        self.joint_vels_ideal = self._get_ik_vels(desired_vel=np.zeros(3))

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
        - joint_cmd
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
        self.action = action
        self._update_jacobian()

        # Controller output
        # x_vel_ctrl = self.z_insertion_speed * np.sin(np.deg2rad(self.speed_misalignment))
        # z_vel_ctrl = self.z_insertion_speed * np.cos(np.deg2rad(self.speed_misalignment))
        # rot_vel_ctrl = 0.0
        # self.ee_vel_ctrl = np.array([x_vel_ctrl, -z_vel_ctrl, rot_vel_ctrl])
        self.ee_vel_ctrl = np.array([0, 0, 0])

        # Augmented action
        self.ee_vel_aug = self.ee_vel_ctrl + self.action_coeff * self.action
        joint_vels_aug = self._get_ik_vels(desired_vel=self.ee_vel_aug)

        # Expected joint velocities
        # self.joint_vels_ideal = self._get_ik_vels(desired_vel=self.ee_vel_ctrl)  # From ctrl
        self.joint_vels_ideal = self.joints_vel_traj[self.step_count]  # From traj

        # Apply actions
        self.joint_cmd = np.array([joint_vels_aug[0], joint_vels_aug[1], joint_vels_aug[2]])
        self.robot.set_joints_vels(self.joint_cmd)

        # Step the simulation
        for _ in range(self._n_sim_steps):
            self.vortex_env.step()

        # Observations
        self.obs, self.obs_normalized = self._get_obs()

        # Info
        self.info = self._get_info()  # plug force and torque

        # --- Reward ---
        # DENSE
        reward = self._compute_reward()

        # # SPARSE
        # is_success = self._is_success()
        # if is_success:
        #     reward = 1
        #     self.info['is_success'] = True
        #     self.ep_completed = True
        # else:
        #     self.info['is_success'] = False
        #     reward = 0

        # --- Success ---
        success = self._is_success()
        if success:
            reward += 10
            self.ep_completed = True
            self.info['is_success'] = success

        # Done flag
        self.step_count += 1
        if self.step_count >= self.max_step_per_ep:
            # self.ep_completed = True
            terminated = True

            # Check if it is a success
            self.info['is_success'] = success

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
        - joint_vels_ideal (np.array): Target joint velocities [j2, j4, j6] [deg/s]
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

        joint_vels_ideal = self.joint_vels_ideal.astype(np.float32)

        obs = {
            # 'joint_angles': joint_angles,
            'joint_vels': joint_vels,
            'joint_torques': joint_torques,
            'joint_vels_ideal': joint_vels_cmd,  # joint_vels_ideal
        }

        # # Normalize the observations

        # vels_normalized = joint_vels / self._joint_max_speed
        # vels_cmds_normalized = joint_vels_cmd / self._joint_max_speed
        # torques_normalized = joint_torques / self._joint_max_torque
        # angles_normalized = joint_angles / 180.0

        # obs_normalized = {
        #     'joint_angles': angles_normalized,
        #     'joint_vels': vels_normalized,
        #     'joint_torques': torques_normalized,
        #     'joint_vels_ideal': vels_cmds_normalized,
        # }

        # TODO: Add noise to the observations
        ...

        return obs, obs

    def _get_info(self) -> dict:
        """Get additional information about the environment.

        - action (np.array): The action taken by the agent [j2_aug, j6_aug]
        - joint_cmd (np.array): The joint_cmd sent to the robot [j2, j4, j6]
        - peg_force (np.array): The force applied to the peg [fx, fy, fz]
        - peg_torque (np.array): The torque applied to the peg [tx, ty, tz]
        - peg_pose ((np.array, np.array)): The pose of the tool ([x, y, z], [roll, pitch, yaw])
        - ee_pose ((np.array, np.array)): The pose of the end-effector ([x, y, z], [roll, pitch, yaw])
        //- insertion_depth (float): The depth of the peg in the hole
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
            'joint_cmd': self.joint_cmd,  # Command sent to the robot [j2, j4, j6]
            'joint_vels_ideal': self.joint_vels_ideal,  # Expected joint velocities from IK trajectory or controller
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
        joint_vels_ideal = self.joint_vels_ideal
        joint_torques = obs['joint_torques']

        # # --- Force-based reward ---
        # reward = -self.reward_weight * np.sum(abs((joint_vels_ideal - joint_vels) * joint_torques))

        # # --- z-dist, variable ---
        # peg_z_start = 0.09037613998260946
        # exp_dz = self.z_insertion_speed * self.step_count * self.sim_time_step * self._n_sim_steps
        # # z_goal = peg_z_start - exp_dz

        # peg_pose_z = self.info['peg_pose_z']
        # k_peg_dz = peg_z_start - peg_pose_z

        # reward = -(abs(exp_dz - k_peg_dz))

        # --- 2-norm ---
        z_goal = 0.02
        x_goal = 0.55
        goal_array = np.array([x_goal, z_goal])
        peg_pose = self.info['peg_pose'][0]
        peg_pose_array = np.array([peg_pose[0], peg_pose[2]])

        reward = -np.linalg.norm(goal_array - peg_pose_array)

        return reward

    def _update_jacobian(self):
        """Update the Jacobian matrix of the robot and compute the inverse."""
        self._J = self.robot.compute_jacob0_3dof()

        self._J_inv = np.linalg.inv(self._J)

    def _get_ik_vels(self, desired_vel: np.ndarray = None) -> np.ndarray:
        """Compute the joint velocities so the end-effector moves with the desired velocity.

        Args:
            desired_vel (np.ndarray): Desired velocity [x, z, rot] [m/s, m/s, rad/s]

        Returns:
            np.ndarray: Desired joint velocities [j2, j4, j6] [deg/s]
        """
        q_vel = self._J_inv @ desired_vel

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
        z_lims = [0.01, 0.03]

        peg_pose = self.info['peg_pose'][0]
        peg_pose_z = peg_pose[2]
        peg_pose_x = peg_pose[0]

        return (z_lims[0] <= peg_pose_z <= z_lims[1]) and (x_lims[0] <= peg_pose_x <= x_lims[1])
