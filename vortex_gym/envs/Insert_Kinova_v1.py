import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pyvortex.vortex_env import VortexEnv
from vortex_gym.robot.kinova_gen_2 import KinovaGen2, KinovaVxIn, KinovaVxOut
from vortex_gym import ASSETS_DIR


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
        misaligment_range=(0.0, 0.0),
        eval_mode=False,
    ):
        print('[InsertKinovaV1.__init__] Initializing InsertKinovaV1 gym environment')
        # Task parameters
        self.sim_time_step = sim_time_step  # Simulation time step [sec]
        self.insertion_time = insertion_time  # Time to insert the peg in the hole [sec]
        self.z_insertion = z_insertion  # Depth of the peg insertion [m]
        self.z_insertion_speed = self.z_insertion / self.insertion_time  # Speed of the peg insertion [m/s]

        self.misalignment = 0.0  # Misalignment of the peg insertion. Changes the angle of the velocity [deg]
        self.misalignment_range: tuple = misaligment_range  # Range of misalignment [deg]

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

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.vortex_env = VortexEnv(
            assets_dir=ASSETS_DIR,
            h=self.sim_time_step,
            config_file=self._config_file,
            content_file=self._content_file,
            viewpoints=['Perspective'],  # ['Global', 'Perspective'],
            render=True if render_mode == 'human' else False,
        )

        self.robot = KinovaGen2(self.vortex_env)

        # Init observation and action spaces
        self._init_spaces()

        # RL Variables and Hyperparameters
        self.action = np.zeros(2)  # Last action taken by the agent
        self.command = np.zeros(3)  # Command sent to the robot [j2, j4, j6]
        self.ik_joints_vels = np.zeros(3)  # Joint velocities computed by the IK

        self.obs = None  # observation dict from the last step, updated in `step` method
        self.info = None  # info dict from the last step, updated in `step` method
        self.ep_completed = False  # Flag indicating if the simulation is completed

        self.step_count = 0  # Number of steps taken in the current episode
        self.episode_count = 0  # Number of episodes taken in the current training session
        self.max_step_per_ep = 250  # Maximum number of steps per episode

        # RL HP
        self.action_coeff = 0.01
        self.reward_weight = 0.04

        # Initialize robot
        self.robot.go_home()
        self.vortex_env.save_current_frame()

        self.reset()
        print('[InsertKinovaV1.__init__] InsertKinovaV1 environment initialized')

    def _init_spaces(self):
        # Observation space
        self.observation_space = spaces.Dict(
            {
                'angles': self.robot.joints_angles_obs_space,
                'velocities': self.robot.joints_vels_obs_space,
                'torques': self.robot.joints_torques_obs_space,
                'target_vels': self.robot.joints_vels_obs_space,
            }
        )

        # Action space
        # Actuator bound to scale the action
        self._actuator_low_bound = self.robot.joints_torques_obs_space.low[[0, 2]]
        self._actuator_high_bound = self.robot.joints_torques_obs_space.high[[0, 2]]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # --------------------------------------------------------------------------------------------
    # MARK: Gym methods
    # --------------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset vortex
        self.vortex_env.reset_saved_frame()

        self.misalignment = np.random.uniform(self.misalignment_range[0], self.misalignment_range[1])

        # Get observation and info
        self.obs = self._get_obs()
        info = self._get_info()

        self.step_count = 0
        self.ep_completed = False

        self.render()

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
            action (np.Array): The action to take. Defined as a correction to the joint velocities

        Returns:
            obs (np.Array): The observation of the environment after taking the step
            reward (float): The reward obtained after taking the step
            sim_completed (bool): Flag indicating if the simulation is completed
            done (bool): Flag indicating if the episode is done
            info (dict): Additional information about the step
        """
        terminated = False

        self.action = action

        self.ik_joints_vels = self._get_ik_vels(self.obs['angles'])
        # self.ik_joints_vels = [5.0, 5.0, 5.0]

        # Scale actions
        act_j2 = self.action_coeff * self.action[0] * self._actuator_high_bound[0]
        act_j6 = self.action_coeff * self.action[1] * self._actuator_high_bound[1]

        j2_vel = self.ik_joints_vels[0] - act_j2
        j4_vel = self.ik_joints_vels[1] + act_j6 - act_j2
        j6_vel = self.ik_joints_vels[2] + act_j6

        # Apply actions
        self.command = np.array([j2_vel, j4_vel, j6_vel])
        self.robot.set_joints_vels(self.command)

        # Step the simulation
        self.vortex_env.step()

        # Observations
        self.obs = self._get_obs()

        # Info
        self.info = self._get_info()  # plug force and torque

        # Reward
        reward = self._compute_reward()

        # Done flag
        self.step_count += 1
        if self.step_count >= self.max_step_per_ep:
            self.ep_completed = True

        return self.obs, reward, self.ep_completed, terminated, self.info

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
        joints_states = self.robot.joints

        angles = np.array([joints_states.angles[1], joints_states.angles[3], joints_states.angles[5]], dtype=np.float32)
        vels = np.array([joints_states.vels[1], joints_states.vels[3], joints_states.vels[5]], dtype=np.float32)
        torques = np.array(
            [joints_states.torques[1], joints_states.torques[3], joints_states.torques[5]], dtype=np.float32
        )
        vels_cmds = np.array(
            [joints_states.vels_cmds[1], joints_states.vels_cmds[3], joints_states.vels_cmds[5]], dtype=np.float32
        )

        return {
            'angles': angles,
            'velocities': vels,
            'torques': torques,
            'target_vels': vels_cmds,
        }

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

        peg_force = self.robot.get_peg_force()
        peg_torque = self.robot.get_peg_torque()

        info_dict = {
            'action': self.action,
            'command': self.command,
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
            'misaligment': self.misalignment,
            # 'insertion_depth': self.robot.get_insertion_depth(),
        }

        return info_dict

    def _compute_reward(self) -> float:
        joint_vels = self.obs['velocities']
        joint_id_vels = self.obs['target_vels']
        joint_torques = self.obs['torques']

        reward = self.reward_weight * (
            -abs((joint_id_vels[0] - joint_vels[0]) * joint_torques[0])
            - abs((joint_id_vels[1] - joint_vels[1]) * joint_torques[1])
            - abs((joint_id_vels[2] - joint_vels[2]) * joint_torques[2])
        )
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

    def _get_ik_vels(self, q: np.ndarray):
        """Compute the joint velocities to go straight down with a misalignment.

        Args:
            q (np.ndarray): Current joint positions [j2, j4, j6] [deg]

        Returns:
            np.ndarray: Desired joint velocities [j2, j4, j6] [deg/s]
        """
        x_vel = self.z_insertion_speed * np.sin(np.deg2rad(self.misalignment))
        z_vel = self.z_insertion_speed * np.cos(np.deg2rad(self.misalignment))
        rot_vel = 0.0

        cartesian_vel = np.array([x_vel, -z_vel, rot_vel])
        J = self._build_Jacobian(q)

        Jinv = np.linalg.inv(J)
        q_vel = np.dot(Jinv, cartesian_vel)

        return np.rad2deg(q_vel)
