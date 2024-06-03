import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pyvortex.vortex_env import VortexEnv
from vortex_gym.robot.kinova_gen_2 import KinovaGen2, KinovaVxIn, KinovaVxOut
from vortex_gym import ASSETS_DIR


class ReachKinovaV1(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    # --------------------------------------------------------------------------------------------
    # MARK: Initialization
    # --------------------------------------------------------------------------------------------
    def __init__(self, render_mode=None, task_cfg: None = None):
        self.task_cfg = task_cfg

        # Vortex environment
        self._assets_dir = ASSETS_DIR
        self._config_file = 'config.vxc'
        self._content_file = 'Kinova Gen2 Unjamming/Scenes/kinova_peg-in-hole.vxscene'

        self._kinova_vx_in = KinovaVxIn()
        self._kinova_vx_out = KinovaVxOut()

        self.vortex_env = VortexEnv(
            assets_dir=ASSETS_DIR,
            config_file=self._config_file,
            content_file=self._content_file,
            viewpoints=['Global', 'Perspective'],
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

        self.action_coeff = 0.01

        # Initialize robot
        self.robot.go_home()
        self.vortex_env.save_current_frame()

        # Render
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.reset()

        print('KinovaGen2 environment initialized')

    def _init_spaces(self):
        # TODO: Get min/max obs from robot
        self.observation_space = spaces.Dict(
            {
                'angles': spaces.Box(low=-np.pi, high=np.pi, shape=(self.robot.n_joints,), dtype=np.float32),
                'velocities': spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot.n_joints,), dtype=np.float32),
                'torques': spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot.n_joints,), dtype=np.float32),
                'command': spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot.n_joints,), dtype=np.float32),
            }
        )

        # TODO: Get min/max action from robot
        # self.act_low_bound = np.array([self.robot_cfg.actuators.j2.torque_min, self.robot_cfg.actuators.j6.torque_min])
        # self.act_high_bound = np.array([self.robot_cfg.actuators.j2.torque_max, self.robot_cfg.actuators.j6.torque_max])
        self._actuator_low_bound = np.array([-30.5, -6.8])
        self._actuator_high_bound = np.array([30.5, 6.8])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # --------------------------------------------------------------------------------------------
    # MARK: Gym methods
    # --------------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset vortex
        self.vortex_env.reset_saved_frame()

        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        self.step_count = 0
        self.ep_completed = False

        # if self.render_mode == 'human':
        #     self._render_frame()

        return observation, info

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

        # self.ik_joints_vels = self._get_ik_vels(self.insertz, self.step_count, step_types=self.insertion_steps)
        self.ik_joints_vels = [5.0, 5.0, 5.0]

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

    def render(self, mode='human'): ...

    def close(self): ...

    # --------------------------------------------------------------------------------------------
    # MARK: Utilities
    # --------------------------------------------------------------------------------------------
    def _get_obs(self) -> dict:
        joints_states = self.robot.joints

        angles = np.array([joints_states.angles[1], joints_states.angles[3], joints_states.angles[5]])
        vels = np.array([joints_states.vels[1], joints_states.vels[3], joints_states.vels[5]])
        torques = np.array([joints_states.torques[1], joints_states.torques[3], joints_states.torques[5]])
        vels_cmds = np.array([joints_states.vels_cmds[1], joints_states.vels_cmds[3], joints_states.vels_cmds[5]])

        return {
            'angles': angles,
            'velocities': vels,
            'torques': torques,
            'command': vels_cmds,
        }

    def _get_info(self) -> dict:
        """Get additional information about the environment.

        - command (np.array): The command sent to the robot [j2, j4, j6]
        - peg_force (np.array): The force applied to the peg [fx, fy, fz]
        - peg_torque (np.array): The torque applied to the peg [tx, ty, tz]
        - peg_pose ((np.array, np.array)): The pose of the tool ([x, y, z], [roll, pitch, yaw])
        - ee_pose ((np.array, np.array)): The pose of the end-effector ([x, y, z], [roll, pitch, yaw])
        # - insertion_depth (float): The depth of the peg in the hole

        Returns:
            dict: _description_
        """
        ee_pose = self.robot.ee_pose
        peg_pose = self.robot.peg_pose

        info_dict = {
            'command': self.command,
            'peg_force': self.robot.get_peg_force(),
            'peg_torque': self.robot.get_peg_torque(),
            'peg_pose': (peg_pose.t, peg_pose.rpy(order='xyz', unit='deg')),
            'ee_pose': (ee_pose.t, ee_pose.rpy(order='xyz', unit='deg')),
            # 'insertion_depth': self.robot.get_insertion_depth(),
        }

        return info_dict

    def _compute_reward(self) -> float:
        return 0.0
