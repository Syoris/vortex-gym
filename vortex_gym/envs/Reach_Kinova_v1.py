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
    def __init__(self, task_cfg: None = None):
        self.task_cfg = task_cfg

        # Vortex environment
        self._assets_dir = ASSETS_DIR
        self._config_file = 'config.vxc'
        self._content_file = 'Kinova Gen2 Unjamming/Scenes/kinova_peg-in-hole.vxscene'

        self._kinova_vx_in = KinovaVxIn()
        self._kinova_vx_out = KinovaVxOut()

        self._vortex_env = VortexEnv(
            assets_dir=ASSETS_DIR,
            config_file=self._config_file,
            content_file=self._content_file,
            inputs_interface=self._kinova_vx_in,
            outputs_interface=self._kinova_vx_out,
            viewpoints=['Global', 'Perspective'],
        )

        self._robot = KinovaGen2(self._vortex_env)

        # Init observation and action spaces
        self._init_spaces()

        # RL Hyperparameters

        # Initialize robot
        self._robot.go_home()

        ...

    def _init_spaces(self):
        # TODO: Get min/max from robot
        self.observation_space = spaces.Dict(
            {
                'angles': spaces.Box(low=-np.pi, high=np.pi, shape=(self._robot.n_joints,), dtype=np.float32),
                'velocities': spaces.Box(low=-np.inf, high=np.inf, shape=(self._robot.n_joints,), dtype=np.float32),
                'torques': spaces.Box(low=-np.inf, high=np.inf, shape=(self._robot.n_joints,), dtype=np.float32),
                'command': spaces.Box(low=-np.inf, high=np.inf, shape=(self._robot.n_joints,), dtype=np.float32),
            }
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # --------------------------------------------------------------------------------------------
    # MARK: Gym methods
    # --------------------------------------------------------------------------------------------
    def reset(self): ...

    def step(self, action): ...

    def render(self, mode='human'): ...

    def close(self): ...

    # --------------------------------------------------------------------------------------------
    # MARK: Utilities
    # --------------------------------------------------------------------------------------------
    def _get_obs(self):
        joints_states = self._robot.joints

        angles = joints_states.angles[1, 3, 5]
        vels = joints_states.vels[1, 3, 5]
        torques = joints_states.torques[1, 3, 5]
        vel_cmds = joints_states.vel_cmds[1, 3, 5]

        return {
            'angles': angles,
            'velocities': vels,
            'torques': torques,
            'command': vel_cmds,
        }
