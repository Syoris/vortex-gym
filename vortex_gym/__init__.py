from pathlib import Path
from gymnasium.envs.registration import register

ASSETS_DIR = Path(__file__).parent / 'assets'
ROBOT_CFG_DIR = Path(__file__).parent / 'robot' / 'cfg'

register(
    id='InsertKinova-v1',
    entry_point='vortex_gym.envs:InsertKinovaV1',
)
