"""
To check the Vortex environment and installation of pyvortex
"""

# import math
import pytest

from vortex_gym import ASSETS_DIR

from pyvortex.vortex_env import VortexEnv
from pyvortex.vortex_classes import AppMode


@pytest.fixture(scope='class', params=[('config.vxc', 'Kinova Gen2 Unjamming/Scenes/kinova_peg-in-hole.vxscene')])
def env_files(request):
    """Config and scene files for the Vortex environment

    Args:
        request (tuple): (config, scene)

    Returns:
        tuple: (config, scene)
    """
    conf_file, scene_file = request.param

    return (conf_file, scene_file)


@pytest.fixture()
def vortex_env(env_files):
    """Create a Vortex environment from the env_files fixture

    Args:
        env_files (tuple): config and scene files

    Returns:
        VortexEnv: Vortex environment
    """
    config_file, content_file = env_files
    vortex_env = VortexEnv(
        assets_dir=ASSETS_DIR,
        config_file=config_file,
        content_file=content_file,
        viewpoints=['Global', 'Perspective'],
    )

    vortex_env.set_app_mode(AppMode.SIMULATING)
    vortex_env.app.pause(False)

    return vortex_env


class TestPyvortex:
    def test_init(self, vortex_env):
        assert isinstance(vortex_env, VortexEnv)
