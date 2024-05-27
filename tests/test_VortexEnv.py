import pytest

from vortex_gym.VortexEnv import VortexEnv
from pyvortex.vortex_interface import AppMode


@pytest.fixture(scope='class', params=[('config.vxc', 'Kinova Gen2 Unjamming/Scenes/kinova_peg-in-hole.vxscene')])
def env_files(request):
    conf_file, scene_file = request.param

    return (conf_file, scene_file)


@pytest.fixture(scope='class')
def vortex_env(env_files):
    config_file, content_file = env_files
    vortex_env = VortexEnv(config_file=config_file, content_file=content_file)

    return vortex_env


class TestVortexEnv:
    def test_init(self, vortex_env):
        assert isinstance(vortex_env, VortexEnv)

    def test_kinova_ctrl(self, vortex_env):
        print('Moving kinova arm')
        vortex_env.vx_interface.set_app_mode(AppMode.SIMULATING)
        vortex_env.vx_interface.app.pause(False)

        vortex_env.step()

    # test_rendering()

    # test_cameras

    # test_
