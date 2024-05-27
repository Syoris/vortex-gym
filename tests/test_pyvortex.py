"""
To check the Vortex environment and installation of pyvortex
"""

import math
import pytest

from vortex_gym import ASSETS_DIR

from pyvortex.vortex_env import VortexEnv
from pyvortex.vortex_classes import AppMode, VortexInterface


class VX_Inputs(VortexInterface):
    j2_vel_id: str = 'j2_vel_id'
    j4_vel_id: str = 'j4_vel_id'
    j6_vel_id: str = 'j6_vel_id'


class VX_Outputs(VortexInterface):
    j2_pos_real: str = 'j2_pos'
    j4_pos_real: str = 'j4_pos'
    j6_pos_real: str = 'j6_pos'
    j2_vel_real: str = 'j2_vel'
    j4_vel_real: str = 'j4_vel'
    j6_vel_real: str = 'j6_vel'
    j2_torque: str = 'j2_torque'
    j4_torque: str = 'j4_torque'
    j6_torque: str = 'j6_torque'
    hand_pos_rot: str = 'hand_pos_rot'
    socket_force: str = 'socket_force'
    socket_torque: str = 'socket_torque'
    plug_force: str = 'plug_force'
    plug_torque: str = 'plug_torque'


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


@pytest.fixture(scope='class')
def inputs_interface():
    return VX_Inputs()


@pytest.fixture(scope='class')
def outputs_interface():
    return VX_Outputs()


@pytest.fixture()
def vortex_env(env_files, inputs_interface, outputs_interface):
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
        inputs_interface=inputs_interface,
        outputs_interface=outputs_interface,
        viewpoints=['Global', 'Perspective'],
    )

    vortex_env.set_app_mode(AppMode.SIMULATING)
    vortex_env.app.pause(False)

    return vortex_env


class TestPyvortex:
    def test_init(self, vortex_env):
        assert isinstance(vortex_env, VortexEnv)
