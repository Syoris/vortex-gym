import math
import pytest

from vortex_gym.VortexEnv import VortexEnv, VX_Interface
from pyvortex.vortex_interface import AppMode


class VX_Inputs(VX_Interface):
    j2_vel_id: str = 'j2_vel_id'
    j4_vel_id: str = 'j4_vel_id'
    j6_vel_id: str = 'j6_vel_id'


class VX_Outputs(VX_Interface):
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
        config_file=config_file,
        content_file=content_file,
        inputs_interface=inputs_interface,
        outputs_interface=outputs_interface,
    )

    vortex_env.vx.set_app_mode(AppMode.SIMULATING)
    vortex_env.vx.app.pause(False)
    # vortex_env.step()

    return vortex_env


class TestVortexEnv:
    def test_init(self, vortex_env):
        assert isinstance(vortex_env, VortexEnv)

    def test_get_inputs(self, vortex_env, inputs_interface):
        val_j2 = vortex_env.vx.get_input(inputs_interface.j2_vel_id)
        val_j4 = vortex_env.vx.get_input(inputs_interface.j4_vel_id)
        val_j6 = vortex_env.vx.get_input(inputs_interface.j6_vel_id)

        assert val_j2 == 0.0
        assert val_j4 == 0.0
        assert val_j6 == 0.0

    def test_get_outputs(self, vortex_env, outputs_interface):
        val_j2_pos = vortex_env.vx.get_output(outputs_interface.j2_pos_real)
        val_j4_pos = vortex_env.vx.get_output(outputs_interface.j4_pos_real)
        val_j6_pos = vortex_env.vx.get_output(outputs_interface.j6_pos_real)

        assert math.isclose(val_j2_pos, 0, abs_tol=1e-5)
        assert math.isclose(val_j4_pos, 0, abs_tol=1e-5)
        assert math.isclose(val_j6_pos, 0, abs_tol=1e-5)

    def test_in_out(self, vortex_env, inputs_interface, outputs_interface):
        """Test the input and output interfaces. Set the joints velocities and check if they moved as expected."""
        time = 1  # sec
        j2_goal = -45  # deg
        j4_goal = 10  # deg
        j6_goal = 90  # deg

        # Inputs
        j2_speed = math.radians(j2_goal / time)  # rad/sec
        j4_speed = math.radians(j4_goal / time)  # rad/sec
        j6_speed = math.radians(j6_goal / time)  # rad/sec

        vortex_env.vx.set_input(inputs_interface.j2_vel_id, j2_speed)
        vortex_env.vx.set_input(inputs_interface.j4_vel_id, j4_speed)
        vortex_env.vx.set_input(inputs_interface.j6_vel_id, j6_speed)

        val_j2_vel_id = vortex_env.vx.get_input(inputs_interface.j2_vel_id)
        val_j4_vel_id = vortex_env.vx.get_input(inputs_interface.j4_vel_id)
        val_j6_vel_id = vortex_env.vx.get_input(inputs_interface.j6_vel_id)

        assert val_j2_vel_id == j2_speed
        assert val_j4_vel_id == j4_speed
        assert val_j6_vel_id == j6_speed

        # Outputs
        val_j2_pos = vortex_env.vx.get_output(outputs_interface.j2_pos_real)
        val_j4_pos = vortex_env.vx.get_output(outputs_interface.j4_pos_real)
        val_j6_pos = vortex_env.vx.get_output(outputs_interface.j6_pos_real)

        assert math.isclose(val_j2_pos, 0, abs_tol=1e-5)
        assert math.isclose(val_j4_pos, 0, abs_tol=1e-5)
        assert math.isclose(val_j6_pos, 0, abs_tol=1e-5)

        # Step for `time`
        sim_time = vortex_env.sim_time
        assert sim_time == 0.0

        for _ in range(int(time / vortex_env.h)):
            sim_time = vortex_env.sim_time
            vortex_env.step()

        sim_time = vortex_env.sim_time
        assert math.isclose(sim_time, time + 0.01, abs_tol=1e-5)

        # Check joints positions
        val_j2_pos = math.degrees(vortex_env.vx.get_output(outputs_interface.j2_pos_real))
        val_j4_pos = math.degrees(vortex_env.vx.get_output(outputs_interface.j4_pos_real))
        val_j6_pos = math.degrees(vortex_env.vx.get_output(outputs_interface.j6_pos_real))

        assert math.isclose(val_j2_pos, j2_goal, abs_tol=2), f'Assertion failed: {val_j2_pos} != {j2_goal}'
        assert math.isclose(val_j4_pos, j4_goal, abs_tol=2), f'Assertion failed: {val_j4_pos} != {j4_goal}'
        assert math.isclose(val_j6_pos, j6_goal, abs_tol=2), f'Assertion failed: {val_j6_pos} != {j6_goal}'

    # test_rendering()

    # test_cameras

    # test_
