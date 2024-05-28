import pytest
import yaml

from vortex_gym import ASSETS_DIR, ROBOT_CFG_DIR
from vortex_gym.robot.kinova_gen_2 import KinovaGen2

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


@pytest.fixture()
def vortex_env():
    """Create a Vortex environment from the env_files fixture

    Args:
        env_files (tuple): config and scene files

    Returns:
        VortexEnv: Vortex environment
    """
    config_file, content_file = ('config.vxc', 'Kinova Gen2 Unjamming/Scenes/kinova_peg-in-hole.vxscene')
    inputs_interface = VX_Inputs()
    outputs_interface = VX_Outputs()

    vortex_env = VortexEnv(
        assets_dir=ASSETS_DIR,
        config_file=config_file,
        content_file=content_file,
        inputs_interface=inputs_interface,
        outputs_interface=outputs_interface,
        viewpoints=['Global'],
    )

    vortex_env.set_app_mode(AppMode.SIMULATING)
    vortex_env.app.pause(False)

    return vortex_env


class TestKinovaGen2:
    def test_init(self, vortex_env):
        kinova_robot = KinovaGen2(vortex_env)

        # Check __init__ ok
        assert isinstance(kinova_robot, KinovaGen2)

    def test_load_config(self, vortex_env):
        kinova_robot = KinovaGen2(vortex_env)

        # Check .yaml file loaded
        robot_cfg = kinova_robot.robot_cfg

        # Load expected config file
        with open(ROBOT_CFG_DIR / 'KinovaGen2.yaml', 'r') as file:
            expected_cfg = yaml.safe_load(file)

        assert robot_cfg == expected_cfg

    def test_update_state(self, vortex_env):
        kinova_robot = KinovaGen2(vortex_env)

        joints = kinova_robot.joints
        print(joints)

        assert kinova_robot.joints.j2.angle != 0.0
        assert kinova_robot.joints.j4.angle != 0.0
        assert kinova_robot.joints.j6.angle != 0.0

        assert kinova_robot.joints.j2.vel != 0.0
        assert kinova_robot.joints.j4.vel != 0.0
        assert kinova_robot.joints.j6.vel != 0.0

        assert kinova_robot.joints.j2.torque != 0.0
        assert kinova_robot.joints.j4.torque != 0.0
        assert kinova_robot.joints.j6.torque != 0.0

        assert kinova_robot.joints.j2.vel_cmd == 0.0
        assert kinova_robot.joints.j4.vel_cmd == 0.0
        assert kinova_robot.joints.j6.vel_cmd == 0.0

    # test set_state

    # go_to_pose

    # go_to_angles
