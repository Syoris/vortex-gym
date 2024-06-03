import math
import pytest
import yaml
import numpy as np

from spatialmath import SE3


from vortex_gym import ASSETS_DIR, ROBOT_CFG_DIR
from vortex_gym.robot.kinova_gen_2 import KinovaGen2

from pyvortex.vortex_env import VortexEnv
from pyvortex.vortex_classes import AppMode, VortexInterface


import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

RENDER_VX = True


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


@pytest.fixture(scope='function')
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
        render=RENDER_VX,
    )

    vortex_env.set_app_mode(AppMode.SIMULATING)
    vortex_env.app.pause(False)

    return vortex_env

    # vortex_env = None


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

        assert kinova_robot.joints.j2._vel_cmd == 0.0
        assert kinova_robot.joints.j4._vel_cmd == 0.0
        assert kinova_robot.joints.j6._vel_cmd == 0.0

    def test_set_input_list(self, vortex_env):
        kinova_robot = KinovaGen2(vortex_env)

        # Set joint velocities
        joints_vels = [0.1, 0.2, 0.3]
        kinova_robot.set_joints_vels(joints_vels)

        joints = kinova_robot.joints
        print(joints)

        assert joints.j2.vel_cmd == 0.1
        assert joints.j4.vel_cmd == 0.2
        assert joints.j6.vel_cmd == 0.3

    def test_set_input_dict(self, vortex_env):
        kinova_robot = KinovaGen2(vortex_env)

        # Set joint velocities
        joints_vels = {'j2': 1.1, 'j4': 1.2, 'j6': 1.3}
        kinova_robot.set_joints_vels(joints_vels)

        joints = kinova_robot.joints
        print(joints)

        assert joints.j2.vel_cmd == 1.1
        assert joints.j4.vel_cmd == 1.2
        assert joints.j6.vel_cmd == 1.3

    def test_set_input_moves(self, vortex_env):
        # Goals
        time = 2  # sec
        j2_goal = -45  # deg
        j4_goal = 10  # deg
        j6_goal = 90  # deg

        j2_speed = j2_goal / time
        j4_speed = j4_goal / time
        j6_speed = j6_goal / time

        # Set joint velocities
        kinova_robot = KinovaGen2(vortex_env)
        kinova_robot.set_joints_vels([j2_speed, j4_speed, j6_speed])

        # Start state
        start_state = kinova_robot.joints
        print('\n\n Start State:')
        print(start_state)
        start_time = vortex_env.sim_time

        # Sim for `time`
        for _ in range(int((time - start_time) / vortex_env.get_simulation_time_step())):
            sim_time = vortex_env.sim_time
            vortex_env.step()

        assert math.isclose(sim_time - start_time, time, abs_tol=0.1), f'Assertion failed: {sim_time} != {time}'

        # Check final state
        final_state = kinova_robot.joints
        print('\n\n Final State:')
        print(final_state)

        j2_angle = final_state.j2.angle
        j4_angle = final_state.j4.angle
        j6_angle = final_state.j6.angle

        assert math.isclose(j2_angle, j2_goal, abs_tol=2), f'Assertion failed: {j2_angle} != {j2_goal}'
        assert math.isclose(j4_angle, j4_goal, abs_tol=2), f'Assertion failed: {j4_angle} != {j4_goal}'
        assert math.isclose(j6_angle, j6_goal, abs_tol=2), f'Assertion failed: {j6_angle} != {j6_goal}'

    def test_go_to_angles(self, vortex_env):
        # Goals
        j2_goal = -45  # deg
        j4_goal = 10  # deg
        j6_goal = 90  # deg

        kinova_robot = KinovaGen2(vortex_env)

        kinova_robot.go_to_angles([j2_goal, j4_goal, j6_goal])

        # Check final state
        final_state = kinova_robot.joints
        print('\n\n Final State:')
        print(final_state)

        j2_angle = final_state.j2.angle
        j4_angle = final_state.j4.angle
        j6_angle = final_state.j6.angle

        assert math.isclose(j2_angle, j2_goal, abs_tol=2), f'Assertion failed: {j2_angle} != {j2_goal}'
        assert math.isclose(j4_angle, j4_goal, abs_tol=2), f'Assertion failed: {j4_angle} != {j4_goal}'
        assert math.isclose(j6_angle, j6_goal, abs_tol=2), f'Assertion failed: {j6_angle} != {j6_goal}'

    # TODO: Test for all 7 joints
    @pytest.mark.parametrize(
        'angles_goal',
        [
            [-45, 0, 0],
            [0, -45, 0],
            [0, 0, -45],
            [-45, -45, -45],
            [45, 45, 45],
        ],
    )
    def test_rbt_fkine(self, vortex_env, angles_goal):
        kinova_robot = KinovaGen2(vortex_env)
        kinova_robot.go_to_angles(angles_goal)

        # Get state
        angles = kinova_robot.joints.angles

        # EE pose from vortex
        vx_pose = kinova_robot.ee_pose
        vx_trans = vx_pose.t
        vx_rot = vx_pose.R

        # EE pose from forward kin
        angles_rad = [np.deg2rad(x) for x in angles]
        # kinova_robot.robot_model.plot(angles_rad, backend='pyplot')
        Te = kinova_robot.robot_model.fkine(angles_rad)

        fkine_trans = Te.t
        fkine_rot = Te.R

        tol = 0.001  # 1mm
        assert np.allclose(vx_trans, fkine_trans, atol=tol), f'Assertion failed: {vx_trans} != {fkine_trans}'
        assert np.allclose(vx_rot, fkine_rot, atol=tol), f'Assertion failed: {vx_rot} != {fkine_rot}'

        # # --- Uncomment to plot frames ---
        # plt.figure()  # create a new figure
        # kinova_robot.robot_model.plot(angles_rad, backend='pyplot')
        # SE3().plot(frame='0', dims=[-3, 3], color='black')
        # vx_pose.plot(frame='EE', dims=[-1, 1], color='red')
        # Te.plot(frame='ee', dims=[-1, 1], color='green')
        # ...

    @pytest.mark.parametrize(
        'pose_goal',  # ([x, y, z], [roll, pitch, yaw])
        [
            (
                [0.66475637, -0.0098, 0.93941385],
                [-180, -45, -180],
            ),  # [-3.14159265, -0.78603249, -3.14159265]
        ],
    )
    def test_rbt_ikine_sol(self, vortex_env, pose_goal):
        kinova_robot = KinovaGen2(vortex_env)
        rbt_model = kinova_robot.robot_model

        # TODO: Choose between angles or Tep as arguments to test
        Tep = rbt_model.fkine(np.deg2rad([0, -45, 0, 0, 0, 0, 0]))
        # Tep = SE3(pose_goal[0]) * SE3.RPY(np.deg2rad(pose_goal[1]), order='xyz')

        # IKine solution
        sol = rbt_model.ikine_LM(
            Tep, q0=[0, 0, 0, 0, 0, 0, 0], joint_limits=False, ilimit=1000, slimit=5000, mask=[1, 1, 1, 0, 0, 0]
        )
        print(f'Success: {sol.success}')
        print(sol.q)

        # Move robot to solution
        kinova_robot.go_to_angles([sol.q[1], sol.q[3], sol.q[5]], degrees=False)
        sim_pose = kinova_robot.ee_pose

        # # --- Uncomment to plot frames ---
        # rbt_model.plot(sol.q, backend='pyplot')
        # SE3().plot(frame='0', dims=[-3, 3], color='black')
        # sim_pose.plot(frame='sim', dims=[-1, 1], color='red')
        # Tep.plot(frame='Tep', dims=[-1, 1], color='green')

        # Assert
        tol = 0.001  # 1mm
        assert sol.success, 'IKine failed'
        assert np.allclose(sim_pose.t, Tep.t, atol=tol), f'Assertion failed: {sim_pose.t} != {Tep.t}'
        # assert np.allclose(sim_pose.R, Tep.R, atol=tol), f'Assertion failed: {sim_pose.R} != {Tep.R}'

    # TODO
    # def test_go_to_pose(self, vortex_env, pose_goal):
    #     ...
