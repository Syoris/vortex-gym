import math
import pytest
import yaml
import numpy as np


from vortex_gym import ASSETS_DIR, ROBOT_CFG_DIR
from vortex_gym.robot.kinova_gen_2 import KinovaGen2, Joint, KinovaGen2Joints

from pyvortex.vortex_env import VortexEnv
from pyvortex.vortex_classes import AppMode


import matplotlib
import matplotlib.pyplot as plt

from spatialmath import SE3

matplotlib.use('TkAgg')

RENDER_VX = False
PLOT = False


@pytest.fixture(scope='class')
def vortex_env():
    """Create a Vortex environment from the env_files fixture

    Args:
        env_files (tuple): config and scene files

    Returns:
        VortexEnv: Vortex environment
    """
    config_file, content_file = ('config.vxc', 'Kinova Gen2 Unjamming/Scenes/kinova_peg-in-hole.vxscene')

    vortex_env = VortexEnv(
        assets_dir=ASSETS_DIR,
        config_file=config_file,
        content_file=content_file,
        viewpoints=['Global', 'Perspective'],
        render=RENDER_VX,
    )

    vortex_env.set_app_mode(AppMode.SIMULATING)
    vortex_env.app.pause(False)

    yield vortex_env

    vortex_env = None


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

        j2_angle_start = start_state.j2.angle
        j4_angle_start = start_state.j4.angle
        j6_angle_start = start_state.j6.angle

        # Sim for `time`
        for _ in range(int(time / vortex_env.get_simulation_time_step())):
            sim_time = vortex_env.sim_time
            vortex_env.step()

        assert math.isclose(sim_time - start_time, time, abs_tol=0.1), f'Assertion failed: {sim_time} != {time}'

        # Check final state
        final_state = kinova_robot.joints
        print('\n\n Final State:')
        print(final_state)

        j2_angle_final = final_state.j2.angle
        j4_angle_final = final_state.j4.angle
        j6_angle_final = final_state.j6.angle

        assert math.isclose(
            j2_angle_final - j2_angle_start, j2_goal, abs_tol=2
        ), f'Assertion failed: {-(j2_angle_start - j2_angle_final)} != {j2_goal}'
        assert math.isclose(
            j4_angle_final - j4_angle_start, j4_goal, abs_tol=2
        ), f'Assertion failed: {-(j4_angle_start - j4_angle_final)} != {j4_goal}'
        assert math.isclose(
            j6_angle_final - j6_angle_start, j6_goal, abs_tol=2
        ), f'Assertion failed: {-(j6_angle_start - j6_angle_final)} != {j6_goal}'

    def test_KinovaGen2Joints(self, vortex_env):
        """Test KinovaGen2Joints class. Check if attributes returned are correct"""
        kinova_robot = KinovaGen2(vortex_env)

        angles = kinova_robot.joints.angles
        vels = kinova_robot.joints.vels
        torques = kinova_robot.joints.torques
        vel_cmds = kinova_robot.joints.vels_cmds
        joints_list = kinova_robot.joints.joints_list

        assert isinstance(kinova_robot.joints, KinovaGen2Joints)

        assert isinstance(angles, list)
        assert len(angles) == 7

        assert isinstance(vels, list)
        assert len(vels) == 7

        assert isinstance(torques, list)
        assert len(torques) == 7

        assert isinstance(vel_cmds, list)
        assert len(vel_cmds) == 7

        assert isinstance(joints_list, list)
        assert isinstance(joints_list[0], Joint)
        assert len(joints_list) == 7

    def test_go_to_angles(self, vortex_env):
        # Goals
        j2_goal = 135  # deg
        j4_goal = 190  # deg
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

    def test_go_home(self, vortex_env):
        kinova_robot = KinovaGen2(vortex_env)
        kinova_robot.go_home()

        sim_angles = kinova_robot.joints.angles
        sim_angles = [sim_angles[1], sim_angles[3], sim_angles[5]]
        expected_angles = kinova_robot.home_angles

        ang_tol = 2  # deg
        assert np.allclose(
            sim_angles, expected_angles, atol=ang_tol
        ), f'Assertion failed: {sim_angles} != {expected_angles}'

    @pytest.mark.parametrize(
        'angles_goal',
        [
            [180, 180, 0],
            [-45, 0, 0],
            [0, -45, 0],
            [0, 0, -45],
            [-45, -45, -45],
            [45, 45, 45],
        ],
    )
    def test_rbt_fkine_3dof(self, vortex_env, angles_goal):
        kinova_robot = KinovaGen2(vortex_env)
        kinova_robot.go_to_angles(angles_goal)
        vortex_env.step()

        # Get state
        angles = kinova_robot.joints.angles

        # EE pose from vortex
        vx_pose = kinova_robot.ee_pose
        vx_trans = vx_pose.t
        vx_rot = vx_pose.R

        # EE pose from forward kin
        angles_rad = np.deg2rad(angles)[[1, 3, 5]]
        if PLOT:
            kinova_robot.robot_model.plot(angles_rad, backend='pyplot')
        Te = kinova_robot.robot_model.fkine(angles_rad)

        fkine_trans = Te.t
        fkine_rot = Te.R

        # --- Uncomment to plot frames ---
        if PLOT:
            kinova_robot.robot_model.plot(angles_rad, backend='pyplot')
            SE3().plot(frame='0', dims=[-3, 3], color='black')
            vx_pose.plot(frame='EE', dims=[-1, 1], color='red')
            Te.plot(frame='ee', dims=[-1, 1], color='green')

        tol = 0.001  # 1mm
        assert np.allclose(vx_trans, fkine_trans, atol=tol), f'Assertion failed: {vx_trans} != {fkine_trans}'
        assert np.allclose(vx_rot, fkine_rot, atol=tol), f'Assertion failed: {vx_rot} != {fkine_rot}'

    @pytest.mark.skip(reason='Not implemented yet')
    def test_rbt_fkine_7dof(self, vortex_env, angles_goal): ...

    @pytest.mark.parametrize(
        'pose_goal',  # ([x, y, z], [roll, pitch, yaw])
        [
            [[0.65, -0.0098, 0.75], [0, 90, 90]],  # Facing forward (peg down)
            [[0.65, -0.0098, 0.3], [-90, 0, 180]],  # Facing down
        ],
    )
    def test_rbt_ikine_sol_3dof(self, vortex_env, pose_goal):
        kinova_robot = KinovaGen2(vortex_env)
        rbt_model = kinova_robot.robot_model

        # Tep = rbt_model.fkine(np.deg2rad([0, -45, 0, 0, 0, 0, 0]))
        T_goal = SE3.Trans(pose_goal[0]) * SE3.RPY(pose_goal[1], order='xyz', unit='deg')

        # IKine solution
        sol = rbt_model.ikine_LM(T_goal, q0=[0, 0, 0], joint_limits=False)
        print(f'Success: {sol.success}')
        print(sol.q)

        # Move robot to solution
        kinova_robot.go_to_angles(list(sol.q), degrees=False)
        sim_pose = kinova_robot.ee_pose

        # --- Uncomment to plot frames ---
        if PLOT:
            rbt_model.plot(sol.q, backend='pyplot')
            SE3().plot(frame='0', dims=[-3, 3], color='black')
            sim_pose.plot(frame='sim', dims=[-1, 1], color='red')
            T_goal.plot(frame='goal', dims=[-1, 1], color='green')
            plt.show(block=False)

        # Assert
        assert sol.success, 'IKine failed'

        tol = 0.01  # 1cm
        assert np.allclose(sim_pose.t, T_goal.t, atol=tol), f'Assertion failed: {sim_pose.t} != {T_goal.t}'
        assert np.allclose(sim_pose.R, T_goal.R, atol=tol), f'Assertion failed: {sim_pose.R} != {T_goal.R}'


"""
Tests requiring a new environment
"""


def test_peg_pose(vortex_env):
    kinova_robot = KinovaGen2(vortex_env)

    peg_pose = kinova_robot.peg_pose
    # print(peg_pose)

    T_peg_exp = SE3([0.160, -0.007, 1.215]) * SE3.RPY(np.deg2rad([0, 90, 0]), order='xyz')

    tol = 0.001  # 1mm
    assert np.allclose(peg_pose.t, T_peg_exp.t, atol=tol), f'Assertion failed: {peg_pose.t} != {T_peg_exp.t}'
    assert np.allclose(peg_pose.R, T_peg_exp.R, atol=tol), f'Assertion failed: {peg_pose.R} != {T_peg_exp.R}'


@pytest.mark.parametrize(
    'pose_goal',  # [[x, y, z], [roll, pitch, yaw], [j2_exp, j4_exp, j6_exp]]
    [
        [[0.55, -0.0098, 0.25], [0, 90, 90], [[133.76, -52.71, -83.51]]],  # Top of socket,  Facing forward (peg down)
        [[0.75, -0.0098, 0.25], [0, 90, 90], [122.95, -94.03, -53.01]],  # Top of socket,  Facing forward (peg down)
        [[0.55, -0.0098, 0.35], [-90, 0, 180], [143.9, -118.9, 82.0]],  # Top of socket,  Facing forward (peg down)
    ],
)
def test_go_to_pose(vortex_env, pose_goal):
    T_goal = SE3.Trans(pose_goal[0]) * SE3.RPY(pose_goal[1], order='xyz', unit='deg')
    expected_angles = pose_goal[2]

    kinova_robot = KinovaGen2(vortex_env)
    kinova_robot.go_home()

    # Move robot to goal pose
    kinova_robot.go_to_pose(pose=pose_goal[0], orientation=pose_goal[1], raise_exception=True)

    # Get current pose
    sim_pose = kinova_robot.ee_pose
    sim_angles = kinova_robot.joints.angles
    sim_angles = [sim_angles[1], sim_angles[3], sim_angles[5]]

    tol = 0.01  # 1cm
    assert np.allclose(sim_pose.t, T_goal.t, atol=tol), f'Assertion failed: {sim_pose.t} != {T_goal.t}'
    assert np.allclose(sim_pose.R, T_goal.R, atol=tol), f'Assertion failed: {sim_pose.R} != {T_goal.R}'

    ang_tol = 2  # deg
    assert np.allclose(
        sim_angles, expected_angles, atol=ang_tol
    ), f'Assertion failed: {sim_angles} != {expected_angles}'
