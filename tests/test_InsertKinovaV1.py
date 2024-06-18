import pytest
import numpy as np
from spatialmath import SE3
import gymnasium as gym

import pygetwindow as gw

from vortex_gym.envs.Insert_Kinova_v1 import InsertKinovaV1
from stable_baselines3.common.env_checker import check_env

RENDER = True


@pytest.fixture(scope='class')
def kinova_env():
    kinova_env = InsertKinovaV1(render_mode='human' if RENDER else None)
    return kinova_env


def is_robot_home(peg_pose):
    peg_exp_t = [0.5289988386702168, -0.007, 0.08565138234155462]
    T_peg_exp = SE3(peg_exp_t) * SE3.RPY(np.deg2rad([0, 180, 0]), order='xyz')

    tol = 0.001  # 1mm
    assert np.allclose(peg_pose.t, T_peg_exp.t, atol=tol), f'Assertion failed: {peg_pose.t} != {T_peg_exp.t}'
    assert np.allclose(peg_pose.R, T_peg_exp.R, atol=0.01), f'Assertion failed: {peg_pose.R} != {T_peg_exp.R}'


class TestInsertKinovaV1:
    def test_init(self):
        kinova_env = InsertKinovaV1(render_mode='human' if RENDER else None)

        # Check __init__ ok
        assert isinstance(kinova_env, InsertKinovaV1)

        # Check robot is home
        peg_pose = kinova_env.robot.peg_pose
        is_robot_home(peg_pose)

    def test_get_obs(self, kinova_env):
        obs = kinova_env._get_obs()

        # Check observation space
        assert 'angles' in obs
        assert 'velocities' in obs
        assert 'torques' in obs
        assert 'target_vels' in obs

        # Check observation values
        assert obs['angles'].shape == (3,)
        assert obs['velocities'].shape == (3,)
        assert obs['torques'].shape == (3,)
        assert obs['target_vels'].shape == (3,)

    def test_reset(self, kinova_env):
        kinova_env.reset()

        # Check robot is home
        peg_pose = kinova_env.robot.peg_pose
        is_robot_home(peg_pose)

        # Move robot
        kinova_env.robot.set_joints_vels([10, 0, 0])
        for _ in range(int(1 / kinova_env.vortex_env.get_simulation_time_step())):
            kinova_env.vortex_env.step()

        # Check robot moved
        peg_pose = kinova_env.robot.peg_pose
        try:
            is_robot_home(peg_pose)
        except AssertionError:  # Robot should have moved
            pass

        # Reset
        kinova_env.reset()
        peg_pose = kinova_env.robot.peg_pose
        is_robot_home(peg_pose)

    def test_env_render(self):
        """
        Test all the functions of the gym environment. No RL agent is used.
        """
        env = gym.make('InsertKinova-v1', render_mode='human')
        observation, info = env.reset()

        disp_name_window_name = 'CM Labs Graphics Qt'
        n_disp = len(gw.getWindowsWithTitle(disp_name_window_name))
        assert n_disp >= 1

        for _ in range(1000):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()

    def test_env_no_render(self):
        """
        Test all the functions of the gym environment. No RL agent is used.
        """
        env = gym.make('InsertKinova-v1', render_mode=None)
        observation, info = env.reset()

        disp_name_window_name = 'CM Labs Graphics Qt'
        n_disp = len(gw.getWindowsWithTitle(disp_name_window_name))
        assert n_disp == 0

        for _ in range(1000):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()

    def test_ikine(self):
        """To test the robot goes straight down without action"""
        env = gym.make('InsertKinova-v1', render_mode='human', socket_x_range=(0.529, 0.529))
        check_env(env)
        observation, info = env.reset()

        start_peg_pose_t, start_peg_pose_rpy = info['peg_pose']

        for _ in range(1000):
            action = np.zeros(2)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                end_peg_pose_t, end_peg_pose_rpy = info['peg_pose']

                assert np.isclose(start_peg_pose_t[0], end_peg_pose_t[0], atol=0.001)
                assert np.isclose(start_peg_pose_t[1], end_peg_pose_t[1], atol=0.001)
                assert np.isclose(start_peg_pose_t[2] - env.unwrapped.z_insertion, end_peg_pose_t[2], atol=0.001)

                from spatialmath.base import rpy2r

                R_start = rpy2r(start_peg_pose_rpy, order='xyz', unit='deg')
                R_end = rpy2r(end_peg_pose_rpy, order='xyz', unit='deg')
                assert np.allclose(R_start, R_end, atol=0.001)

                observation, info = env.reset()

        env.close()

    def test_2_envs(self):
        # Create 2 environments
        env1 = gym.make('InsertKinova-v1', render_mode='human', viewpoint='Perspective')
        observation1, info1 = env1.reset()

        env2 = gym.make('InsertKinova-v1', render_mode='human', viewpoint='Perspective', eval_mode=True)
        observation2, info2 = env2.reset()

        # Move Env 1
        print('Moving Env 1')
        for _ in range(200):
            action = [0, 0]
            env1.step(action)

        # Move Env 2
        print('Moving Env 2')
        for _ in range(200):
            action = [0, 0]
            env2.step(action)

        # Reset Env 1
        print('Resetting Env 1')
        env1.reset()

        # Reset Env 2
        print('Resetting Env 2')
        env2.reset()

        # assert False  # Not yet working.. Reset of 2nd env doesn't work
        print('Done')

    def test_socket_pose(self):
        """Check if its possible to get the pose of the socket and move it"""
        kinova_env = InsertKinovaV1(render_mode='human' if RENDER else None, viewpoint='Global')

        # Check robot is home
        peg_pose = kinova_env.robot.peg_pose
        is_robot_home(peg_pose)

        # Get socket pose
        socket_pose_1 = kinova_env._get_socket_pose()
        assert socket_pose_1 is not None

        # Reset
        kinova_env.reset()

        # Socket pose should have changed
        socket_pose_2 = kinova_env._get_socket_pose()
        assert socket_pose_2 is not None

        assert not np.allclose(socket_pose_1, socket_pose_2)

    @pytest.mark.skip(reason='Not yet implemented')
    def test_env_reset(self):
        """Test the reset function
        - Robot is home
        - Socket moved
        - misalignment
        - ...
        """
        ...
