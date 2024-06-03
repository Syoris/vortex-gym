import pytest
import numpy as np
from spatialmath import SE3
import gymnasium as gym

from vortex_gym.envs.Reach_Kinova_v1 import ReachKinovaV1


RENDER = False


@pytest.fixture(scope='class')
def kinova_env():
    kinova_env = ReachKinovaV1(render_mode='human' if RENDER else None)
    return kinova_env


def is_robot_home(peg_pose):
    peg_exp_t = [0.5289988386702168, -0.007, 0.08565138234155462]
    T_peg_exp = SE3(peg_exp_t) * SE3.RPY(np.deg2rad([0, 180, 0]), order='xyz')

    tol = 0.001  # 1mm
    assert np.allclose(peg_pose.t, T_peg_exp.t, atol=tol), f'Assertion failed: {peg_pose.t} != {T_peg_exp.t}'
    assert np.allclose(peg_pose.R, T_peg_exp.R, atol=0.01), f'Assertion failed: {peg_pose.R} != {T_peg_exp.R}'


class TestKinovaGen2:
    def test_init(self):
        kinova_env = ReachKinovaV1()

        # Check __init__ ok
        assert isinstance(kinova_env, ReachKinovaV1)

        # Check robot is home
        peg_pose = kinova_env.robot.peg_pose
        is_robot_home(peg_pose)

    def test_get_obs(self, kinova_env):
        obs = kinova_env._get_obs()

        # Check observation space
        assert 'angles' in obs
        assert 'velocities' in obs
        assert 'torques' in obs
        assert 'command' in obs

        # Check observation values
        assert obs['angles'].shape == (3,)
        assert obs['velocities'].shape == (3,)
        assert obs['torques'].shape == (3,)
        assert obs['command'].shape == (3,)

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

    # TODO
    # def test_render(self, kinova_env):
    #     kinova_env.reset()
    #     kinova_env.render()

    def test_env(self):
        """
        Test all the functions of the gym environment. No RL agent is used.
        """
        env = gym.make('vx_envs/ReachKinova-v1', render_mode='human')
        observation, info = env.reset()

        for _ in range(1000):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()
