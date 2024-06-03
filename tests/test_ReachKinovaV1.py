import pytest

from vortex_gym.envs.Reach_Kinova_v1 import ReachKinovaV1


class TestKinovaGen2:
    def test_init(self):
        kinova_env = ReachKinovaV1()

        # Check __init__ ok
        assert isinstance(kinova_env, ReachKinovaV1)

        # Check robot is home
