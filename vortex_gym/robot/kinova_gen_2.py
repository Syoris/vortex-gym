import time
from gymnasium import spaces
import numpy as np
import logging
from omegaconf import OmegaConf
from typing import Union

from pyvortex.vortex_env import VortexEnv
from pyvortex.vortex_classes import AppMode, VortexInterface
# from peg_in_hole.settings import app_settings

from vortex_gym.robot.robot_base import RobotBase
from vortex_gym import ROBOT_CFG_DIR

logger = logging.getLogger(__name__)
# robot_logger = logging.getLogger('robot_state')


""" Vortex Scene Inputs, Outputs and Parameters"""


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


VX_IN = VX_Inputs()
VX_OUT = VX_Outputs()


""" Kinova Robot Interface """


class KinovaGen2(RobotBase):
    def __init__(self, vx_env: VortexEnv):
        init_start_time = time.time()

        self.vx_env = vx_env

        """Load config"""
        self._get_robot_config()

        # General params
        self.sim_time = 0.0
        self.step_count = 0  # Step counter

        self._joints = KinovaGen2Joints(vx_env)
        self._init_joints()

        self._init_obs_space()

        self._init_action_space()

        # Controller
        self.controller = PIDController(kp=0.6, ki=0.2, kd=0.1, dt=self.vx_env.h, n_joints=3)

        """ Robot Parameters """
        # Link lengths
        self.L12 = self.robot_cfg.links_length.L12  # from base to joint 2, [m] 0.2755
        self.L34 = self.robot_cfg.links_length.L34  # from joint 2 to joint 4, [m], 0.410
        self.L56 = self.robot_cfg.links_length.L56  # from joint 4 to joint 6, [m], 0.3111
        self.L78 = self.robot_cfg.links_length.L78  # from joint 6 to edge of EE where peg is attached, [m], 0.2188
        self.Ltip = self.robot_cfg.links_length.Ltip  # from edge of End-Effector to tip of peg, [m], 0.16

        # # Set parameters values. Done after going home so its not limited by joint torques
        # self.vx_env.set_app_mode(AppMode.EDITING)
        # for field, field_value in {**self.joints_range, **self.forces_range}.items():
        #     self.vx_env.set_parameter(field, field_value)

        print(f'KinovaGen2 initialized. Time: {time.time() - init_start_time} sec')

    def _get_robot_config(self):
        """Load robot config from .yaml file"""
        config_path = ROBOT_CFG_DIR / 'KinovaGen2.yaml'
        self.robot_cfg = OmegaConf.load(config_path)

    def _init_joints(self):
        """Initialize joints"""
        for joint, joint_cfg in self.robot_cfg.actuators.items():
            self._joints.__dict__[joint].position_max = joint_cfg.position_max
            self._joints.__dict__[joint].position_min = joint_cfg.position_min
            self._joints.__dict__[joint].vel_max = joint_cfg.vel_max
            self._joints.__dict__[joint].vel_min = joint_cfg.vel_min
            self._joints.__dict__[joint].torque_max = joint_cfg.torque_max
            self._joints.__dict__[joint].torque_min = joint_cfg.torque_min

    def _init_obs_space(self):
        """Observation space (12 observations: position, vel, ideal vel, torque, for each of 3 joints)"""
        self.obs = np.zeros(9)

        # Observation: [joint_positions, joint_velocities, joint_ideal_vel, joint_torques]
        # Each one is 1x(n_joints). Total size: 4*(n_joints)
        pos_min = [np.deg2rad(act.position_min) for act in self.robot_cfg.actuators.values()]
        pos_max = [np.deg2rad(act.position_max) for act in self.robot_cfg.actuators.values()]
        vel_min = [np.deg2rad(act.vel_min) for act in self.robot_cfg.actuators.values()]
        vel_max = [np.deg2rad(act.vel_max) for act in self.robot_cfg.actuators.values()]
        torque_min = [act.torque_min for act in self.robot_cfg.actuators.values()]
        torque_max = [act.torque_max for act in self.robot_cfg.actuators.values()]

        # Minimum and Maximum joint position limits (in deg)
        self.joints_range = {}
        self.joints_range['j2_pos_min'], self.joints_range['j2_pos_max'] = (
            pos_min[0],
            pos_max[0],
        )
        self.joints_range['j4_pos_min'], self.joints_range['j4_pos_max'] = (
            pos_min[1],
            pos_max[1],
        )
        self.joints_range['j6_pos_min'], self.joints_range['j6_pos_max'] = (
            pos_min[2],
            pos_max[2],
        )

        # Minimum and Maximum joint force/torque limits (in N*m)
        self.forces_range = {}
        self.forces_range['j2_torque_min'], self.forces_range['j2_torque_max'] = (
            torque_min[0],
            torque_max[0],
        )
        self.forces_range['j4_torque_min'], self.forces_range['j4_torque_max'] = (
            torque_min[1],
            torque_max[1],
        )
        self.forces_range['j6_torque_min'], self.forces_range['j6_torque_max'] = (
            torque_min[2],
            torque_max[2],
        )

        obs_low_bound = np.concatenate((pos_min, vel_min, vel_min, torque_min))
        obs_high_bound = np.concatenate((pos_max, vel_max, vel_max, torque_max))

        self.observation_space = spaces.Box(
            low=obs_low_bound,
            high=obs_high_bound,
            dtype=np.float32,
        )

    def _init_action_space(self):
        """Action space (2 actions: j2 aug, j6, aug)"""
        self.action = np.array([0.0, 0.0])  # Action outputed by RL
        self.command = np.array([0.0, 0.0, 0.0])  # Vel command to send to the robot
        self.next_j_vel = np.zeros(3)  # Next target vel
        self.prev_j_vel = np.zeros(3)  # Prev target vel

        self.act_low_bound = np.array(
            [
                self.robot_cfg.actuators.j2.torque_min,
                self.robot_cfg.actuators.j6.torque_min,
            ]
        )
        self.act_high_bound = np.array(
            [
                self.robot_cfg.actuators.j2.torque_max,
                self.robot_cfg.actuators.j6.torque_max,
            ]
        )

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32,
        )

    """ Actions """

    def go_home(self):
        """
        To bring the peg on top of the hole
        """
        self.vx_env.set_app_mode(AppMode.SIMULATING)
        self.vx_env.app.pause(False)

        """ Phase 1 """
        # Set joint velocities to initialize
        self.update_sim()

        j4_vel_id = (np.pi / 180.0) * 90.0 / self.t_init_step
        # j6_vel_id = self.insertion_misalign / self.t_init_step
        self.command = np.array([0.0, j4_vel_id, 0.0])
        # print(f'Start vel: {self._readJvel()}')

        # Step the Vortex simulation
        for i in range(self.init_steps):
            self.update_sim()

        # Read reference position and rotation
        th_current = self._update_joints_angles()
        pos_current = self._read_tips_pos_fk(th_current)
        print('X after Phase 1:')
        print(pos_current[0])

        # Phase 1 pause
        self.command = np.array([0.0, 0.0, 0.0])

        for i in range(self.pause_steps):
            self.update_sim()

        """ Phase 2 (move downwards quickly and also make the tips aligned with the hole) """
        for i in range(self.pre_insert_steps):
            th_current = self._update_joints_angles()
            self.cur_j_vel = self._get_ik_vels(self.pre_insertz, i, step_types=self.pre_insert_steps)
            self.command = self.cur_j_vel

            self.update_sim()

        th_current = self._update_joints_angles()
        pos_current = self._read_tips_pos_fk(th_current)
        print('X, Z, rot after Phase 2:')
        print(f'{pos_current[0]}, {pos_current[1]}, {pos_current[2]}')

        # Phase 2 pause
        self.command = np.array([0.0, 0.0, 0.0])

        for i in range(self.pause_steps):
            self.update_sim()

    def go_to_angles(self, target_angles: Union[list, dict]):
        """
        Moves the robot arm to the specified target angles.

        Args:
            target_angles (Union[list, dict]): The target angles to move the robot arm to.

        Returns:
            None
        """
        if isinstance(target_angles, dict):
            target_angles = [target_angles['j2'], target_angles['j4'], target_angles['j6']]
        elif not isinstance(target_angles, list):
            raise TypeError('target_angles must be a list or dict')

        if len(target_angles) != 3:
            raise ValueError('target_angles must have 3 elements')

        self.vx_env.set_app_mode(AppMode.SIMULATING)
        self.vx_env.app.pause(False)

        current_state = self.joints
        start_time = self.vx_env.sim_time
        print(f'Start time: {start_time}')
        print(f'Start state: \n{current_state}')

        goal_reached = False
        while not goal_reached:
            current_state = self.joints
            current_angles = [current_state.j2.angle, current_state.j4.angle, current_state.j6.angle]

            vels = self.controller.compute_velocities(current_angles, target_angles)

            self.set_joints_vels(vels)

            self.vx_env.step()

            if np.allclose(current_angles, target_angles, atol=0.1):
                goal_reached = True

        print(f'Operation time: {self.vx_env.sim_time - start_time}')
        print(f'End state: \n{current_state}')

    """ Vortex interface functions """

    @property
    def joints(self):
        """Update robot states (joints angles, velocities, torques) from Vortex"""

        for joint in self._joints.__dict__.values():
            joint.update_state()

        return self._joints

    def set_joints_vels(self, target_vels: Union[list, dict]):
        """Set joint velocities"""
        if isinstance(target_vels, list):
            self.joints.j2.vel_cmd = target_vels[0]
            self.joints.j4.vel_cmd = target_vels[1]
            self.joints.j6.vel_cmd = target_vels[2]

        elif isinstance(target_vels, dict):
            self.joints.j2.vel_cmd = target_vels['j2']
            self.joints.j4.vel_cmd = target_vels['j4']
            self.joints.j6.vel_cmd = target_vels['j6']

        else:
            raise TypeError('target_vels must be a list or dict')

    def _get_plug_force(self) -> np.array:
        """Read plug force

        Returns:
            np.array(1x3): [x, y, z]
        """
        plug_force = self.vx_env.get_output(VX_OUT.plug_force)
        return np.array([plug_force.x, plug_force.y, plug_force.z])

    def _get_plug_torque(self) -> np.array:
        """Read plug torque

        Returns:
            np.array: [x, y, z]
        """
        plug_torque = self.vx_env.get_output(VX_OUT.plug_torque)
        return np.array([plug_torque.x, plug_torque.y, plug_torque.z])

    def _send_joint_target_vel(self, target_vels):
        self.vx_env.set_input(VX_IN.j2_vel_id, target_vels[0])
        self.vx_env.set_input(VX_IN.j4_vel_id, target_vels[1])
        self.vx_env.set_input(VX_IN.j6_vel_id, target_vels[2])

    """ Control """

    def _get_ik_vels(self, down_speed, cur_count, step_types):
        th_current = self._update_joints_angles()
        current_pos = self._read_tips_pos_fk(th_current)
        if step_types == self.pre_insert_steps:
            x_set = current_pos[0] - self.xpos_hole
            x_vel = -x_set / (self.h)
            z_vel = down_speed / (step_types * self.h)

        elif step_types == self.insertion_steps:
            vel = down_speed / (step_types * self.h)
            x_vel = vel * np.sin(np.deg2rad(self.insertion_misalign))
            z_vel = vel * np.cos(np.deg2rad(self.insertion_misalign))

        else:
            print('STEP TYPES DOES NOT MATCH')

        rot_vel = 0.0
        next_vel = [x_vel, -z_vel, rot_vel]
        J = self._build_Jacobian(th_current)
        Jinv = np.linalg.inv(J)
        j_vel_next = np.dot(Jinv, next_vel)
        return j_vel_next

    def _read_tips_pos_fk(self, th_current):
        q2 = th_current[0]
        q4 = th_current[1]
        q6 = th_current[2]

        current_tips_posx = (
            self.L34 * np.sin(-q2)
            + self.L56 * np.sin(-q2 + q4)
            + self.L78 * np.sin(-q2 + q4 - q6)
            + self.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)
        )
        current_tips_posz = (
            self.L12
            + self.L34 * np.cos(-q2)
            + self.L56 * np.cos(-q2 + q4)
            + self.L78 * np.cos(-q2 + q4 - q6)
            + self.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)
        )
        current_tips_rot = -q2 + q4 - q6 + 90.0 * (np.pi / 180.0)

        return current_tips_posx, current_tips_posz, current_tips_rot

    def _build_Jacobian(self, th_current):
        q2 = th_current[0]
        q4 = th_current[1]
        q6 = th_current[2]

        a_x = (
            -self.L34 * np.cos(-q2)
            - self.L56 * np.cos(-q2 + q4)
            - self.L78 * np.cos(-q2 + q4 - q6)
            - self.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)
        )
        b_x = (
            self.L56 * np.cos(-q2 + q4)
            + self.L78 * np.cos(-q2 + q4 - q6)
            + self.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)
        )
        c_x = -self.L78 * np.cos(-q2 + q4 - q6) - self.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)

        a_z = (
            self.L34 * np.sin(-q2)
            + self.L56 * np.sin(-q2 + q4)
            + self.L78 * np.sin(-q2 + q4 - q6)
            + self.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)
        )
        b_z = (
            -self.L56 * np.sin(-q2 + q4)
            - self.L78 * np.sin(-q2 + q4 - q6)
            - self.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)
        )
        c_z = self.L78 * np.sin(-q2 + q4 - q6) + self.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)

        J = [[a_x, b_x, c_x], [a_z, b_z, c_z], [-1.0, 1.0, -1.0]]

        return J


class Joint:
    def __init__(
        self,
        name,
        vortex_env: VortexEnv,
        position_min,
        position_max,
        vel_min,
        vel_max,
        torque_min,
        torque_max,
        vx_angle_name=None,
        vx_vel_name=None,
        vx_torque_name=None,
        vx_vel_cmd_name=None,
    ):
        self._vortex_env = vortex_env

        self.name = name
        self.position_min = position_min  # [degrees]
        self.position_max = position_max  # [degrees]
        self.vel_min = vel_min  # [degrees/s]
        self.vel_max = vel_max  # [degrees/s]
        self.torque_min = torque_min  # [N*m]
        self.torque_max = torque_max  # [N*m]

        # States
        self.angle = 0.0  # [degrees]
        self.vel = 0.0  # [degrees/s]
        self.torque = 0.0  # [N*m]
        self._vel_cmd = 0.0  # [degrees/s]

        # Vortex names
        self.vx_angle_name = vx_angle_name
        self.vx_vel_name = vx_vel_name
        self.vx_torque_name = vx_torque_name
        self.vx_vel_cmd_name = vx_vel_cmd_name

    def update_state(self):
        """Update joint state from Vortex

        Args:
            vortex_env (VortexEnv): Vortex environment
        """
        if self.vx_angle_name is not None:
            self.angle = np.rad2deg(self._vortex_env.get_output(self.vx_angle_name))

        if self.vx_vel_name is not None:
            self.vel = np.rad2deg(self._vortex_env.get_output(self.vx_vel_name))

        if self.vx_torque_name is not None:
            self.torque = self._vortex_env.get_output(self.vx_torque_name)

        if self.vx_vel_cmd_name is not None:
            self._vel_cmd = np.rad2deg(self._vortex_env.get_input(self.vx_vel_cmd_name))

    @property
    def vel_cmd(self):
        return self._vel_cmd

    @vel_cmd.setter
    def vel_cmd(self, value):
        self._vel_cmd = max(self.vel_min, min(value, self.vel_max))

        self._vortex_env.set_input(self.vx_vel_cmd_name, np.deg2rad(self._vel_cmd))

    def __repr__(self):
        return f'{self.name}: angle={self.angle:<10.4f}, vel={self.vel:<10.4f}, torque={self.torque:<10.4f}, vel_cmd={self.vel_cmd:<10.4f}'


class KinovaGen2Joints:
    def __init__(self, vortex_env: VortexEnv) -> None:
        self.j1 = Joint('j1', vortex_env, -180.0, 180.0, -180.0, 180.0, -10.0, 10.0)
        self.j2 = Joint(
            'j2',
            vortex_env,
            -180.0,
            180.0,
            -180.0,
            180.0,
            -10.0,
            10.0,
            vx_angle_name=VX_OUT.j2_pos_real,
            vx_vel_name=VX_OUT.j2_vel_real,
            vx_torque_name=VX_OUT.j2_torque,
            vx_vel_cmd_name=VX_IN.j2_vel_id,
        )
        self.j3 = Joint('j3', vortex_env, -180.0, 180.0, -180.0, 180.0, -10.0, 10.0)
        self.j4 = Joint(
            'j4',
            vortex_env,
            -180.0,
            180.0,
            -180.0,
            180.0,
            -10.0,
            10.0,
            vx_angle_name=VX_OUT.j4_pos_real,
            vx_vel_name=VX_OUT.j4_vel_real,
            vx_torque_name=VX_OUT.j4_torque,
            vx_vel_cmd_name=VX_IN.j4_vel_id,
        )
        self.j5 = Joint('j5', vortex_env, -180.0, 180.0, -180.0, 180.0, -10.0, 10.0)
        self.j6 = Joint(
            'j6',
            vortex_env,
            -180.0,
            180.0,
            -180.0,
            180.0,
            -10.0,
            10.0,
            vx_angle_name=VX_OUT.j6_pos_real,
            vx_vel_name=VX_OUT.j6_vel_real,
            vx_torque_name=VX_OUT.j6_torque,
            vx_vel_cmd_name=VX_IN.j6_vel_id,
        )
        self.j7 = Joint('j7', vortex_env, -180.0, 180.0, -180.0, 180.0, -10.0, 10.0)

    def __repr__(self):
        str_list = []
        for joint in self.__dict__.values():
            str_list.append(str(joint))

        return '\n'.join(str_list)


class PIDController:
    def __init__(self, kp, ki, kd, dt, n_joints=7):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.n_joints = n_joints
        self.integral = np.zeros(self.n_joints)
        self.prev_error = np.zeros(self.n_joints)

    def compute_velocities(self, current_angles, desired_angles):
        # Compute the error
        error = np.array(desired_angles) - np.array(current_angles)

        # Proportional term
        p = self.kp * error

        # Integral term
        self.integral += error * self.dt
        i = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        d = self.kd * derivative

        # Update previous error
        self.prev_error = error

        # Compute PID output
        velocities = p + i + d

        return velocities.tolist()
