import roboticstoolbox as rtb
from roboticstoolbox.robot.DHLink import RevoluteDH

from vortex_gym import ASSETS_DIR, ROBOT_CFG_DIR
import numpy as np

# # robot = rtb.models.DH.Jaco()
# robot = rtb.Robot.URDF(ASSETS_DIR / 'Kinova Gen2 Unjamming' / 'j2s7s300_ee.urdf')

# print(robot)

# Te = robot.fkine(robot.q)  # forward kinematics
# print(Te)

# from spatialmath import SE3

# Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
# sol = robot.ik_LM(Tep)  # solve IK
# print(sol)

# q_pickup = sol[0]
# print(robot.fkine(q_pickup))  # FK shows that desired end-effector pose was achie

# qt = rtb.jtraj(robot.qr, q_pickup, 50)
# robot.plot(qt.q, backend='pyplot', movie='panda1.gif')

# ...
# robot.plot(qt.q)

# ...

# import swift
# import roboticstoolbox as rtb
# import spatialmath as sm
# import numpy as np

# env = swift.Swift()
# env.launch(realtime=True)

# panda = rtb.models.Panda()
# panda.q = panda.qr

# Tep = panda.fkine(panda.q) * sm.SE3.Trans(0.2, 0.2, 0.45)

# arrived = False
# env.add(panda)

# dt = 0.05

# while not arrived:
#     v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
#     panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
#     env.step(dt)

# # Uncomment to stop the browser tab from closing
# env.hold()

# robot_model = rtb.Robot.URDF(ASSETS_DIR / 'Kinova Gen2 Unjamming' / 'j2s7s300_ee.urdf')

# d1 = 0.2755
# d2 = 0.2050
# d3 = 0.2050
# d4 = 0.2073
# d5 = 0.1038
# d6 = 0.1038
# d7 = 0.1150  # 0.1600
# e2 = 0.0098

# robot_model2 = rtb.DHRobot(
#     [
#         RevoluteDH(alpha=np.pi / 2, d=d1),  # 1
#         RevoluteDH(alpha=np.pi / 2),  # 2
#         RevoluteDH(alpha=np.pi / 2, d=-(d2 + d3)),  # 3
#         RevoluteDH(alpha=np.pi / 2, d=-e2, offset=np.pi),  # 4
#         RevoluteDH(alpha=np.pi / 2, d=-(d4 + d5)),  # 5
#         RevoluteDH(alpha=np.pi / 2, offset=np.pi),  # 6
#         RevoluteDH(alpha=np.pi, d=-(d6 + d7)),  # 7
#     ],
#     name='KinovaGen2',
# )

# print('URDF Model:')
# print(robot_model)
# print('\nDH Model:')
# print(robot_model2)

# robot_model.plot([0, np.deg2rad(45), 0, 0, 0, 0, 0], backend='pyplot')

# q = np.deg2rad([0, 45, 0, 45, 0, 0, 0])
# robot_model2.plot(q, backend='pyplot')

"""
Spatial Math
"""
# from spatialmath import SE3
# import matplotlib.pyplot as plt

# T1 = SE3.Tx(1)
# plt.figure()  # create a new figure
# SE3().plot(frame='0', dims=[-3, 3], color='black')
# T1.plot(frame='1')


"""
3 DoF Kinova Robot
"""
d1 = 0.2755
d2 = 0.2050
d3 = 0.2050
d4 = 0.2073
d5 = 0.1038
d6 = 0.1038
d7 = 0.1150  # 0.1600
e2 = 0.0098

print('--- 7 DoF ---')
kin_7 = rtb.DHRobot(
    [
        RevoluteDH(alpha=np.pi / 2, d=d1, qlim=[0, 0]),  # 1
        RevoluteDH(alpha=np.pi / 2),  # 2
        RevoluteDH(alpha=np.pi / 2, d=-(d2 + d3)),  # 3
        RevoluteDH(alpha=np.pi / 2, d=-e2, offset=np.pi),  # 4
        RevoluteDH(alpha=np.pi / 2, d=-(d4 + d5)),  # 5
        RevoluteDH(alpha=np.pi / 2, offset=np.pi),  # 6
        RevoluteDH(alpha=0, d=-(d6 + d7), offset=np.pi),  # 7
    ],
    name='KinovaGen2_7DoF',
)
print(kin_7)
q_1 = np.deg2rad([0, 45, 0, 45, 0, 90, 0])
kin_7.plot(q_1)

print('--- 3 DoF ---')
kin_3 = rtb.DHRobot(
    [
        RevoluteDH(a=(d2 + d3), alpha=0, d=0),  # 1
        RevoluteDH(a=(d4 + d5), d=e2, alpha=0),  # 1
        RevoluteDH(a=(d6 + d7), alpha=-np.pi / 2, d=0, offset=np.pi / 2),  # 1
        # RevoluteDH(alpha=np.pi / 2),  # 2
        # RevoluteDH(alpha=np.pi / 2, d=-(d2 + d3)),  # 3
        # RevoluteDH(alpha=np.pi / 2, d=-e2, offset=np.pi),  # 4
        # RevoluteDH(alpha=np.pi / 2, d=-(d4 + d5)),  # 5
        # RevoluteDH(alpha=np.pi / 2, offset=np.pi),  # 6
        # RevoluteDH(alpha=0, d=-(d6 + d7), offset=np.pi),  # 7
    ],
    name='KinovaGen2_3DoF',
)
print(kin_3)
q_2 = np.deg2rad([0, 0, 0])
# kin_3.plot(q_2)

print('--- URDF ---')
kin_urdf = rtb.Robot.URDF(ASSETS_DIR / 'Kinova Gen2 Unjamming' / 'j2s7s300_ee.urdf')
print(kin_urdf)
# kin_urdf.plot(q_1, backend='pyplot')


rtb.models.ETS.GenericSeven()
print('Done')


import numpy as np
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.robot.Link import Link


class Kinova7DoF(Robot):
    """
    Create model of a generic seven degree-of-freedom robot

    robot = GenericSeven() creates a robot object. This robot is represented
    using the elementary transform sequence (ETS).
    """

    def __init__(self):
        d1 = 0.2755
        d2 = 0.2050
        d3 = 0.2050
        d4 = 0.2073
        d5 = 0.1038
        d6 = 0.1038
        d7 = 0.1150  # 0.1600
        e2 = 0.0098

        l0 = Link(ETS(ET.Rx(np.pi)) * ET.Rz(np.pi) * ET.Rz(), name='link0', parent=None)  # J1

        l1 = Link(ET.tz(-d1) * ET.Rx(np.pi / 2) * ET.Rz(np.pi) * ET.Rz(), name='link1', parent=l0)  # J2

        l2 = Link(ET.ty(-d2) * ET.Rx(np.pi / 2) * ET.Rz(), name='link2', parent=l1)  # J3

        l3 = Link(ET.tz(d3) * ET.Rx(np.pi / 2) * ET.Rz(np.pi) * ET.Rz(), name='link3', parent=l2)  # J4

        l4 = Link(ET.ty(d4) * ET.tx(e2) * ET.Rx(np.pi / 2) * ET.Rz(np.pi) * ET.Rz(), name='link4', parent=l3)  # J5

        l5 = Link(ET.tz(-d5) * ET.Rx(np.pi / 2) * ET.Rz(np.pi) * ET.Rz(), name='link5', parent=l4)  # J6

        l6 = Link(ET.ty(d6) * ET.Rx(np.pi / 2) * ET.Rz(), name='link6', parent=l5)  # J7

        ee = Link(ET.tz(-d7) * ET.Rx(np.pi) * ET.Rz(-np.pi / 2), name='ee', parent=l6)  # EE

        elinks = [l0, l1, l2, l3, l4, l5, l6, ee]

        super(Kinova7DoF, self).__init__(elinks, name='Kinova7DoF')

        self.qr = np.deg2rad(np.array([180, 240, 0, 60, 90, 90, 180]))  # DH Example position
        self.qc = np.deg2rad(np.array([180, 180, 180, 180, 180, 180, 180]))  # Candle like position
        self.qz = np.zeros(7)

        self.addconfiguration('qr', self.qr)
        self.addconfiguration('qz', self.qz)
        self.addconfiguration('qc', self.qc)


class Kinova3DoF(Robot):
    """
    Create model of a generic seven degree-of-freedom robot

    robot = GenericSeven() creates a robot object. This robot is represented
    using the elementary transform sequence (ETS).
    """

    def __init__(self):
        d1 = 0.2755
        d2 = 0.2050
        d3 = 0.2050
        d4 = 0.2073
        d5 = 0.1038
        d6 = 0.1038
        d7 = 0.1150  # 0.1600
        e2 = 0.0098

        l0 = Link(ETS(ET.Rx(np.pi)) * ET.Rz(np.pi) * ET.Rz(0), name='link0', parent=None)  # J1

        l1 = Link(ET.tz(-d1) * ET.Rx(np.pi / 2) * ET.Rz(np.pi) * ET.Rz(), name='link1', parent=l0)  # J2

        l2 = Link(ET.ty(-d2) * ET.Rx(np.pi / 2) * ET.Rz(0), name='link2', parent=l1)  # J3

        l3 = Link(ET.tz(d3) * ET.Rx(np.pi / 2) * ET.Rz(np.pi) * ET.Rz(), name='link3', parent=l2)  # J4

        l4 = Link(ET.ty(d4) * ET.tx(e2) * ET.Rx(np.pi / 2) * ET.Rz(np.pi) * ET.Rz(0), name='link4', parent=l3)  # J5

        l5 = Link(ET.tz(-d5) * ET.Rx(np.pi / 2) * ET.Rz(np.pi) * ET.Rz(), name='link5', parent=l4)  # J6

        l6 = Link(ET.ty(d6) * ET.Rx(np.pi / 2) * ET.Rz(0), name='link6', parent=l5)  # J7

        ee = Link(ET.tz(-d7) * ET.Rx(np.pi) * ET.Rz(-np.pi / 2), name='ee', parent=l6)  # EE

        elinks = [l0, l1, l2, l3, l4, l5, l6, ee]

        super(Kinova3DoF, self).__init__(elinks, name='Kinova3DoF')

        # self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        # self.qz = np.zeros(3)
        self.qc = np.deg2rad(np.array([180, 180, 180]))

        # self.addconfiguration('qr', self.qr)
        # self.addconfiguration('qz', self.qz)
        self.addconfiguration('qc', self.qc)


q = np.deg2rad(np.array([0, 240, 0, 60, 0, 90, 0]))
# q_ex = np.deg2rad(np.array([180, 240, 0, 60, 90, 90, 180]))  # DH Example position

robot_7dof = Kinova7DoF()
print(robot_7dof)


robot_3dof = Kinova3DoF()
print(robot_3dof)

robot_7dof.plot(robot_7dof.qc, backend='pyplot')
robot_3dof.plot(q[[1, 3, 5]], backend='pyplot')
...
