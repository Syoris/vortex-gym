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
# d1 = 0.2755
# d2 = 0.2050
# d3 = 0.2050
# d4 = 0.2073
# d5 = 0.1038
# d6 = 0.1038
# d7 = 0.1150  # 0.1600
# e2 = 0.0098

# print('--- 7 DoF ---')
# kin_7 = rtb.DHRobot(
#     [
#         RevoluteDH(alpha=np.pi / 2, d=d1, qlim=[0, 0]),  # 1
#         RevoluteDH(alpha=np.pi / 2),  # 2
#         RevoluteDH(alpha=np.pi / 2, d=-(d2 + d3)),  # 3
#         RevoluteDH(alpha=np.pi / 2, d=-e2, offset=np.pi),  # 4
#         RevoluteDH(alpha=np.pi / 2, d=-(d4 + d5)),  # 5
#         RevoluteDH(alpha=np.pi / 2, offset=np.pi),  # 6
#         RevoluteDH(alpha=0, d=-(d6 + d7), offset=np.pi),  # 7
#     ],
#     name='KinovaGen2_7DoF',
# )
# print(kin_7)
# q_1 = np.deg2rad([0, 45, 0, 45, 0, 90, 0])
# kin_7.plot(q_1)

# print('--- 3 DoF ---')
# kin_3 = rtb.DHRobot(
#     [
#         RevoluteDH(a=(d2 + d3), alpha=0, d=0),  # 1
#         RevoluteDH(a=(d4 + d5), d=e2, alpha=0),  # 1
#         RevoluteDH(a=(d6 + d7), alpha=-np.pi / 2, d=0, offset=np.pi / 2),  # 1
#         # RevoluteDH(alpha=np.pi / 2),  # 2
#         # RevoluteDH(alpha=np.pi / 2, d=-(d2 + d3)),  # 3
#         # RevoluteDH(alpha=np.pi / 2, d=-e2, offset=np.pi),  # 4
#         # RevoluteDH(alpha=np.pi / 2, d=-(d4 + d5)),  # 5
#         # RevoluteDH(alpha=np.pi / 2, offset=np.pi),  # 6
#         # RevoluteDH(alpha=0, d=-(d6 + d7), offset=np.pi),  # 7
#     ],
#     name='KinovaGen2_3DoF',
# )
# print(kin_3)
# q_2 = np.deg2rad([0, 0, 0])
# # kin_3.plot(q_2)

# print('--- URDF ---')
# kin_urdf = rtb.Robot.URDF(ASSETS_DIR / 'Kinova Gen2 Unjamming' / 'j2s7s300_ee.urdf')
# print(kin_urdf)
# # kin_urdf.plot(q_1, backend='pyplot')


# rtb.models.ETS.GenericSeven()
# print('Done')


import numpy as np
from vortex_gym.robot.kinova_gen_2 import Kinova7DoFModel, Kinova3DoFModel
from spatialmath import SE3

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

"""
FKIN
"""
# # q = np.deg2rad(np.array([0, 240, 0, 60, 0, 90, 0]))
# q = np.deg2rad(np.array([180, 240, 0, 60, 90, 90, 180]))  # DH Example position

# robot_7dof = Kinova7DoFModel()
# print(robot_7dof)


# robot_3dof = Kinova3DoFModel()
# print(robot_3dof)

# robot_7dof.plot(q, backend='pyplot')
# robot_3dof.plot(q[[1, 3, 5]], backend='pyplot')

"""
IKINE
"""
robot_3dof = Kinova3DoFModel()

goal = [0.55, -0.0098, 0.25]
orientation = [0, 90, 90]

T_goal = SE3.Trans(goal) * SE3.RPY(orientation, order='xyz', unit='deg')

# sol = robot_3dof.ikine_LM(T_goal, q0=[0, 0, 0])
# robot_3dof.plot(sol.q, backend='pyplot')
# plt.show()

# q = np.deg2rad([40.70877862, 43.24971183, 87.45907129])
q_deg = [180 - 40.70877862, 360 - 43.24971183, -87.45907129]
q = np.deg2rad(q_deg)
sol2 = robot_3dof.ikine_LM(T_goal, q0=[q[0], 0, 0])
robot_3dof.plot(sol2.q, backend='pyplot')
print(f'My sol: {q_deg}')
print(f'IKINE: {np.rad2deg(sol2.q)}')

print('IKINE Done')
