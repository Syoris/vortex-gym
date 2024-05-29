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

...

from spatialmath import SE3
import matplotlib.pyplot as plt

T1 = SE3.Tx(1)
plt.figure()  # create a new figure
SE3().plot(frame='0', dims=[-3, 3], color='black')
T1.plot(frame='1')

...
