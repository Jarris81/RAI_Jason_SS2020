import sys
pathRepo = '../../git/robotics-course/'
from os.path import join
#sys.path.append(join(pathRepo,' build'))
import numpy as np
import libry as ry
import time


def basics():
    # Real world configs
    # -- REAL WORLD configuration, which is attached to the physics engine
    # accessing this directly would be cheating!
    RealWorld = ry.Config()
    RealWorld.addFile(join(pathRepo, "scenarios/challenge.g"))

    S = RealWorld.simulation(ry.SimulatorEngine.physx, True)
    S.addSensor("camera")

    # -- MODEL WORLD configuration, this is the data structure on which you represent
    # what you know about the world and compute things (controls, contacts, etc)
    C = ry.Config()
    # D = C.view() #rather use the ConfiguratioViewer below
    C.addFile(join(pathRepo, "scenarios/pandasTable.g"))

    # -- using the viewer, you can view configurations or paths
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)

    # -- the following is the simulation loop
    tau = .01

    # for t in range(300):
    #     time.sleep(0.01)
    #
    #     # grab sensor readings from the simulation
    #     q = S.get_q()
    #     if t % 10 == 0:
    #         [rgb, depth] = S.getImageAndDepth()  # we don't need images with 100Hz, rendering is slow
    #
    #     # some good old fashioned IK
    #     C.setJointState(q)  # set your robot model to match the real q
    #     V.setConfiguration(C)  # to update your model display
    #     [y, J] = C.evalFeature(ry.FS.position, ["R_gripper"])
    #     vel = J.T @ np.linalg.inv(J @ J.T + 1e-2 * np.eye(y.shape[0])) @ [0., -0.1, -1e-1];
    #
    #     # send velocity controls to the simulation
    #     S.step(vel, tau, ry.ControlMode.velocity)

    # add a new frame to the MODEL configuration
    # (Perception will later have to do exactly this: add perceived objects to the model)
    obj = C.addFrame("object")

    # set frame parameters, associate a shape to the frame,
    obj.setPosition([.8, 0, 1.5])
    obj.setQuaternion([0, 0, .5, 0])
    obj.setShape(ry.ST.capsule, [.2, .02])
    obj.setColor([1, 0, 1])
    V.setConfiguration(C)
    V.recopyMeshes(C)  # this is rarely necessary, on

    radius = 0.4

    x_origin = 0.4
    y_origin = 0.0
    #C.selectJointsByTag(["armL"])
    print('joint names: ', C.getJointNames())

    # -- the following is the simulation loop
    tau = .01

    for t in range(1000):
        time.sleep(0.01)

        x = x_origin + np.cos(t/200) * radius
        y = y_origin + np.sin(t/200) * radius
        obj.setQuaternion([1, 0, .5, 0])

        obj.setPosition([x, y, 1.5])

        # grab sensor readings from the simulation
        q = S.get_q()
        if t % 10 == 0:
            [rgb, depth] = S.getImageAndDepth()  # we don't need images with 100Hz, rendering is slow

        # some good old fashioned IK
        C.setJointState(q)  # set your robot model to match the real q
        V.setConfiguration(C)  # to update your model display
        [y, J] = C.evalFeature(ry.FS.scalarProductXX, ["R_gripperCenter", "object"])
        vel = J.T @ np.linalg.inv(J @ J.T + 1e-2 * np.eye(y.shape[0])) @ (-y);

        #print(J.shape)
        #print(vel.shape)

        # send velocity controls to the simulation
        S.step(vel, tau, ry.ControlMode.velocity)


if __name__ == "__main__":
    basics()
