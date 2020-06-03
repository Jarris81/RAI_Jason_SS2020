import random

import libry as ry
import numpy as np
import util.ball_tracking as perc
from os.path import join

pathRepo = '/home/nika/git/robotics-course'


def setup_challenge_env(add_red_ball=False, number_objects=30, show_background=False):
    # -- Add empty REAL WORLD configuration and camera
    R = ry.Config()
    R.addFile(join(pathRepo, "scenarios/pandasTable.g"))
    S = R.simulation(ry.SimulatorEngine.physx, True)
    S.addSensor("camera")

    back_frame = perc.extract_background(S, duration=2, vis=show_background)

    R = ry.Config()
    R.addFile(join(pathRepo, "scenarios/challenge.g"))

    # Change color of objects
    if add_red_ball:
        # only add 1 red ball
        number_objects = 2
        # you can also change the shape & size
        R.getFrame("obj0").setColor([1., 0, 0])
        R.getFrame("obj0").setShape(ry.ST.sphere, [.03])
        # RealWorld.getFrame("obj0").setShape(ry.ST.ssBox, [.05, .05, .2, .01])
        R.getFrame("obj0").setPosition([0.0, .05, 2.])
        R.getFrame("obj0").setContact(1)

        R.getFrame("obj1").setColor([0, 0, 1.])
        R.getFrame("obj1").setShape(ry.ST.sphere, [.03])
        # RealWorld.getFrame("obj0").setShape(ry.ST.ssBox, [.05, .05, .2, .01])
        R.getFrame("obj1").setPosition([0.0, .3, 2.])
        R.getFrame("obj1").setContact(1)

        for o in range(number_objects, 30):
            name = "obj%i" % o
            print("deleting", name)
            R.delFrame(name)

    S = R.simulation(ry.SimulatorEngine.physx, True)

    # Change color of objects
    S.addSensor("camera")

    C = ry.Config()
    C.addFile(join(pathRepo, "scenarios/pandasTable.g"))
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    C.addFrame("goal")

    return R, S, C, V, back_frame


def setup_color_challenge_env():
    random.seed(9)

    R = ry.Config()

    R.addFile(join(pathRepo, "scenarios/challenge.g"))
    # Change color of objects
    obj_count = 0
    for n in R.getFrameNames():
        if n.startswith("obj"):
            obj_count += 1

    for o in range(0, obj_count):
        # color = list(np.random.choice(np.arange(0, 1, 0.05), size=2)) + [1]
        # color = list(np.random.choice(np.arange(0, 1, 0.01), size=3))
        color = random.choice([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                               [1, 0.5, 0], [0.5, 0, 1], [0, 1, 0.5], [0, 0.5, 1], [0.5, 1, 0]])
        name = "obj%i" % o
        R.frame(name).setColor(color)
        # info = R.frame(name).info()

    S = R.simulation(ry.SimulatorEngine.physx, True)
    S.addSensor("camera")

    C = ry.Config()
    C.addFile(join(pathRepo, 'scenarios/pandasTable.g'))
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    cameraFrame = C.frame("camera")

    q0 = C.getJointState()
    R_gripper = C.frame("R_gripper")
    R_gripper.setContact(1)
    L_gripper = C.frame("L_gripper")
    L_gripper.setContact(1)

    return R, S, C, V


def setup_goal1():
    R = ry.Config()
    R.addFile(join(pathRepo, 'scenarios/pandasTable.g'))

    S = R.simulation(ry.SimulatorEngine.physx, True)
    S.addSensor("camera")

    C = ry.Config()
    C.addFile(join(pathRepo, 'scenarios/pandasTable.g'))
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    cameraFrame = C.frame("camera")

    q0 = C.getJointState()
    R_gripper = C.frame("R_gripper")
    R_gripper.setContact(1)
    L_gripper = C.frame("L_gripper")
    L_gripper.setContact(1)

    obj1 = R.addFrame("obj1")
    obj1.setColor([0, 0, 1])
    obj1.setShape(ry.ST.ssBox, [0.2, 0.15, 0.1, 0])
    obj1.setPosition([-.6, 0, .7])

    return R, S, C, V


def setup_camera(C):
    # setup camera
    cameraFrame = C.frame("camera")
    # the focal length
    f = 0.895
    f = f * 360.
    fxfypxpy = [f, f, 320., 180.]

    return cameraFrame, fxfypxpy
