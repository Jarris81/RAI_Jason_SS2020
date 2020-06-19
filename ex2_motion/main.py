from os.path import join
import libry as ry
import numpy as np
import time

from util.bezier import create_bezier

pathRepo = '../../git/robotics-course/'


def setup():
    # -- MODEL WORLD configuration, this is the data structure on which you represent
    # what you know about the world and compute things (controls, contacts, etc)
    C = ry.Config()
    C.addFile(join(pathRepo, "scenarios/pandasTable.g"))

    # add box
    box = C.addFrame("goal")
    side = 0.08
    box.setShape(ry.ST.ssBox, size=[side, side, side, .02])
    box.setColor([.5, 1, 1])
    box.setPosition([0, .05, 0.68])

    box.setShape(ry.ST.sphere, [.03])
    #box.setPosition([0.0, .05, 0.68])

    box.setContact(1)

    # set contact flag for grippers
    C.frame("R_gripper").setContact(1)
    C.frame("L_gripper").setContact(1)
    C.frame("goal").setContact(1)
    return C

def vis(C, duration):
    # -- using the viewer, you can view configurations or paths
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    time.sleep(duration)

def ex1a(C):

    # Create KOMO
    iK = C.komo_IK(False)

    # add objectives for iK
    # restrict joints as
    iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e-2]*16)
    # objective between grippers
    # define gripper relation
    # same same position
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=['R_gripper', 'L_gripper'], target=[0.0, 0.0, 0.0])
    # r gripper z axis on world frame x axis, and grippers have distance
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=['R_gripper'], target=[1, 0, 0])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.distance, frames=['R_gripper', "L_gripper"], target=[-0.1])
    # relative distance from each other
    # iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=['R_gripper', 'L_gripper'], target=[0.0, 0.0, -0.2])

    # Z-axis should be in opposite direction (face each other)
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=['R_gripper', 'L_gripper'], target=[-1])
    # grippers hands should be orthogonal, so no collision
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=['R_gripper', 'L_gripper'], target=[0])
    # no contact
    iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions)
    iK.addObjective()
    # calculate and show report
    iK.optimize(True)

    # set config
    C.setFrameState(iK.getConfiguration(0))
    C.computeCollisions()
    print(C.getCollisions())

    return C


def ex1b(C):
    # Create KOMO
    iK = C.komo_IK(False)
    # add objectives for iK
    # restrict joints
    iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e-2] * 16)

    # define gripper relation
    # same position
    # iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=['R_gripper', 'L_gripper'], target=[0.0, 0.0, 0.0])
    # r gripper z axis on world frame x axis, and grippers have distance
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=['R_gripper'], target=[1, 0, 0])
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.distance, frames=['R_gripper', "L_gripper"], target=[-0.1])
    # relative distance from each other
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=['R_gripper', 'L_gripper'], target=[0.0, 0.0, -0.2])

    # Z-axis should be in opposite direction (face each other)
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=['R_gripper', 'L_gripper'], target=[-1])
    # grippers hands should be orthogonal, so no collision
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=['R_gripper', 'L_gripper'], target=[0])

    # define relation between box and gripper
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=["goal", "R_gripper"], target=[0.0, 0.0, -0.1])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=['goal', 'R_gripper'], target=[1])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYY, frames=['goal', 'R_gripper'], target=[0])

    # no contact
    iK.addObjective(feature=ry.FS.accumulatedCollisions, type=ry.OT.ineq)
    iK.optimize(True)

    C.setFrameState(iK.getConfiguration(0))
    C.computeCollisions()

    return C

def ex1c(C):

    # Create KOMO
    iK = C.komo_IK(False)
    C.frame("R_gripper").setContact(1)
    C.frame("L_gripper").setContact(1)
    C.frame("L_frame").setContact(1)
    C.frame("R_frame").setContact(1)
    C.frame("goal").setContact(1)
    C.frame("R_finger1").setContact(1)
    C.frame("R_finger2").setContact(1)

    # add objectives for iK
    # restrict joints
    iK.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, scale=[1e-2] * 16)

    # define gripper relation
    # same position
    # iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=['R_gripper', 'L_gripper'], target=[0.0, 0.0, 0.0])
    # r gripper z axis on world frame x axis, and grippers have distance
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.vectorZ, frames=['R_gripper'], target=[1, 0, 0])
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.distance, frames=['R_gripper', "L_gripper"], target=[-0.1])
    # relative distance from each other
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=['R_gripper', 'L_gripper'], target=[0.0, 0.0, -0.2])

    # Z-axis should be in opposite direction (face each other)
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=['R_gripper', 'L_gripper'], target=[-1])
    # grippers hands should be orthogonal, so no collision
    #iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=['R_gripper', 'L_gripper'], target=[0])

    # define relation between box and gripper
    w = 2
    iK.addObjective(type=ry.OT.sos, feature=ry.FS.distance, frames=["goal", "R_gripper"], scale=[-1])
    #iK.addObjective(type=ry.OT.sos, feature=ry.FS.positionDiff, frames=["goal", "R_gripper"], scale=[w, w, w])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=['R_gripper', 'goal'], target=[1])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=['goal', 'R_gripper'], target=[1])
    iK.addObjective(type=ry.OT.eq, feature=ry.FS.positionRel, frames=['goal', 'R_gripper'],
                    target=[0.0, 0.0, 0.0], scale=[1, 1, 0])
    iK.addObjective(type=ry.OT.sos, feature=ry.FS.positionRel, frames=['goal', 'R_gripper'],
                    target=[0.0, 0.0, 1.0], scale=[0, 0, 1])
    # no contact
    iK.addObjective(type=ry.OT.ineq, feature=ry.FS.accumulatedCollisions, scale=[100])
    iK.optimize(True)

    C.setFrameState(iK.getConfiguration(0))
    C.computeCollisions()
    print(C.getCollisions())

    return C

def ex2a(C):

    # get q-config from ex1b
    q_start = C.getJointState()
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    q_final = ex1b(C).getJointState()

    q_start = np.asarray(q_start)
    q_final = np.asarray(q_final)
    delta = q_final-q_start

    smoothBezier = create_bezier("EaseInOutSine")

    time.sleep(2)

    steps = np.linspace(0,1,100)
    for t in steps:
        factor = smoothBezier.solve(t)
        q = q_start + delta*factor

        C.setJointState(q)
        V.setConfiguration(C)
        time.sleep(0.01)
    time.sleep(10)


def ex2c(C, show=False):

    # add obstacle
    obs = C.addFrame("obstacle_1")
    obs.setShape(ry.ST.ssBox, size=[0.1, 0.1, 1, .02])
    obs.setColor([1, 0, 0])
    obs.setPosition([0.5, 0.25, 0.8])
    obs.setContact(1)

    vis(C, 5)

    # Create KOMO
    komo = C.komo_path(1, 20, 5., True)

    # relative distance from each other
    komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1] * 16, order=2)
    komo.addObjective(time=[1.],type=ry.OT.sos, feature=ry.FS.positionRel, frames=['R_gripper', 'L_gripper'],
                    target=[0.0, 0.0, -0.2], scale=[1e2]*3)

    # Z-axis should be in opposite direction (face each other)
    komo.addObjective(time=[0.9,1.], type=ry.OT.sos, feature=ry.FS.scalarProductZZ, frames=['R_gripper', 'L_gripper'], target=[-1])
    # grippers hands should be orthogonal, so no collision
    komo.addObjective(time=[0.9,1.], type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=['R_gripper', 'L_gripper'], target=[0])

    # define relation between box and gripper
    komo.addObjective(time=[1.], type=ry.OT.sos, feature=ry.FS.positionRel, frames=["goal", "R_gripper"], target=[0.0, 0.0, -0.1], scale=[1e2]*3)
    komo.addObjective(time=[0.9, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=['R_gripper', 'goal'], target=[1], scale=[1e2])
    komo.addObjective(time=[0.9, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=['goal', 'R_gripper'], target=[1], scale=[1e2])

    # no contact
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e2])
    komo.optimize(True)

    # visualize
    if show:
        V = komo.view()
        time.sleep(5)
        V.playVideo()
        time.sleep(20)


def ex3a(C):

    #how far we want to look ahead
    def get_komo(C, step):

        komo = C.komo_path(1, 50, step, True)

        # relative distance from each other
        komo.addObjective(time=[], type=ry.OT.sos, feature=ry.FS.qItself, scale=[1] * 16, order=2)
        komo.addObjective(time=[1.], type=ry.OT.sos, feature=ry.FS.positionRel, frames=['R_gripper', 'L_gripper'],
                          target=[0.0, 0.0, -0.2], scale=[1e2] * 3)

        # Z-axis should be in opposite direction (face each other)
        komo.addObjective(time=[0.9, 1.], type=ry.OT.sos, feature=ry.FS.scalarProductZZ, frames=['R_gripper', 'L_gripper'],
                          target=[-1])
        # grippers hands should be orthogonal, so no collision
        komo.addObjective(time=[0.9, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductXX, frames=['R_gripper', 'L_gripper'],
                          target=[0])

        # define relation between box and gripper
        komo.addObjective(time=[1.], type=ry.OT.sos, feature=ry.FS.positionRel, frames=["goal", "R_gripper"],
                          target=[0.0, 0.0, -0.1], scale=[1e2] * 3)
        komo.addObjective(time=[0.9, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=['R_gripper', 'goal'],
                          target=[1], scale=[1e2])
        komo.addObjective(time=[0.9, 1.], type=ry.OT.eq, feature=ry.FS.scalarProductXZ, frames=['goal', 'R_gripper'],
                          target=[1], scale=[1e2])

        # no contact
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e2])
        komo.optimize(True)

        return komo

    # visualize
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    step_size = 5
    duration = 30
    for t in range(duration//step_size):

        step_future = duration - step_size * t

        komo = get_komo(C, step_future)
        for i in range(step_size):
            C.setFrameState(komo.getConfiguration(i))
            V.setConfiguration(C)
            time.sleep(0.1)

    time.sleep(10)

if __name__ == "__main__":

    C = setup()
    C = ex1b(C)
    vis(C, 10)
