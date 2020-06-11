import sys
import cv2 as cv
import libry as ry
from util.setup import setup_env_subgoal_1
from util.setup import setup_camera
import util.perception as perc
import util.geom as geom
import numpy as np
import time
from util.planner import check_if_goal_constant
import util.primitive as grasp
import util.transformations as _tf
from util.planner import set_goal_ball
import util.primitive as prim
from util.behavior import GrabAndLift
from util.behavior import PickAndPlace
from transitions import Machine
from functools import partial
from util.transformations import quaternion_from_matrix

pathRepo = '/home/jason/git/robotics-course/'

"""
Short example for testing the transition library, 
and building a state machine for the different primitives
"""

def cheat_update_obj(obj):
    C.addFrame(obj)
    C.frame(obj).setPosition(S.getGroundTruthPosition(obj))
    C.frame(obj).setShape(ry.ST.ssBox, size=S.getGroundTruthSize(obj))
    C.frame(obj).setQuaternion(quaternion_from_matrix(S.getGroundTruthRotationMatrix(obj)))
    C.frame(obj).setColor([1, 0, 0])
    C.frame(obj).setContact(1)


if __name__ == "__main__":

    # setup env and get background
    R, S, C, V, back_frame = setup_env_subgoal_1(False)
    cameraFrame, fxfypxpy = setup_camera(C)    # the focal length
    tau = .01
    rate_camera = 10

    state = 0

    hasGoal = False

    panda = PickAndPlace(C, S, V, tau)

    for t in range(1000):
        time.sleep(tau)

        # frame rate of camera, do perception here
        if t > 200 and not hasGoal:
            cheat_update_obj("obj0")
            cheat_update_obj("obj1")
            panda.set_blocks(["obj0", "obj1"])
            hasGoal = True

        panda.step(t)

    time.sleep(5)
