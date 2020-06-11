import sys
import cv2 as cv
import libry as ry
from util.setup import setup_challenge_env
from util.setup import setup_camera
from util.setup import setup_env_test_edge_grasp
from util.transformations import quaternion_from_matrix
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

def cheat_update_goal(goal):
    goal.setPosition(S.getGroundTruthPosition("obj0"))
    goal.setShape(ry.ST.ssBox, size=S.getGroundTruthSize("obj0"))
    goal.setQuaternion(quaternion_from_matrix(S.getGroundTruthRotationMatrix("obj0")))

if __name__ == "__main__":

    # setup env and get background
    R, S, C, V, back_frame = setup_env_test_edge_grasp(show_background=False)
    cameraFrame, fxfypxpy = setup_camera(C)    # the focal length

    tau = .01
    # add goal to config
    goal = C.addFrame("goal")
    goal.setContact(1)

    # cheat and set goal in config from simulation
    cheat_update_goal(goal)
    V.setConfiguration(C)

    align_push = prim.AlignPush(C, S, V, tau, 100, gripper="R_gripper", vis=True)
    align_push.create_primitive(0)
    for t in range(200):

        align_push.step(t)

    time.sleep(10)



