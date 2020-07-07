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
from util.behavior import EdgeGrasper
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
    rate_camera = 10
    tau = .01
    # add goal to config
    goal = C.addFrame("goal")
    goal.setContact(1)

    # cheat and set goal in config from simulation
    cheat_update_goal(goal)
    V.setConfiguration(C)

    # pull_in = prim.PullIn(C, S, V, tau, 200, interpolation=True, vis=False)
    # pull_in.create_primitive(gripper="R_gripper", goal="goal", t_start=0)
    # push_to_edge = prim.PushToEdge(C, S, V, tau, 600, interpolation=True, vis=False)
    # #push_to_edge.create_primitive(gripper="R_gripper", goal="goal", t_start=0)
    # edge_grasp = prim.EdgeGrasp(C, S, V, tau, 200, interpolation=True, vis=True)
    # #edge_grasp.create_primitive(gripper="R_gripper", goal="goal", t_start=0)
    # grav_comp = prim.GravComp(C, S, V, tau, 1000)
    # state = edge_grasp

    panda = EdgeGrasper(C, S, V, tau)

    for t in range(10000):
        time.sleep(tau)
        if t > 100 and t % rate_camera:
            cheat_update_goal(goal)
            panda.set_blocks(["goal"])

        panda.step(t)

    time.sleep(10)



