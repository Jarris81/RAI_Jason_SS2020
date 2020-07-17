import sys
import cv2 as cv
import libry as ry
from util.setup import setup_challenge_env
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
from util.behavior import TowerBuilder
from transitions import Machine
from functools import partial

pathRepo = '/home/jason/git/robotics-course/'

"""
Short example for testing the transition library, 
and building a state machine for the different primitives
"""

if __name__ == "__main__":

    # setup env and get background
    R, S, C, V, back_frame = setup_challenge_env(True, 0, show_background=False)
    cameraFrame, fxfypxpy = setup_camera(C)    # the focal length
    C.addFrame("goal")
    tau = .01
    rate_camera = 10

    goals = []
    goals_stored = 3

    state = 0

    hasGoal = False

    panda = TowerBuilder(C, S, V, tau)

    for t in range(1000):
        time.sleep(tau)

        # frame rate of camera, do perception here
        if t % rate_camera == 0 and not hasGoal:
            frame = S.getImageAndDepth()  # we don't need images with 100Hz, rendering is slow
            goal = perc.get_red_ball_contours(frame, back_frame, cameraFrame, fxfypxpy, vis=True)
            if len(goal):
                goals.append(goal)
            if len(goals) > goals_stored:
                goals.pop(0)
            if check_if_goal_constant(goals, tolerance=0.5):
                # get the closest ball and set as goal
                goal_arg = geom.closest_point(C.frame("R_gripper").getPosition(), goals[-1])
                goal_pos = goals[-1][goal_arg]
                set_goal_ball(C, V, goal_pos, 0.03)
                panda.set_blocks(["goal"])
                # only need one goal
                hasGoal = True

        panda.step(t)


    time.sleep(5)