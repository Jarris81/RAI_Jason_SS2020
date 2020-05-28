import sys
import cv2 as cv
import libry as ry
from util.setup import setup_challenge_env
import util.perception as perc
import util.geom as geom
import numpy as np
import time
from util.planner import check_if_goal_constant
import util.primitive as grasp
import util.transformations as _tf
from util.planner import set_goal_ball
import util.primitive as prim

pathRepo = '/home/jason/git/robotics-course/'


if __name__ == "__main__":

    # setup env
    R, S, C, V = setup_challenge_env(True)

    # setup camera
    cameraFrame = C.frame("camera")
    # the focal length
    f = 0.895
    f = f * 360.
    fxfypxpy = [f, f, 320., 180.]

    # get background
    back_frame = perc.extract_background(S, duration=2)

    points = []
    tau = .01
    rate_camera = 10

    goals = []
    goals_stored = 3

    state = 0

    hasGoal = False
    komo = 0

    retry= 0


    # start simulation loop
    for t in range(1000):
        time.sleep(tau)

        q = S.get_q()

        # frame rate of camera, do perception here
        if t % rate_camera == 0:
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
                hasGoal = True

        # do nothing state, wait for something to happen
        if state == 0:
            if hasGoal:
                state = 1

        if state == 1:
            # generate top grasp
            top_grasp = prim.TopGrasp(C, S, tau, t, 100,"R_gripper", V=V)
            state = 2

        if state == 2:
            if top_grasp.goal_changed_cond():
                if not top_grasp.is_done_cond(t):
                    top_grasp.step(t)
                    V.setConfiguration(C)
                else:
                    state = 3
            else:
                # get the closest ball and set as goal
                state = 1

        if state == 3:
            lift_up = prim.LiftUp(C, S, tau, t, 50, "R_gripper", V=V)
            state = 4
        if state == 4:
            if not lift_up.is_done_cond(t):
                lift_up.step(t)
                V.setConfiguration(C)
            else:
                state = 5
                print("is done lifting")


        else:
            S.step([], tau, ry.ControlMode.none)

    time.sleep(10)
