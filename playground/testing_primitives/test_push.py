import sys
import cv2 as cv
import libry as ry
from util.setup import setup_challenge_env
from util.setup import setup_camera
from util.setup import setup_env_test_edge_grasp
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

if __name__ == "__main__":

    # setup env and get background
    R, S, C, V, back_frame = setup_env_test_edge_grasp(show_background=False)
    cameraFrame, fxfypxpy = setup_camera(C)    # the focal length

    tau = .01
    rate_camera = 10

    state = 0

    hasGoal = False

    #panda = PickAndPlace(C, S, V, tau)

    time.sleep(10)
