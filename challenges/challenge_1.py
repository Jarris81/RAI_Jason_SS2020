import libry as ry
from util.setup import setup_env_subgoal_1
from util.setup import setup_camera
import time

from util.behavior import TowerBuilder
from util.transformations import quaternion_from_matrix

import util.perception.perception as pt

"""
Short example for testing the transition library, 
and building a state machine for the different primitives
"""


def cheat_update_obj(obj):
    C.addFrame(obj)
    C.frame(obj).setPosition(S.getGroundTruthPosition(obj))
    C.frame(obj).setShape(ry.ST.box, size=S.getGroundTruthSize(obj))
    C.frame(obj).setQuaternion(quaternion_from_matrix(S.getGroundTruthRotationMatrix(obj)))
    C.frame(obj).setContact(1)


if __name__ == "__main__":

    # setup env and get background
    R, S, C, V, back_frame = setup_env_subgoal_1(False)
    cameraFrame, fxfypxpy = setup_camera(C)  # the focal length
    tau = .001
    rate_camera = 10

    panda = TowerBuilder(C, S, V, tau)

    # for moving camera at start in a circle
    camera = R.frame("camera")
    camera.setPosition([0, -.85, 1.55])  # TODO could also set in .g file

    # perception = pt.Perception(R, S, C, V, camera, fxfypxpy)
    # perception.init_get_real_colors()
    # perception.runs = True

    t = 0

    # while perception.runs:
    #     t += 1
    #     # time.sleep(0.01)
    #     perception.step(t)

    while True:
        time.sleep(0.01)
        t += 1
        # frame rate of camera, do perception here
        # if t > 100 and not t % rate_camera:
            # panda.set_blocks(perception.computed_blocks)

        panda.step(t)

    time.sleep(5)
