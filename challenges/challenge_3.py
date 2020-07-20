import time

import libry as ry

from util.behavior import TowerBuilder
from util.setup import setup_camera
from util.setup import setup_env_subgoal_3
from util.transformations import quaternion_from_matrix

"""
Short example for testing the transition library, 
and building a state machine for the different primitives
"""


def cheat_update_obj(obj):
    C.addFrame(obj)
    C.frame(obj).setPosition(S.getGroundTruthPosition(obj))
    C.frame(obj).setShape(ry.ST.ssBox, size=S.getGroundTruthSize(obj))
    C.frame(obj).setQuaternion(quaternion_from_matrix(S.getGroundTruthRotationMatrix(obj)))
    C.frame(obj).setContact(1)
    return obj


if __name__ == "__main__":

    # setup env and get background
    R, S, C, V, back_frame = setup_env_subgoal_3(False)
    cameraFrame, fxfypxpy = setup_camera(C)    # the focal length
    tau = .001
    rate_camera = 10

    state = 0

    hasGoal = False

    panda = TowerBuilder(C, S, V, tau)

    # used for shortcutting perception
    num_blocks = 2

    for t in range(10000):
        time.sleep(tau)

        # frame rate of camera, do perception here
        if t > 100 and not t % rate_camera:
            # set blocks in config and add
            panda.set_blocks([cheat_update_obj("obj%i" % i) for i in range(num_blocks)])

        panda.step(t)
    print("Simulation is done")
    time.sleep(5)
