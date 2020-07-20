import libry as ry
from util.setup import setup_env_subgoal_1
from util.setup import setup_camera
import time

from util.behavior import TowerBuilder
from util.transformations import quaternion_from_matrix

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
    cameraFrame, fxfypxpy = setup_camera(C)    # the focal length
    tau = .001
    rate_camera = 10

    panda = TowerBuilder(C, S, V, tau)

    for t in range(5000):
        time.sleep(tau)

        # frame rate of camera, do perception here
        if t > 100 and not t%rate_camera:
            cheat_update_obj("obj0")
            cheat_update_obj("obj1")
            panda.set_blocks(["obj0", "obj1"])

        panda.step(t)

    time.sleep(5)
