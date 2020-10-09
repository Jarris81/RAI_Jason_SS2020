import time

import libry as ry

from util.behavior import TowerBuilder
from util.setup import setup_camera
from util.setup import setup_env_subgoal_2
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
    return obj


if __name__ == "__main__":

    # set True if to use perception, else use shortcut
    usePercption = False

    # setup env and get background
    R, S, C, V, back_frame = setup_env_subgoal_2(False)
    cameraFrame, fxfypxpy = setup_camera(C)  # the focal length
    tau = .01
    rate_camera = 10

    # behavior we want for robot
    panda = TowerBuilder(C, S, V, tau)

    # for moving camera at start in a circle
    camera = R.frame("camera")
    camera.setPosition([0, -.85, 1.85])

    # used for shortcutting perception
    num_blocks = 5

    t = 0

    if usePercption:
        perception = pt.Perception(R, S, C, V, camera, fxfypxpy)
        perception.init_get_real_colors()
        perception.runs = True
        while perception.runs:
            t += 1
            # time.sleep(0.01)
            perception.step(t)

    while True:
        time.sleep(tau)
        t += 1
        # frame rate of camera, do perception here
        if t > 100 and not t % rate_camera:
            if usePercption:
                panda.set_blocks(perception.computed_blocks)
            else:
                panda.set_blocks([cheat_update_obj("obj%i" % i) for i in range(num_blocks)])

        panda.step(t)
    time.sleep(5)
