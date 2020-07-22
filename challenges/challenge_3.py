import time

import libry as ry

from util.behavior import TowerBuilder
from util.setup import setup_camera
from util.setup import setup_env_subgoal_3
from util.transformations import quaternion_from_matrix

import util.perception.perception as pt

"""
Short example for testing the transition library, 
and building a state machine for the different primitives
"""


def cheat_update_obj(obj):
    C.addFrame(obj)
    C.frame(obj).setPosition(S.getGroundTruthPosition(obj))
    size = S.getGroundTruthSize(obj)
    if len(size) == 3:
        C.frame(obj).setShape(ry.ST.box, size=size)
    else:
        C.frame(obj).setShape(ry.ST.ssBox, size=size)
    C.frame(obj).setQuaternion(quaternion_from_matrix(S.getGroundTruthRotationMatrix(obj)))
    C.frame(obj).setContact(1)
    return obj


if __name__ == "__main__":

    usePerception = True
    # setup env and get background
    R, S, C, V, back_frame = setup_env_subgoal_3(False)
    cameraFrame, fxfypxpy = setup_camera(C)  # the focal length
    tau = .01
    rate_camera = 10

    state = 0

    hasGoal = False

    panda = TowerBuilder(C, S, V, tau)
    # for moving camera at start in a circle
    camera = R.frame("camera")
    camera.setPosition([0.6, -.85, 1.85])

    # used for shortcutting perception
    num_blocks = 2
    t = 0

    if usePerception:

        perception = pt.Perception(R, S, C, V, camera, fxfypxpy)
        perception.init_get_real_colors()
        perception.runs = False

        while perception.runs:
            t += 1
            # time.sleep(0.01)
            perception.step(t)

    while True:
        t += 1
        time.sleep(tau)

        # frame rate of camera, do perception here
        if t > 200 and not t % rate_camera:
            # set blocks in config and add
            panda.set_blocks([cheat_update_obj("obj%i" % i) for i in range(num_blocks)])
            # panda.set_blocks(perception.computed_blocks)
        panda.step(t)

    print("Simulation is done")
    time.sleep(5)
