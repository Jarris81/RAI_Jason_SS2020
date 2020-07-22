import time, os

import libry as ry

from util.behavior import TowerBuilder

import util.perception.perception as pt

from util.setup import setup_camera
from util.setup import setup_env_subgoal_4

from util.setup import setup_color_challenge_env
from util.transformations import quaternion_from_matrix


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

    # setup env and get background
    R, S, C, V = setup_color_challenge_env()
    cameraFrame, fxfypxpy = setup_camera(C)  # the focal length
    tau = .001
    rate_camera = 10

    # for moving camera at start in a circle
    camera = R.frame("camera")
    camera.setPosition([0, -.85, 1.75])  # TODO could also set in .g file

    panda = TowerBuilder(C, S, V, tau)

    perception = pt.Perception(R, S, C, V, camera, fxfypxpy)
    perception.init_get_real_colors()
    perception.runs = False

    t = 0

    while perception.runs:
        t += 1
        # time.sleep(0.01)
        perception.step(t)

    duration = 1.5  # seconds
    freq = 500  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

    time.sleep(5)
    # used for shortcutting perception
    num_blocks = 4

    for t in range(100000):
        time.sleep(tau)

        # frame rate of camera, do perception here
        if t > 200 and not t % rate_camera:
            # set blocks in config and add
            panda.set_blocks([cheat_update_obj("obj%i" % i) for i in range(num_blocks)])
            # panda.set_blocks(perception.computed_blocks)
        panda.step(t)
    print("Simulation is done")
    time.sleep(5)
