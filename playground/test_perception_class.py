import time

from util.setup import setup_color_challenge_env
from util.setup import setup_camera

import util.perception.perception as pt

if __name__ == "__main__":

    # setup env and get background
    R, S, C, V = setup_color_challenge_env()
    cameraFrame, fxfypxpy = setup_camera(C)  # the focal length

    # for moving camera at start in a circle
    camera = R.frame("camera")
    camera.setPosition([0, -.85, 1.85])  # TODO could also set in .g file

    perception = pt.Perception(R, S, C, V, camera, fxfypxpy)
    perception.init_get_real_colors()

    t = 0

    while True:
        t += 1
        # time.sleep(0.01)
        perception.step(t)