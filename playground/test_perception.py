import libry as ry
import time
import numpy as np
import cv2 as cv

import util
from util.setup import setup_color_challenge_env
from util.setup import setup_camera
from util.perception.centroidtracker import CentroidTracker

from util.perception.cuboid_detection import detectCuboids
from util.perception.cuboid_detection import get_obj_infos
from util.perception.cuboid_detection import mask_colored_object


if __name__ == "__main__":

    R, S, C, V = setup_color_challenge_env()
    cameraFrame, fxfypxpy = setup_camera(C)  # the focal length

    obj_info = get_obj_infos(R)

    ct = CentroidTracker()

    tau = .01
    t = 0

    while True:
        time.sleep(0.02)
        t += 1
        # grab sensor readings from the simulation
        q = S.get_q()

        if t % 10 == 0:
            [rgb, depth] = S.getImageAndDepth()  # we don't need images with 100Hz, rendering is slow

            color_masks, center_points = mask_colored_object(obj_info, rgb)

            objects = ct.update(center_points)

            # loop over the tracked objects
            # for (objectID, centroid) in objects.items():
            #     # draw both the ID of the object and the centroid of the
            #     # object on the output frame
            #     text = "ID {}".format(objectID)
            #     cv.putText(rgb, text, (centroid[0] - 10, centroid[1] - 10),
            #                 cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #     cv.circle(rgb, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            detectCuboids(color_masks, rgb=rgb, depth=depth)

            #
            # good = np.zeros(rgb.shape, np.uint8)
            # cv.drawContours(good, good_contours, -1, (0, 255, 0), 1)
            #
            #if len(rgb)>0: cv.imshow('OPENCV - rgb', rgb)
            # if len(edges)>0: cv.imshow('OPENCV - gray_bgd', edges)
            # if len(good)>0: cv.imshow('OPENCV - depth', good)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        S.step([], tau, ry.ControlMode.none)

    cv.destroyAllWindows()
    R = 0
    S = 0
    C = 0
    V = 0