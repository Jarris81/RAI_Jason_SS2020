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
    camera = R.frame("camera")
    colors = get_obj_infos(R)

    ct = CentroidTracker()

    euc_dist = []

    tau = .01
    t = 0


    while True:
        time.sleep(0.02)
        t += 1


        # grab sensor readings from the simulation
        q = S.get_q()

        S.step([], tau, ry.ControlMode.none)
        [y, J] = R.evalFeature(ry.FS.position, ["L_gripperCenter"])
        camera.setPosition(y)
        [y, J] = R.evalFeature(ry.FS.quaternion, ["L_gripperCenter"])
        camera.setQuaternion(y)

        [y, J] = R.evalFeature(ry.FS.positionDiff, ["L_gripperCenter", "obj0"])
        vel = J.T @ np.linalg.inv(J @ J.T + 1e-2 * np.eye(y.shape[0])) @ (-y)
        S.step(vel, tau, ry.ControlMode.velocity)

        if t % 10 == 0 and t > 50:
            [rgb, depth] = S.getImageAndDepth()  # we don't need images with 100Hz, rendering is slow
            points = S.depthData2pointCloud(depth, fxfypxpy)

            # obj_info = [tuble object_center, color_mask, object_color]
            color_masks, obj_info = mask_colored_object(colors, rgb)

            objects = ct.update(obj_info)

            pos0 = R.frame("obj0").getPosition()
            for id, obj_info in objects.items():
                detectCuboids(obj_info, rgb, depth, points)


            #loop over the tracked objects
            # for (objectID, centroid) in objects.items():
            #     # draw both the ID of the object and the centroid of the
            #     # object on the output frame
            #     text = "ID {}".format(objectID)
            #     cv.putText(rgb, text, (centroid[0] - 10, centroid[1] - 10),
            #                 cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #     cv.circle(rgb, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


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