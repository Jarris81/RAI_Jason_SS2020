import libry as ry

import numpy as np
import cv2 as cv

from util.setup import setup_color_challenge_env
from util.setup import setup_camera

import util.perception.cuboid_detection as pt

from util.perception.centroidtracker import CentroidTracker

if __name__ == "__main__":

    # setup env and get background
    R, S, C, V = setup_color_challenge_env()
    cameraFrame, fxfypxpy = setup_camera(C)  # the focal length

    # for moving camera at start in a circle
    camera = R.frame("camera")
    camera.setPosition([0, -.8, 1.85])  # TODO could also set in .g file

    # save the "real" colors of the objects from the RealView
    colors = pt.get_obj_infos(R)

    # create Centroid Tracker for tracking each object
    ct = CentroidTracker()

    rate_camera = 10
    tau = .01
    t = 0

    # for the camera rotation
    angle = 0
    radius = 0.085
    perception = True
    # Perception part - takes 10.000
    while perception:
        t += 1
        if t % rate_camera == 0:
            # update camera position and orientation at each step - move camera in circle
            angle = pt.move_camera(t, camera, angle, radius)

            [rgb, depth] = S.getImageAndDepth()  # we don't need images with 100Hz, rendering is slow

            # get the amount of objects and the "real" color - little cheat here :)
            # also creates an object info dict with infos like color and color mask
            obj_info = pt.mask_colored_object(colors, rgb)

            # object tracking
            # TODO objects don't disappear when they are move out from the cam
            objects = ct.update(obj_info)

            # track the pos and leght of the founded objects - if we have enough of each we can move to next step
            seen_obj = np.zeros(len(objects))
            # for each object save the sidelenght and position - here main part of object recognition
            for id, obj_info in objects.items():
                pos, x, y, z = pt.detectCuboids(S, camera, fxfypxpy, rgb, depth, objects, id, obj_info)
                # if we have enough data or each object we can then add the to the configuration space
                if len(pos) > 25 and len(x) > 20 and len(y) > 20 and len(z) > 20:
                    seen_obj[id] = 1

            if np.all((seen_obj == 1)):
                # now compute the average position and leght of each side and create a frame in
                # Configuration space
                for id, obj_info in objects.items():
                    pt.add_comp_frame(id, objects, C)
                    V.setConfiguration(C)
                perception = False

            # loop over the tracked objects
            for (objectID, obj_info) in objects.items():
                centroid = obj_info["center"]
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv.putText(rgb, text, (centroid[0] - 10, centroid[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv.circle(rgb, (centroid[0], centroid[1]), 3, (0, 255, 0), -1)

            if len(rgb) > 0: cv.imshow('OPENCV - rgb', rgb)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        S.step([], tau, ry.ControlMode.none)
    # TODO: decide on goal from the frames

    # TODO decide on Grasp - also add perception stuff. Maybe check object position with the last object IDs?
    cv.destroyAllWindows()
    R = 0
    S = 0
    C = 0
    V = 0