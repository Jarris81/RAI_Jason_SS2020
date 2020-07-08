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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from squaternion import Quaternion

if __name__ == "__main__":

    R, S, C, V = setup_color_challenge_env()
    cameraFrame, fxfypxpy = setup_camera(C)  # the focal length
    camera = R.frame("camera")
    camera.setPosition([0, -.6, 1.4])
    colors = get_obj_infos(R)
    # world = R.frame("world").getQuaternion()
    ct = CentroidTracker()

    euc_dist = []

    tau = .01
    t = 0
    angle = 0
    radius = 0.06

    while True:
        t += 1

        # grab sensor readings from the simulation
        q = S.get_q()

        if t % 20 == 0 and t > 200:
            cam_px, cam_py, cam_pz = camera.getPosition()
            cam_q1, cam_q2, cam_q3, cam_q4 = camera.getQuaternion()
            q = Quaternion(cam_q1, cam_q2, cam_q3, cam_q4)
            e1, e2, e3 = q.to_euler(degrees=True)
            quat_new = Quaternion.from_euler(e1 + 0.1, e2, e3 + 5, degrees=True)

            cam_px_new = cam_px + np.cos(angle) * radius;
            cam_py_new = cam_py + np.sin(angle) * radius;
            camera.setPosition([cam_px_new, cam_py_new, cam_pz - 0.001])
            camera.setQuaternion([quat_new[0], quat_new[1],quat_new[2],quat_new[3]])
            angle += 0.0872665


            [rgb, depth] = S.getImageAndDepth()  # we don't need images with 100Hz, rendering is slow
            points = S.depthData2pointCloud(depth, fxfypxpy)

            # obj_info = [tuble object_center, color_mask, object_color]
            obj_info = mask_colored_object(colors, rgb)

            objects = ct.update(obj_info)
            # pos0 = R.frame("obj0").getPosition()
            quat0 = R.frame("obj0").getQuaternion()
            q = Quaternion(quat0[0], quat0[1], quat0[2], quat0[3])
            # e = q.to_euler(degrees=True)
            for id, obj_info in objects.items():
                points, lenght = detectCuboids(objects, id, obj_info, rgb, depth, points, camera)


            if t == 3000:
                points.sort()
                print(sum(points[3:-3])/ len(points[3:-3]))
                # corner0 = []
                # kmeans = KMeans(n_clusters=8).fit(X)
                # y_pred = kmeans.predict(X)
                #
                # plt.figure('Corner Points', figsize=(7, 7))
                # ax = plt.axes(projection='3d')
                # plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, cmap='viridis')
                # ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                #            kmeans.cluster_centers_[:, 2],
                #            s=100, c='r', marker='*', label='Centroid')

                # plt.autoscale(enable=True, axis='x', tight=True)
                # plt.show()
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                #
                # # For each set of style and range settings, plot n random points in the box
                # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
                # for ptn in points:
                #     xs = ptn[0]
                #     ys = ptn[1]
                #     zs = ptn[2]
                #     ax.scatter(xs, ys, zs)
                #
                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Y Label')
                # ax.set_zlabel('Z Label')
                #
                # plt.show()
                break

            # loop over the tracked objects
            for (objectID, obj_info) in objects.items():
                centroid = obj_info["center"]
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv.putText(rgb, text, (centroid[0] - 10, centroid[1] - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.circle(rgb, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            #
            # good = np.zeros(rgb.shape, np.uint8)
            # cv.drawContours(good, good_contours, -1, (0, 255, 0), 1)
            #
            if len(rgb)>0: cv.imshow('OPENCV - rgb', rgb)
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
