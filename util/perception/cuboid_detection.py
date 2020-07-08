import cv2 as cv
import numpy as np
import colorsys
from scipy.spatial import distance

from numpy import *
import sys
from operator import itemgetter
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import mayavi.mlab as m

"""

"""


def get_obj_infos(R):
    obj_info = []

    names = R.getFrameNames()
    for name in names:
        if name.startswith("obj"):
            rgb_color = R.frame(name).info()['color']
            hsv_color = colorsys.rgb_to_hsv(rgb_color[2], rgb_color[1], rgb_color[0])
            obj_info.append([hsv_color[0] * 180, hsv_color[1] * 255, hsv_color[2] * 255])

    return obj_info


def mask_colored_object(hsv_colors, rgb):
    obj_info = []
    obj_dict = {}
    hsv_image = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)
    for hsv in hsv_colors:
        center_points = []
        lower_color = np.array([hsv[0] - 1, 150, 150])
        upper_color = np.array([hsv[0] + 1, 255, 255])
        mask = cv.inRange(hsv_image, lower_color, upper_color)

        obj_dict["color_mask"] = mask

        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if contours:

            # compute center point of the color object --> this one we want track later
            M = cv.moments(contours[0])
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # if center point already in list (some contours are doubled) then skip
                if (cX, cY) in center_points:
                    break
                # center_points.append((cX, cY))
            else:
                cX, cY = 0, 0

            obj_dict["center"] = (cX, cY)
            obj_dict["obj_color"] = hsv
            obj_info.append(obj_dict.copy())

    return obj_info


def computeObjectSize(a, b, c, d):
    euc_dist = []

    # minRect = cv.minAreaRect(one_hull)
    side1 = np.linalg.norm(a - c)
    side2 = np.linalg.norm(b - d)
    side3 = np.linalg.norm(a - b)
    side4 = np.linalg.norm(c - d)
    euc_dist.append((side1 + side2) / 2)
    euc_dist.append((side3 + side4) / 2)
    # Which points have the same edge? Sometimes the corner points are not wrong
    return euc_dist


# Sort Contours on the basis of their x-axis coordinates in ascending order (from left to right)
def sort_contours(contours):
    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes)
                                            , key=lambda b: b[1][0], reverse=False))
    # return the list of sorted contours
    return contours


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    # D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    # (br, tr) = rightMost[np.argsort(D)[::-1], :]
    if rightMost[0][1] > rightMost[1][1]:
        br = rightMost[0]
        tr = rightMost[1]
    else:
        br = rightMost[1]
        tr = rightMost[0]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int32")


def detectCuboids(objects, id, obj_info, rgb, depth, pointcloud, camera):
    mask = obj_info['color_mask']
    founded_sides = []
    euc_dist = []
    gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

    number_of_sides_found = 0
    masked_image = gray * mask

    # get edges inside the colored object
    # canny edge detection
    edges = cv.Canny(masked_image, 0, 15)
    # kernel = np.ones((3, 3), np.uint8)
    edges = cv.dilate(edges, None, iterations=1)
    edges = cv.erode(edges, None, iterations=1)

    # find contours in edges
    contours, hierarchy = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

    # Grab only the innermost child components
    inner_contours = [c[0] for c in zip(contours, hierarchy)]
    if contours:
        sorted_contours = sort_contours(contours)
        for cnt in sorted_contours:

            # if small contour area - ignore
            if cv.contourArea(cnt) < 200:
                continue

            # Ignore the objects which are too far away or in the background
            mask2 = np.zeros(rgb.shape[:2], np.uint8)
            cv.drawContours(mask2, cnt, -1, 255, 1)
            mean_depth = cv.mean(depth, mask=mask2)

            approx = cv.approxPolyDP(cnt, 0.03 * cv.arcLength(cnt, True), True)

            if mean_depth[0] < 2 and len(approx) == 4:

                number_of_sides_found += 1
                # good_contour.append(cnt)
                hull = cv.convexHull(approx, False)

                lenght = len(hull)
                if len(hull) != 4:
                    break

                a, b, c, d = hull
                # corner_points = [a[0].tolist(), b[0].tolist(), c[0].tolist(), d[0].tolist()]
                corner_points = np.array([a[0], b[0], c[0], d[0]])
                # order the point in top-left, top-right, bottom-right and bottom-left
                corner_points_ordered = order_points(corner_points)

                # hull_num = cv.convexHull(approx, returnPoints=False)
                founded_sides.append(corner_points_ordered)

                # # if we found 3 sides of the cube then we compute the width, height and depth
                if number_of_sides_found == 3:
                    computeObjectInfo(founded_sides, camera, pointcloud, depth, objects, id)
                    # founded_sides_sorted = founded_sides.sort()
                    # eucl = []
                    #


                    #     eucl.extend(computeObjectSize(point1, point2, point3, point4))
                    #
                    # eucl.sort()
                    # for p1 in side1:
                    #     for p2 in side2:
                    #         for p3 in side3:
                    #             if (np.linalg(np.array(p3) - np.array(p2))) < 4:
                    #                 side23.append((np.array(p3) - np.array(p2))/2)
                    #             if (np.linalg(np.array(p3) - np.array(p1))) < 4:
                    #                 side13.append((np.array(p3) - np.array(p1)) / 2)
                    # if (np.array(p2) - np.array(p1)) / 2 < 4:
                    #     side12.append((np.array(p2) - np.array(p1)) / 2)

                    # side1.sort()
                    # side2.sort()
                    # side3.sort()
                    # first_set = set(map(tuple, side1))
                    # secnd_set = set(map(tuple, side2))
                    # third_set = set(map(tuple, side3))
                    # print(first_set & secnd_set)
                    # print(first_set & third_set)
                    # print(third_set & secnd_set)

                    # remove_dubl = list(k for k, _ in itertools.groupby(founded_sides))
                    # lenght_list = len(remove_dubl)
                    # for i, (x,y) in enumerate(remove_dubl, 1):
                    #     if (i < lenght_list):
                    #         if((remove_dubl[i][0]- remove_dubl[i-1][0])/2) < 3 or ((remove_dubl[i][1]- remove_dubl[i-1][1])/2) < 3:
                    #             #remove_dubl[i-1] = (np.array(remove_dubl[i]) + np.array(remove_dubl[i-1]) / 2).tolist()
                    #             remove_dubl.remove(remove_dubl[i])
                    #             lenght_list -= 1

                    # if len(remove_dubl) != 8:
                    #     break
                    #
                    # remove_dubl.sort()
                    # print(remove_dubl)
                    # compute the center of the contour
                    M = cv.moments(approx)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # draw the contour and center of the shape on the image
                        # if 110 < cX < 300 and 110 < cY < 600:
                        # point1 = pointcloud[corner_points[0][1], corner_points[0][0]] @ cam_rot.T + cam_trans
                        # point2 = pointcloud[corner_points[1][1], corner_points[1][0]] @ cam_rot.T + cam_trans
                        # point3 = pointcloud[corner_points[2][1], corner_points[2][0]] @ cam_rot.T + cam_trans
                        # point4 = pointcloud[corner_points[3][1], corner_points[3][0]] @ cam_rot.T + cam_trans
                        # lenght = computeObjectSize(point1, point2, point3, point4)
                        # objects[id]["lenght"].extend(lenght)
                        # objects[id]["points"].append(pointcloud[cY, cX] @ cam_rot.T + cam_trans)

                        # cont_3D = tf.camera_to_world(point1, camera, fxfypxpy)

                        # objects[id]["points"].append(point1.tolist())
                        # objects[id]["points"].append(point2.tolist())
                        # objects[id]["points"].append(point3.tolist())
                        # objects[id]["points"].append(point4.tolist())

                        cv.circle(rgb, (cX, cY), 3, (255, 255, 255), -1)
                        cv.putText(rgb, "center", (cX - 20, cY - 20),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # cv.drawContours(rgb, founded_sides, -1, (0, 0, 0), 2)

    if len(rgb) > 0: cv.imshow('OPENCV - rgb', rgb)
    return objects[id]["points"], objects[id]["lenght"]

def computeObjectInfo(founded_sides, camera, pointcloud, depth, objects, id):
    side1, side2, side3 = founded_sides
    # if the 3 sides are overlapping than pass
    if np.linalg.norm(side1[3] - side2[3]) < 3 or np.linalg.norm(side1[3] - side3[3]) < 3:
        return
    cam_rot = camera.getRotationMatrix()
    cam_trans = camera.getPosition()

    point11 = pointcloud[side1[0][1]+1, side1[0][0]] @ cam_rot.T + cam_trans
    point12 = pointcloud[side1[1][1], side1[1][0]] @ cam_rot.T + cam_trans
    point13 = pointcloud[side1[2][1], side1[2][0]] @ cam_rot.T + cam_trans
    point14 = pointcloud[side1[3][1], side1[3][0]] @ cam_rot.T + cam_trans
    point21 = pointcloud[side2[0][1], side2[0][0]] @ cam_rot.T + cam_trans
    point22 = pointcloud[side2[1][1], side2[1][0]] @ cam_rot.T + cam_trans
    depth22 = depth[side2[1][1]][side2[1][0]]
    point23 = pointcloud[side2[2][1], side2[2][0]-2] @ cam_rot.T + cam_trans
    depth23 = depth[side2[2][1], side2[2][0]]
    point24 = pointcloud[side2[3][1], side2[3][0]] @ cam_rot.T + cam_trans
    depth24 = depth[side2[3][1], side2[3][0]]
    point31 = pointcloud[side3[0][1], side3[0][0]] @ cam_rot.T + cam_trans
    depth31 = depth[side3[0][1], side3[0][0]]
    point32 = pointcloud[side3[1][1], side3[1][0]] @ cam_rot.T + cam_trans
    depth32 = depth[side3[1][1], side3[1][0]]
    point33 = pointcloud[side3[2][1], side3[2][0]] @ cam_rot.T + cam_trans
    point34 = pointcloud[side3[3][1], side3[3][0]] @ cam_rot.T + cam_trans

    origin = [0, 0, 0]
    X, Y, Z = zip(point11, point12, point13, point14, point21, point22, point23, point24, point31, point32, point33, point34)
    U, V, W = zip(point12-point11, point13-point12, point14-point13, point11- point14,
                  point22-point21, point23-point22, point24-point23, point21- point24,
                  point32-point31, point33-point32, point34-point33, point31- point34)


    # X, Y, Z = zip(point11, point12, point13, point14, point21, point22, point23, point24)
    # U, V, W = zip(point12-point11, point13-point12, point14-point13, point11- point14,
    #               point22-point21, point23-point22, point24-point23, point21- point24)

    lenght3 = (np.linalg.norm(point12 - point11) + np.linalg.norm(point14-point13) + np.linalg.norm(point23-point22) + np.linalg.norm(point21-point24))/4
    lenght1 = (np.linalg.norm(point22-point21) + np.linalg.norm(point24-point23) + np.linalg.norm(point32-point31) + np.linalg.norm(point34-point33)) /4
    lenght2 = (np.linalg.norm(point13-point12) + np.linalg.norm(point11-point14) + np.linalg.norm(point33-point32) + np.linalg.norm(point31-point34)) /4

    objects[id]["points"].append(lenght3)
    # print("lenght1: " + str(lenght1) + " lenght2: " + str(lenght2) + " lenght3: " + str(lenght3))
    # if(lenght1 > 15 and lenght2 > 9 and lenght3 > 5):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.quiver(X, Y, Z, U, V, W)
    #     plt.show()
    #     print(lenght1)
    # lenght1 = np.linalg.norm(point12 - point11)

    vec1 = point22-point21
    vecx = np.array([1, 0, 0])

    vec3 = point13-point12
    vecz = np.array([0, -1, 0])

    vec1_unit = vec1 / np.linalg.norm(vec1)
    vec3_unit = vec3 / np.linalg.norm(vec3)

    # c = np.dot(vec1_unit / np.linalg.norm(vec1_unit), vecx / np.linalg.norm(vecx))
    # anglex = np.rad2deg(np.arccos(np.clip(c, -1, 1)))

    d = np.dot(vec3_unit / np.linalg.norm(vec3_unit), vecz / np.linalg.norm(vecz))
    angley = np.rad2deg(np.arccos(np.clip(d, -1, 1)))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.quiver(X, Y, Z, U, V, W)
    # plt.show()
    #
    # print(angle)