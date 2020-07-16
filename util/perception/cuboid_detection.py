import cv2 as cv
import libry as ry
import numpy as np
import colorsys
from scipy.spatial import distance

from numpy import *
import sys
from operator import itemgetter
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from squaternion import Quaternion

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

def move_camera(t, camera, angle, radius):
    """
    This function moves the camera in a circle around the table.
    After 1000 steps the camera moves lower (to get a different angle and maybe more
    nice data)

    TODO: Maybe other camera movement?

    :param t: step
    :param camera: camera frame
    :param angle: angle in which camera need to move
    :param radius: around which camera is moving
    :return: updated angle
    """
    cam_px, cam_py, cam_pz = camera.getPosition()
    cam_q1, cam_q2, cam_q3, cam_q4 = camera.getQuaternion()
    q = Quaternion(cam_q1, cam_q2, cam_q3, cam_q4)
    e1, e2, e3 = q.to_euler(degrees=True)

    cam_px_new = cam_px + np.cos(angle) * radius
    cam_py_new = cam_py + np.sin(angle) * radius
    if t % 1000 == 0:
        cam_pz = cam_pz - 0.05
        e1 = e1 + 2.5

    quat_new = Quaternion.from_euler(e1, e2, e3 + 5, degrees=True)
    camera.setPosition([cam_px_new, cam_py_new, cam_pz])
    camera.setQuaternion([quat_new[0], quat_new[1], quat_new[2], quat_new[3]])
    angle += 0.0872665
    return angle

def mask_colored_object(hsv_colors, rgb):
    """
    Detects the color mask for each object

    :param hsv_colors: colors of the object in the RealView
    :param rgb: rgb image
    :return: dict of the detected objects with the center point, real color and detected color mask
    """
    obj_info = []
    obj_dict = {}
    hsv_image = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)
    for hsv in hsv_colors:
        center_points = []
        lower_color = np.array([hsv[0], 150, 150])
        upper_color = np.array([hsv[0], 255, 255])
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
            else:
                cX, cY = 0, 0

            obj_dict["center"] = (cX, cY)
            obj_dict["obj_color"] = cv.cvtColor(np.uint8([[[hsv[0], hsv[1], hsv[2]]]]), cv.COLOR_HSV2BGR)[0][0]
            obj_info.append(obj_dict.copy())

    return obj_info


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

# TODO fnish comments
def detectCuboids(S, camera, fxfypxpy, rgb, depth, objects, id, obj_info):
    """

    :param S: Simulation View - later for computing the pointcloud (to not do it in each step)
    :param camera: RealWorld Camera
    :param fxfypxpy: Also neccessary for the pointcloud
    :param rgb: rgb image
    :param depth: depth image
    :param objects: the tracked objects with dict of all object onfos
    :param id: id of the current object
    :param obj_info:
    :return:
    """
    # mask = obj_info['color_mask']
    mask = objects[id]["color_mask"]
    founded_sides = []
    gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

    number_of_sides_found = 0
    masked_image = gray * mask

    # get edges inside the colored object
    # canny edge detection
    edges = cv.Canny(masked_image, 15, 35)
    # kernel = np.ones((3, 3), np.uint8)
    edges = cv.dilate(edges, None, iterations=1)
    edges = cv.erode(edges, None, iterations=1)

    # find contours in edges
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

    # Grab only the innermost child components
    inner_contours = [c[0] for c in zip(contours, hierarchy)]
    sorted_contours = sort_contours(inner_contours)
    if sorted_contours:
        for cnt in sorted_contours:

            # if small contour area - ignore
            if cv.contourArea(cnt) < 200:
                continue

            # Ignore the objects which are too far away or in the background
            mask2 = np.zeros(rgb.shape[:2], np.uint8)
            cv.drawContours(mask2, cnt, -1, 255, 1)
            mean_depth = cv.mean(depth, mask=mask2)

            approx = cv.approxPolyDP(cnt, 0.04 * cv.arcLength(cnt, True), True)

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

                M = cv.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                # # if we found 3 sides of the cube then we csave the size and position of the object
                if number_of_sides_found == 3:
                    computeObjectInfo(founded_sides, camera, fxfypxpy, depth, objects, id, S)

                    # cv.circle(rgb, (cX, cY), 3, (255, 255, 255), -1)
                    # cv.putText(rgb, "center", (cX - 20, cY - 20),
                    #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # cv.drawContours(rgb, founded_sides, -1, (0, 0, 0), 2)

    if len(rgb) > 0: cv.imshow('OPENCV - rgb', rgb)
    return objects[id]["pos"], objects[id]["lenghtX"], objects[id]["lenghtY"], objects[id]["lenghtZ"]


def computeObjectInfo(founded_sides, camera, fxfypxpy, depth, objects, id, S):
    # side1, side2, side3 = founded_sides
    pointcloud = S.depthData2pointCloud(depth, fxfypxpy)

    cam_rot = camera.getRotationMatrix()
    cam_trans = camera.getPosition()
    # #
    # point11 = pointcloud[side1[0][1], side1[0][0]] @ cam_rot.T + cam_trans
    # point12 = pointcloud[side1[1][1], side1[1][0]] @ cam_rot.T + cam_trans
    # point13 = pointcloud[side1[2][1], side1[2][0]] @ cam_rot.T + cam_trans
    # point14 = pointcloud[side1[3][1], side1[3][0]] @ cam_rot.T + cam_trans
    #
    # point21 = pointcloud[side2[0][1], side2[0][0]] @ cam_rot.T + cam_trans
    # point22 = pointcloud[side2[1][1], side2[1][0]] @ cam_rot.T + cam_trans
    # point23 = pointcloud[side2[2][1], side2[2][0]] @ cam_rot.T + cam_trans
    # point24 = pointcloud[side2[3][1], side2[3][0]] @ cam_rot.T + cam_trans
    #
    # point31 = pointcloud[side3[0][1], side3[0][0]] @ cam_rot.T + cam_trans
    # point32 = pointcloud[side3[1][1], side3[1][0]] @ cam_rot.T + cam_trans
    # point33 = pointcloud[side3[2][1], side3[2][0]] @ cam_rot.T + cam_trans
    # point34 = pointcloud[side3[3][1], side3[3][0]] @ cam_rot.T + cam_trans
    # #
    # X, Y, Z = zip(point11, point12, point13, point14, point21, point22, point23, point24, point31, point32, point33,
    #               point34)
    # U, V, W = zip(point12 - point11, point13 - point12, point14 - point13, point11 - point14,
    #               point22 - point21, point23 - point22, point24 - point23, point21 - point24,
    #               point32 - point31, point33 - point32, point34 - point33, point31 - point34)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.quiver(X, Y, Z, U, V, W)
    # plt.show()

    vecx = [-1, 0, 0]
    vecy = [0, 1, 0]
    vecz = [0, 0, 1]

    # get 3d points of each corner image point
    points3d = []
    vectors3d = []

    numx = 0
    numy = 0
    numz = 0
    for side in founded_sides:
        points = []
        vectors = []
        for i, point in enumerate(side):
            points.append(pointcloud[point[1], point[0]]
                          @ cam_rot.T + cam_trans)
            if i == 3:
                vectors.append(
                    [points[1] - points[0], points[1] - points[2], points[2] - points[3], points[0] - points[3]])
                # Check if all sides in 90 degree to each other
                for j, v in enumerate(vectors[0]):
                    if j == 3:
                        vec_angle = angle_betweenVectors(v, vectors[0][0])
                    else:
                        vec_angle = angle_betweenVectors(v, vectors[0][j + 1])
                    if 70 > vec_angle or vec_angle > 110:
                        return

                    else:
                        if angle_betweenVectors(v, vecx) < 15 or angle_betweenVectors(v, vecx) > 170:
                            numx += 1
                            objects[id]["lenghtX"].append(np.linalg.norm(v))
                        elif angle_betweenVectors(v, vecy) < 15 or angle_betweenVectors(v, vecy) > 170:
                            numy += 1
                            objects[id]["lenghtY"].append(np.linalg.norm(v))
                        elif angle_betweenVectors(v, vecz) < 15:
                            numz += 1
                            objects[id]["lenghtZ"].append(np.linalg.norm(v))
        points3d.extend(points)
        vectors3d.extend(vectors)

    if numx != 4 or numy != 4 or numz != 4:
        objects[id]["lenghtX"] = objects[id]["lenghtX"][:-numx]
        objects[id]["lenghtY"] = objects[id]["lenghtY"][:-numy]
        objects[id]["lenghtZ"] = objects[id]["lenghtZ"][:-numz]
        return

    pos = points3d[0] - (points3d[0] - points3d[3]) / 2 - (points3d[0] - points3d[1]) / 2 - (
            points3d[5] - points3d[0]) / 2
    objects[id]["pos"].append(pos)

    pos = points3d[6] + (points3d[7] - points3d[6]) / 2 + (points3d[5] - points3d[6]) / 2 + (
            points3d[10] - points3d[6]) / 2
    objects[id]["pos"].append(pos)


def angle_betweenVectors(v1, v2):
    d = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
    angle = np.rad2deg(np.arccos(np.clip(d, -1, 1)))

    return angle


def add_comp_frame(id, objects, C):
    lenghtX = np.array(sort(objects[id]["lenghtX"]))
    obj_sideX = np.mean(lenghtX[abs(lenghtX - np.mean(lenghtX)) < 2 * np.std(lenghtX)])  # remove outlier

    lenghtY = np.array(sort(objects[id]["lenghtY"]))
    obj_sideY = np.mean(lenghtY[abs(lenghtY - np.mean(lenghtY)) < 2 * np.std(lenghtY)])  # remove outlier

    lenghtZ = np.array(sort(objects[id]["lenghtZ"]))
    obj_sideZ = np.mean(lenghtZ[abs(lenghtZ - np.mean(lenghtZ)) < 2 * np.std(lenghtZ)])  # remove outlier

    pos = remove_outliers(np.array(objects[id]["pos"]))
    pos_est = np.mean(pos, axis=0)

    if id == 4:
        obj = C.addFrame("goal")
    else:
        obj = C.addFrame("obj" + str(id))
    obj.setColor(objects[id]["obj_color"])
    obj.setShape(ry.ST.ssBox, [obj_sideX, obj_sideY, obj_sideZ, 0.01])
    obj.setPosition([pos_est[0], pos_est[1], pos_est[2]])
    # obj.setMass(1)
    obj.setContact(1)
    print(objects[id]["obj_color"])

# numpy.median is rather slow, let's build our own instead
def median(x):
    m, n = x.shape
    middle = np.arange((m - 1) >> 1, (m >> 1) + 1)
    x = np.partition(x, middle, axis=0)
    return x[middle].mean(axis=0)

# main function
def remove_outliers(data, thresh=2.0):
    m = median(data)
    s = np.abs(data - m)
    return data[(s < median(s) * thresh).all(axis=1)]
