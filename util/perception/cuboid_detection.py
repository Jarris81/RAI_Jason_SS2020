import cv2 as cv
import numpy as np
import colorsys
from scipy.spatial import distance

from numpy import *
import sys

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
    color_masks = []
    obj_info = []
    obj_dict = {}
    hsv_image = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)
    for hsv in hsv_colors:
        lower_color = np.array([hsv[0], 150, 150])
        upper_color = np.array([hsv[0], 255, 255])
        mask = cv.inRange(hsv_image, lower_color, upper_color)
        color_masks.append([mask, hsv])

        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if contours:
            # compute center point of the color object --> this one we want track later
            M = cv.moments(contours[0])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # if center point already in list (some contours are doubled) then skip
            if (cX, cY) in obj_info:
                break

            obj_dict["center"] = (cX, cY)
            obj_dict["color_mask"] = mask
            obj_dict["obj_color"] = hsv
            obj_info.append(obj_dict.copy())

    return color_masks, obj_info


def computeObjectSize(object_hulls, points):
    euc_dist = []
    for one_hull in object_hulls:
        point_corners = []
        for i, corner in enumerate(one_hull):
            point = points[corner[0][1], corner[0][0]]
            point_corners.append(point)

        # minRect = cv.minAreaRect(one_hull)
        side1 = np.linalg.norm(point_corners[0] - point_corners[1])
        side2 = np.linalg.norm(point_corners[1] - point_corners[2])
        side3 = np.linalg.norm(point_corners[2] - point_corners[3])
        side4 = np.linalg.norm(point_corners[3] - point_corners[0])
        euc_dist.append((side1 + side3) / 2)
        euc_dist.append((side2 + side4) / 2)
    # Which points have the same edge? Sometimes the corner points are not wrong
    euc_dist = sorted(euc_dist)
    return euc_dist


def detectCuboids(obj_info, rgb, depth, pointcloud, show=False):
    mask = obj_info['color_mask']


    euc_dist = []
    gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

    number_of_sides_found = 0
    foundes_sides = []
    masked_image = gray * mask

    # get edges inside the colored object
    # canny edge detection
    edges = cv.Canny(masked_image, 20, 50)
    # kernel = np.ones((3, 3), np.uint8)
    edges = cv.dilate(edges, None, iterations=1)
    edges = cv.erode(edges, None, iterations=1)

    # find contours in edges
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

    for component in zip(contours, hierarchy):
        cnt = component[0]
        currentHierarchy = component[1]

        # if small contour area - ignore
        if cv.contourArea(cnt) < 200:
            continue

        if currentHierarchy[2] < 0:
            # Ignore the objects which are too far away or in the background
            mask = np.zeros(rgb.shape[:2], np.uint8)
            cv.drawContours(mask, cnt, -1, 255, 1)
            mean_color = cv.mean(rgb, mask=mask)
            mean_depth = cv.mean(depth, mask=mask)

            approx = cv.approxPolyDP(cnt, 0.04 * cv.arcLength(cnt, True), True)
            # print(len(approx))
            if mean_depth[0] < 2 and len(approx) < 5:

                number_of_sides_found += 1
                # good_contour.append(cnt)
                hull = cv.convexHull(approx, False)

                hull_num = cv.convexHull(approx, returnPoints=False)
                foundes_sides.append(hull)

                # if we found 3 sides of the cube then we compute the width, height and depth
                if number_of_sides_found == 3:
                    euc_dist = computeObjectSize(foundes_sides, pointcloud)

                # compute the center of the contour
                # M = cv.moments(approx)
                # if M["m00"] != 0:
                #     cX = int(M["m10"] / M["m00"])
                #     cY = int(M["m01"] / M["m00"])
                #     # draw the contour and center of the shape on the image
                #     cv.circle(rgb, (cX, cY), 3, (255, 255, 255), -1)
                #     cv.putText(rgb, "center", (cX - 20, cY - 20),
                #                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                #
                #     cv.drawContours(rgb, hull, -1, (0, 0, 0), 2)

    if len(rgb) > 0: cv.imshow('OPENCV - rgb', rgb)
    return obj_info