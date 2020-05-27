import cv2 as cv
import time
import numpy as np
import util.geom as geom
import util.transformations as tf
from skimage.metrics import structural_similarity


def get_red_ball_contours(frame, background, cameraFrame, fxypxy, vis=False):
    # get all masks we want to apply, to get the red ball
    score, diff = get_diff_combined(background, frame, color=False)
    diff = (diff * 255).astype(np.uint8)  # convert to 255 image
    mask1 = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    mask2 = filter_color(frame[0], [115, 100, 100], [130, 255, 255])  # red filter
    mask = np.bitwise_and(mask1.astype(bool), mask2.astype(bool)).astype(np.uint8)  # combine masks

    # apply filters
    rgb_masked = cv.bitwise_and(frame[0], frame[0], mask=mask)
    depth_masked = cv.bitwise_and(frame[1], frame[1], mask=mask)

    # get contours, will be used to count objects
    contours, __ = get_contours(mask)

    # if no contours was detected, we can stop
    if not len(contours):
        return []

    if vis:
        if len(contours) > 0:
            cv.drawContours(rgb_masked, contours, -1, (0, 255, 0), 1)
        if len(diff) > 0:
            cv.imshow('OPENCV - diff', convert_rgb_to_bgr(rgb_masked))
        cv.waitKey(1)
    # if len(depth) > 0: cv.imshow('OPENCV - depth', 0.5 * depth)
    #
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break
    red_ball_mid_points = []

    for cont in contours:
        cont = np.vstack(cont)
        mask = np.zeros_like(frame[1]).astype(np.bool)
        mask[cont[:, 1], cont[:, 0]] = 1
        depths = frame[1][mask]

        depths = np.reshape(depths, (len(depths), 1))
        if depths.shape[0] != cont.shape[0] or len(cont) < 30:
            continue
        cont_depth = np.hstack((cont, depths))
        cont_3D = np.asarray([tf.camera_to_world(pt, cameraFrame, fxypxy) for pt in cont_depth])

        red_ball_mid_points.append(cont_3D.mean(axis=0))
    return np.asarray(red_ball_mid_points)


def get_red_ball_hough(frame, background, cameraFrame, fxypxy, vis=False):
    # get all masks we want to apply, to get the red ball
    score, diff = get_diff_combined(background, frame, color=False)
    diff = (diff * 255).astype(np.uint8)  # convert to 255 image
    mask1 = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    mask2 = filter_color(frame[0], [115, 100, 100], [130, 255, 255])  # red filter
    mask = np.bitwise_and(mask1.astype(bool), mask2.astype(bool)).astype(np.uint8)  # combine masks

    # apply filters
    rgb_masked = cv.bitwise_and(frame[0], frame[0], mask=mask)
    depth_masked = cv.bitwise_and(frame[1], frame[1], mask=mask)

    # get contours, will be used to count objects
    rows = rgb_masked.shape[0]
    mask_im = (mask * 255).astype(np.uint8)
    circles =cv.HoughCircles(mask_im, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=10,
                               minRadius=5, maxRadius=12)

    # if no contours was detected, we can stop
    if vis:
        cv.imshow('OPENCV - mask', mask_im)
        cv.waitKey(1)
    if circles is None:
        return []
    #white = np.ones_like(mask_im)
    if vis:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(frame[0], center, 1, (0, 0, 255), 2)
            # circle outline
            radius = i[2]
            cv.circle(frame[0], center, radius, (0, 255, 0), 2)
        if len(circles) > 0:
            cv.imshow('OPENCV - circles', frame[0])
        cv.waitKey(1)
    print(len(circles))
    return circles

    # for cont in contours:
    #     cont = np.vstack(cont)
    #     mask = np.zeros_like(frame[1]).astype(np.bool)
    #     mask[cont[:, 1], cont[:, 0]] = 1
    #     depths = frame[1][mask]
    #
    #     depths = np.reshape(depths, (len(depths), 1))
    #     if depths.shape[0] != cont.shape[0]:
    #         continue
    #     cont_depth = np.hstack((cont, depths))
    #     cont_3D = np.asarray([tf.camera_to_world(pt, cameraFrame, fxypxy) for pt in cont_depth])
    #
    #     red_ball_mid_points.append(cont_3D.mean(axis=0))
    # return np.asarray(red_ball_mid_points)


def extract_background(S, duration=5, fps=1):

    n = duration * fps
    background_rgb, background_depth = [], []

    for i in range(n):
        [rgb, depth] = S.getImageAndDepth()
        background_rgb.append(rgb)
        background_depth.append(depth)
        time.sleep(1/fps)

    background_rgb = np.average(background_rgb, axis=0).astype(np.uint8)
    background_depth = np.average(background_depth, axis=0).astype(np.float32)

    return [background_rgb, background_depth]


def get_diff_rgb(ref_rgb, src_rgb, color=False):
    if not color:
        ref_rgb = cv.cvtColor(ref_rgb, cv.COLOR_BGR2GRAY)
        src_rgb = cv.cvtColor(src_rgb, cv.COLOR_BGR2GRAY)
    (score_rgb, diff_rgb) = structural_similarity(ref_rgb, src_rgb, full=True, multichannel=color)
    return score_rgb, diff_rgb


def get_diff_depth(ref_depth, src_depth, margin=0):
    (score_depth, diff_depth) = structural_similarity(ref_depth, src_depth, full=True)
    if margin > 0:
        temp = diff_depth[diff_depth > margin]
        score_depth = sum(temp)/len(temp)
    return score_depth, diff_depth


def get_diff_combined(ref_frame, src_frame, weight_rgb=0.5, margin=(0,0) ,scale=(1, 1), color=False):

    score_rgb, diff_rgb = get_diff_rgb(ref_frame[0], src_frame[0], color=color)
    score_depth, diff_depth = get_diff_depth(ref_frame[1], src_frame[1], margin=margin[1])
    if color:
        diff_rgb = cv.cvtColor(diff_rgb, cv.COLOR_BGR2GRAY)

    combined_diff = diff_rgb * weight_rgb + diff_depth * (1-weight_rgb)
    # only include actual differences bigger than margin for score

    combined_score = score_rgb * weight_rgb + score_depth * (1 - weight_rgb)
    combined_diff = (combined_diff/np.amax(combined_diff))
    return combined_score, combined_diff


def filter_color(rgb, lower_hsv_limit, upper_hsv_limit):
    lower_hsv_limit = np.asarray(lower_hsv_limit)
    upper_hsv_limit = np.asarray(upper_hsv_limit)
    hsv = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_hsv_limit, upper_hsv_limit)

    return mask


def convert_rgb_to_bgr(rgb):
    return cv.cvtColor(rgb, cv.COLOR_BGR2RGB)


def get_contours(img):
    return cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)




