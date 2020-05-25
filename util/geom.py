import numpy as np
from scipy.spatial.distance import cdist
import numpy as np


def get_middle_point(points):

    return points


def closest_point(ref, points):
    ref = np.reshape(ref, (ref.shape[0], 1)).T
    D = cdist(ref, points)
    closest_arg = np.argmin(D, axis=1)
    return closest_arg[0]


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

