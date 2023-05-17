import sys

import numpy as np
from skimage.morphology import (
    remove_small_objects, 
    erosion
)

from ._helpers import (
    _convert, 
    _bresenham, 
    _midpoint
)
from ..utils import uint8


# fixme: is it rgb or bgr?!
def grayscale(x):
    return np.tensordot(x/255, [.0722, .7152, .2126], axes=((-1, -1)))


def normalisation(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


@uint8(astype=bool)
@_convert
def generate(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    preliminary pre-processing of raw echocardiogram data to generate a binary mask
    corresponding to the echocardiogram image area.

    Arguments:
        bgr (numpy.ndarray): 8-bit bgr (reversed rgb) numpy ndarray of shape [fc, h, w, c]

    Returns:
        three numpy.ndarray (y, x) point arrays representing the left, right and bottom boundaries (respectively)
        of the area corresponding to the echocardiogram image area.
        
    '''
    sy, sx = bgr.shape[1:3]
    # fixme: is it rgb or bgr?!
    gs = grayscale(bgr)
    
    # fixme: thresholding is temperamental
    lower = gs[:, gs.shape[2]//2:]
    # ...range(len(lower) - 1)]) > [threshold]
    lower = remove_small_objects(np.array([lower[x + 1] - lower[x - 1] for x in range(len(lower) - 1)]) > 0.01, 10000)
    # ... axis=0)) > [threshold]
    lower = normalisation(np.sum(lower, axis=0)) > 0.1
    lower = erosion(lower)

    if __debug__:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(bgr[0, bgr.shape[2]//2:])
        plt.imshow(lower, alpha=0.3)

    upper = bgr[..., 0] & bgr[..., 1] & bgr[..., 2]
    upper = remove_small_objects(normalisation(np.sum(upper[:, :upper.shape[2]//2], axis=0)) > 0, 10000)

    largest_dynamic_object = np.vstack((upper, lower))
    bound_pts_array = np.empty((1000, 4), dtype=np.int32)
    k = 0
    pt_prev = [sys.maxsize, -1]

    # greedy approximation of the bounding polygon
    for y in range(len(largest_dynamic_object)):
        try:    
            pt_curr = np.where(largest_dynamic_object[y])[0][[0, -1]]
        except IndexError:
            continue

        if pt_curr[0] < pt_prev[0] and pt_curr[1] > pt_prev[1]:
            bound_pts_array[k] = y, np.sum(pt_curr)//2, *pt_curr
            k += 1
            pt_prev = pt_curr

    bound_pts_array = bound_pts_array[:k]

    # estimation of the centre x-coordinate of the heart's apex
    x1 = np.median(bound_pts_array[:, 1])

    # closest y point to the heart's apex
    y1 = bound_pts_array[0, 0]

    # find the "corners" of the bounding polygon
    y2, x2, x3, y3, x4, x5 = bound_pts_array[np.where(bound_pts_array[:, 1] == x1)][[0, -1]][:, [0, 2, 3]].flatten()

    dx = np.abs(x2 - x4)
    gx = 1 if x2 < x4 else -1
    dy = np.abs(y2 - y3) * -1
    gy = 1 if y2 < y3 else -1

    left_bounds_pt_array = _bresenham(dx, dy, gx, gy, x2, y2, sx, sy)

    dx = np.abs(x3 - x5)
    gx = 1 if x3 < x5 else -1

    right_bounds_pt_array = _bresenham(dx, dy, gx, gy, x3, y2, sx, sy)

    # approximation of the corresponding y-coordinate of x1
    y4 = np.average((
        *left_bounds_pt_array[np.where(left_bounds_pt_array[:, 1] == x1)][:, 0], 
        *right_bounds_pt_array[np.where(right_bounds_pt_array[:, 1] == x1)][:, 0])
    ).astype(np.int32)
    
    # furthest y point from the heart's apex
    y5 = np.max(np.where(largest_dynamic_object > 0)[0])

    circle_pts_array = _midpoint(x1, y4, y5 - y4)
    circle_pts_array = circle_pts_array[circle_pts_array[:, 1].argsort()].astype(np.int32)

    # find the intersection point of left_bounds_pt_array and circle_pts_array
    # and reduce left_bounds_pt_array to valid points
    for y, x in circle_pts_array:
        if x >= left_bounds_pt_array[y, 1]:
            left_bounds_pt_array = left_bounds_pt_array[y1:y]
            break

    # find the intersection point of right_bounds_pt_array and circle_pts_array
    # and reduce right_bounds_pt_array to valid points
    for y, x in circle_pts_array[::-1]:
        if x <= right_bounds_pt_array[y, 1]:
            right_bounds_pt_array = right_bounds_pt_array[y1:y]
            break

    # reduce circle_pts_array to valid points
    circle_pts_array = circle_pts_array[
        np.where((left_bounds_pt_array[-1, 1] < circle_pts_array[:, 1]) 
                 & (circle_pts_array[:, 1] < right_bounds_pt_array[-1, 1]))]

    return left_bounds_pt_array, right_bounds_pt_array, circle_pts_array
