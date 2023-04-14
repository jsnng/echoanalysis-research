from skimage.morphology import remove_small_objects, erosion
import numpy as np

import sys

from .helpers import convert, grayscale, normalisation, bresenham, midpoint
from ..helpers import int32

__all__ = ['generate']

@int32(astype=bool)
@convert
def generate(rgb):
    sy, sx = rgb.shape[1:3]
    gs = grayscale(rgb)
    
    lower = gs[:, gs.shape[2]//2:]
    lower = remove_small_objects(np.array([lower[x + 1] - lower[x - 1] for x in range(len(lower) - 1)]) > 0.05, 10000)
    lower = normalisation(np.sum(lower, axis=0)) > 0.1
    lower = erosion(lower)

    upper = rgb[..., 0] & rgb[..., 1] & rgb[..., 2]
    upper = remove_small_objects(normalisation(np.sum(upper[:, :upper.shape[2]//2], axis=0)) > 0, 10000)

    largest = np.vstack((upper, lower))
    bound_pts_array = np.empty((1000, 4), dtype=np.int32)
    k = 0
    pt_prev = [sys.maxsize, -1]

    for y in range(len(largest)):
        try:
            pt_curr = np.where(largest[y])[0][[0, -1]]
        except IndexError:
            continue

        if pt_curr[0] < pt_prev[0] and pt_curr[1] > pt_prev[1]:
            bound_pts_array[k] = y, np.sum(pt_curr)//2, *pt_curr
            k += 1
            pt_prev = pt_curr

    bound_pts_array = bound_pts_array[:k]

    x1 = np.median(bound_pts_array[:,1])

    y1 = bound_pts_array[0, 0]
    y2, x2, x3, y3, x4, x5 = bound_pts_array[np.where(bound_pts_array[:, 1] == x1)][[0, -1]][:, [0, 2, 3]].flatten()

    dx = np.abs(x2 - x4)
    gx = 1 if x2 < x4 else -1
    dy = np.abs(y2 - y3) * -1
    gy = 1 if y2 < y3 else -1

    left_bounds_pt_array = bresenham(dx, dy, gx, gy, x2, y2, sx, sy)

    dx = np.abs(x3 - x5)
    gx = 1 if x3 < x5 else -1

    right_bounds_pt_array = bresenham(dx, dy, gx, gy, x3, y2, sx, sy)

    y4 = np.average((
        *left_bounds_pt_array[np.where(left_bounds_pt_array[:, 1] == x1)][:, 0], 
        *right_bounds_pt_array[np.where(right_bounds_pt_array[:, 1] == x1)][:, 0])
    ).astype(np.int32)
    
    y5 = np.max(np.where(largest > 0)[0])

    circle_pts_array = midpoint(x1, y4, y5 - y4)
    circle_pts_array = circle_pts_array[circle_pts_array[:, 1].argsort()].astype(np.int32)

    for y, x in circle_pts_array:
        if x >= left_bounds_pt_array[y, 1]:
            left_bounds_pt_array = left_bounds_pt_array[y1:y]
            break

    for y, x in circle_pts_array[::-1]:
        if x <= right_bounds_pt_array[y, 1]:
            right_bounds_pt_array = right_bounds_pt_array[y1:y]
            break

    circle_pts_array = circle_pts_array[
        np.where((left_bounds_pt_array[-1, 1] < circle_pts_array[:, 1]) 
                 & (circle_pts_array [:, 1] < right_bounds_pt_array[-1, 1]))]

    return left_bounds_pt_array, right_bounds_pt_array, circle_pts_array
