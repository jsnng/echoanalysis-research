import functools

import numpy as np

from ..utils import uint8

# functions in helper.py are specifically designed for
# workflow.preprocessing.utils.generate() and won't be exposed.
__all__ = []


def _extend(func):
    @functools.wraps(func)
    def wrapper(dx, dy, sx, sy, xc, yc, w, h):
        edge_pts_array = np.zeros((h + 1, 2))
        to_pts_array = func(dx, dy, sx, sy, xc, yc, w, h)[..., 1]
        from_pts_array = func(dx, dy, -1*sx, -1*sy, xc, yc, w, h)[..., 1]
        for y, (x1, x2) in enumerate(zip(to_pts_array, from_pts_array)):
            edge_pts_array[y] = y, np.max((x1, x2))
        return edge_pts_array
    return wrapper


def _convert(func):
    @functools.wraps(func)
    def wrapper(bgr):
        sy, sx = bgr.shape[1:3]
        left_pts_array, right_pts_array, circle_pts_array = func(bgr)

        mask = np.zeros((sy, sx), dtype=bool)
        y = -1
        x1 = -1
        x2 = -1

        assert left_pts_array[0, 0] == right_pts_array[0, 0]
        y1 = left_pts_array[0, 0]
        y2 = len(left_pts_array)
        y3 = len(right_pts_array)
        for y in range(max(y2, y3)):

            if y < y2:
                x1 = left_pts_array[y, 1]
            
            if y < y3:
                x2 = right_pts_array[y, 1]

            if x1 <= 0 or x2 <= 0:
                continue

            mask[y + y1, x1:x2] = 1

        # approximation of area between maximum y-coordinate of both left_pts_array and right_pts_array
        # and minimum y-coordinate of circle_pts_array
        if y + y1 < circle_pts_array[0, 0]:
            mask[y + y1:circle_pts_array[0, 0], x1:x2] = 1

        for (y, x3), x4 in zip(circle_pts_array, circle_pts_array[::-1, 1]):
            if x3 >= x4:
                break
            mask[y, x3:x4] = 1

        return mask
    return wrapper


@uint8(astype=np.int32)
@_extend
def _bresenham(dx, dy, sx, sy, x_curr, y_curr, width, height):
    pts = np.zeros((height + 1, 2))
    e = sx + sy

    while 0 < x_curr < width and 0 < y_curr < height:
        _, x_prev = pts[y_curr]
        pts[y_curr] = y_curr, np.max((x_curr, x_prev))

        e2 = 2 * e

        if e2 >= dy:
            e += dy
            x_curr += sx
        
        if e2 <= dx:
            e += dx
            y_curr += sy
            
    return pts


@uint8(astype=np.int32)
def _midpoint(x1, y1, r):
    pts = np.zeros((4000, 2))
    x = r
    y = 0
    d = 0
    k = 0

    while x >= y:
        pts[k:k+4, :] = np.array([
            y1 + y, x1 + x,
            y1 + x, x1 + y,
            y1 + x, x1 - y,
            y1 + y, x1 - x,
        ]).reshape((4, 2))

        if d <= 0:
            y += 1
            d += 2*(y+1)

        if d > 0:
            x -= 1
            d -= 2*(x+1)
        k += 4

    return pts[:k]
