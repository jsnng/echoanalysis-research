import abc
import numpy as np

__all__ = [
    'fx',
    'compose',
    'segmentation',
    'square',
    'scrub'
]

class fx(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError
        
class compose():
    def __init__(self, fx):
        self.fx = fx

    def __call__(self, item):
        for t in self.fx:
            item = t(item)
        return item

class segmentation(fx):
    def __init__(self, mask):
        self.mask = mask
    
    def __call__(self, item):
        fc, y, x, c = item.shape
        return item * np.broadcast_to(self.mask, (fc, c, y, x)).transpose(0, 2, 3, 1)

class square(fx):
    def __init__(self, y1, x1, y2, x2):
        self.y1 = y1
        self.x1 = x1
        self.y2 = y2
        self.x2 = x2

    def __call__(self, item):
        x_dist = self.x2 - self.x1
        y_dist = self.y2 - self.y1

        dxy = np.abs(x_dist - y_dist)//2

        if y_dist < x_dist:
            self.x1 += dxy
            self.x2 -= dxy
            self.y2 += dxy%2
        else:
            self.y1 += dxy
            self.y2 -= dxy
            self.x2 += dxy%2

        return item[:, self.y1:self.y2, self.x1:self.x2]
    
class scrub(fx):
    def __init__(self):
        pass
    
    def __call__(self, item):
        fc, y, x, c = item.shape
        rgb = ((item[..., 0] > item[..., 1]) & (item[..., 1] > item[..., 2])) == 0
        return item * np.broadcast_to(rgb, (c, fc, y, x)).transpose(1, 2, 3, 0)