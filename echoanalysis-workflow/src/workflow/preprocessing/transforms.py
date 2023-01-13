"""callable fx transformation classes to pre-proprocess echocardiograms."""

import abc
import functools

import numpy as np
import cv2

from .utils import grayscale

__all__ = [   
    'fx',
    'compose',
    'segmentation',
    'square',
    'scrub',
    'averaging',
    'resize',
    'filter',
    'stack'
]


def _fx_transform_X_args_fix(func):
    @functools.wraps(func)
    def wrapper(self, X):
        if not X.ndim == 4:
            X = np.expand_dims(X, -1)
        return func(self, X)
    return wrapper


class fx(abc.ABC):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def __transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        result = self.__transform(X)
        if not isinstance(result, np.ndarray):
            result = np.asarray(result)
        if np.issubdtype(result.dtype, np.floating):
            result = (result - np.min(result))/(np.max(result) - np.min(result))
            result *= 255
        return result.astype(np.uint8)


class compose(fx):
    """
    runs callable fx transformation classes sequentially on X

    Attributes:
        fx (list): list of callable workflow.preprocessing.fx classes

    Arguments:
        X (numpy.ndarray): 4d numpy.ndarray of shape [fc, h, w, c]

    Returns:
        X after callable fx transformation classes are applied. The return characteristics
        will be the last fx transformation (i.e., fx[-1])
    """
    def __init__(self, fx: list[callable], verbose=False) -> None:
        self.fx = fx
        self.verbose = verbose
        
    def __transform(self, X: np.ndarray) -> np.ndarray:
        for func in self.fx:
            X = func(X)
            if self.verbose:
                print(f"{func=}\n\t{X.shape=}\n\t{X.dtype.name=}\n\t{np.min(X)=}\n\t{np.max(X)=}")
        return X


class segmentation(fx):
    """
    preforms element-wise binary segmentation of X with mask

    Attributes:
        mask (numpy.ndarray): bool numpy.ndarray of shape [h, w]

    Arguments:
        X (numpy.ndarray): a 4d numpy.ndarray of shape [fc, h, w, c]

    Returns:
        X with all False corresponding points as 0

    e.g.,
    >>> X = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]][[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    >>> mask = [[1, 1, 1], [0, 1, 0], [1, 1, 1]]
    >>> workflow.preprocessing.transforms.segmentation(mask)(X)
    [[[1, 2, 3], [0, 5, 0], [7, 8, 9]][[1, 2, 3], [0, 5, 0], [7, 8, 9]]]
    """
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
    
    @_fx_transform_X_args_fix
    def __transform(self, X: np.ndarray) -> np.ndarray:
        fc, y, x, c = X.shape
        return X * np.broadcast_to(self.mask, (fc, c, y, x)).transpose(0, 2, 3, 1)


class square(fx):
    """
    reduction of the larger x or y axis to the same size as the smaller axis
    i.e., changes the [h, w] shape from a rectangle to a square

    Attributes:
        y1 (int): y coordinate of the point closest to the origin
        x1 (int): x coordinate of the point closest to the origin
        y2 (int): y coordinate of the point furthest from the origin
        x2 (int): x coordinate of the point furthest from the origin

    Arguments:
        X (numpy.ndarray): a 4d numpy.ndarray of shape `[fc, h, w, c]`

    Returns:
        X with height or width reduced to the smaller of (x2 - x1) or (y2 - y1)
        starting at either (y1, x?) or (y?, x1)
    """
    def __init__(self, y1: int, x1: int, y2: int, x2: int) -> None:
        self.y1 = y1
        self.x1 = x1
        self.y2 = y2
        self.x2 = x2

    def __transform(self, X: np.ndarray) -> np.ndarray:
        x_dist = self.x2 - self.x1
        y_dist = self.y2 - self.y1

        offset = np.abs(x_dist - y_dist)//2

        padding = x_dist % 2 ^ y_dist % 2
        
        if y_dist < x_dist:
            self.x1 += offset
            self.x2 -= offset
            self.y2 += padding
        
        if y_dist > x_dist:
            self.y1 += offset
            self.y2 -= offset
            self.x2 += padding
        
        return X[:, self.y1:self.y2, self.x1:self.x2]

     
class scrub(fx):    
    """
    zeros out brown-like rgb pixels. 

    Arguments:
        X (numpy.ndarray): a 4d numpy.ndarray of shape [fc, h, w, c]

    Returns:
        X with strictly increasing (i.e., r > g > b) channels replaced with 0
    """
    @_fx_transform_X_args_fix
    def __transform(self, X: np.ndarray) -> np.ndarray:
        fc, y, x, c = X.shape
        brown = ((X[..., 0] > X[..., 1]) & (X[..., 1] > X[..., 2])) == 0
        return X * np.broadcast_to(brown, (c, fc, y, x)).transpose(1, 2, 3, 0)
    

class averaging(fx):
    """
    (replace with description)

    Arguments:
        X (numpy.ndarray): a 4d numpy.ndarray of shape [fc, h, w, c]

    Returns:
       (replace with description)
    """
    def __transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([np.sum(X[i:i+3], axis=0)/3 for i in range(len(X))])
    

class resize(fx):
    """
    wrapper for cv2.resize()
    resizes X from (fc, h, w, c) to (fc, x, y, c)

    Attributes:
        y (int): new size of y-axis 
        x (int): new size of x-axis
        interpolation (cv2 attribute): cv2 interpolation method

    Arguments:
        X (numpy.ndarray): a 4d numpy.ndarray of shape [fc, h, w, c] 

    Returns:
        X with h now at y and w now at x, resized using cv2.resize().
        the default method of interpolation is cv2.INTER_AREA

    e.g., 
    >>> X.shape
    (633, 720, 540, 3)
    >>> X = workflow.preprocessing.transforms.resize(112, 112, interpolation=cv2.INTER_AREA)(X)
    >>> X.shape
    (633, 112, 112, 3)
    """
    import cv2

    def __init__(self, y: int, x: int, interpolation=cv2.INTER_AREA) -> None:
        self.x = x
        self.y = y
        self.interpolation = interpolation

    def __transform(self, X: np.ndarray) -> np.ndarray:
        # lets not reinvent to wheel-
        if np.issubdtype(X.dtype,  np.floating) and np.max(X) <= 1.0: 
            X *= 255
            X = X.astype(np.uint8)
        return np.asarray([cv2.resize(i, (self.x, self.y), self.interpolation) for i in X])
    

class filter(fx):
    """
    performs a first order wavelet decomposition with thresholding

    Arguments:
        X (numpy.ndarray): a 4d numpy.ndarray of shape [fc, h, w, c]

    Returns:
       X after wavelet decomposition. A normalisation is performed so 
       the range of X will be [0.0, 1.0].
    """
    def _transform(self, X: np.ndarray) -> np.ndarray:
        import pywt
        # i.e., low pass filter

        def f(x):
            return pywt.threshold(x, np.std(x)/2, mode='soft')
        # fixme: is it rgb or bgr?!
        X = grayscale(X)
        fc, h, w = X.shape[:3]
        Y = np.zeros((fc, h, w))
        for i in range(len(X)):
            # try a higher order wavelet decomposition?
            cA, (cH, cV, cD) = pywt.dwt2(X[i], wavelet='haar')
            cH = f(cH)
            cV = f(cV)
            cD = f(cD)
            Y[i] = pywt.idwt2((cA, (cH, cV, cD)), wavelet='haar')[:h, :w]
        return Y
    

class stack(fx):
    """
    stacks X three times on axis == -1.

    Arguments:
        X (numpy.ndarray): a 4d numpy.ndarray of shape [fc, h, w]

    Returns:
        X with a shape [fc, h, w, 3]
    
    e.g., 
    >>> X.shape
    (633, 720, 540)
    >>> X = (workflow.preprocessing.transforms.stack)(X)
    >>> X.shape
    (633, 720, 540, 3)
    """
    def __transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 3:
            raise ValueError(f"{X.ndim=} > 3")
        return np.asarray(3*[X]).transpose(1, 2, 3, 0)
