
import abc
import functools
import warnings
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
    'wtfilter',
    'stack'
]


def _fx_expanddims(func):
    @functools.wraps(func)
    def wrapper(self, X):
        if X.ndim == 3:
            X = np.expand_dims(X, -1)
        return func(self, X)
    return wrapper


class fx(abc.ABC):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        _assert_ndarray_is_video(X)
        result = self.transform(X)
        if not isinstance(result, np.ndarray):
            result = np.asarray(result)
        if np.issubdtype(result.dtype, np.floating):
            result = (result - np.min(result))/(np.max(result) - np.min(result))
            result *= 255
        return result.astype(np.uint8)


class compose(fx):
    def __init__(self, fx: list[callable], verbose=False) -> None:
        self.fx = fx
        self.verbose = verbose
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        for func in self.fx:
            X = func(X)
            if self.verbose:
                print(f"{func=}\n\t{X.shape=}\n\t{X.dtype.name=}\n\t{np.min(X)=}\n\t{np.max(X)=}")
        return X


class segmentation(fx):
    def __init__(self, mask: np.ndarray) -> None:
        if not mask.ndims == 2:
            raise ValueError("Argument 'mask' is expected to be a image")
        self.mask = mask
    
    @_fx_expanddims
    def transform(self, X: np.ndarray) -> np.ndarray:
        fc, y, x, c = X.shape
        y1, x1 = self.mask.shape
        if not x == x1 or y == y1:
            warnings.warn(f'dimension mismatched: `X` y, x: {y} {x}, `mask` y, x: {y1}, {x1}')

        return X * np.broadcast_to(self.mask, (fc, c, y, x)).transpose(0, 2, 3, 1)


class square(fx):
    def __init__(self, y1: int, x1: int, y2: int, x2: int) -> None:
        self.y1 = y1
        self.x1 = x1
        self.y2 = y2
        self.x2 = x2

    def transform(self, X: np.ndarray) -> np.ndarray:
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
    @_fx_expanddims
    def transform(self, X: np.ndarray) -> np.ndarray:
        fc, y, x, c = X.shape
        brown = ((X[..., 0] > X[..., 1]) & (X[..., 1] > X[..., 2])) == 0
        return X * np.broadcast_to(brown, (c, fc, y, x)).transpose(1, 2, 3, 0)
    

class averaging(fx):
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([np.sum(X[i:i+3], axis=0)/3 for i in range(len(X))])
    

class resize(fx):
    import cv2

    def __init__(self, y: int, x: int, interpolation=cv2.INTER_AREA) -> None:
        self.x = x
        self.y = y
        self.interpolation = interpolation

    def transform(self, X: np.ndarray) -> np.ndarray:
        #FIXME: cv2.resize fails when X (ndarray) does not have a dtype of uint8
        # lets not reinvent to wheel-
        if np.issubdtype(X.dtype,  np.floating) and np.max(X) <= 1.0: 
            X *= 255
            X = X.astype(np.uint8)
        return np.asarray([cv2.resize(i, (self.x, self.y), self.interpolation) for i in X])
    

class wtfilter(fx):
    def transform(self, X: np.ndarray) -> np.ndarray:
        import pywt
        # i.e., low pass filter

        def f(x):
            return pywt.threshold(x, np.std(x)/2, mode='soft')
        # FIXME: is it rgb or bgr?!
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
    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 3:
            raise ValueError(f"ndarray ndim should be 3. Got {X.ndim}")
        return np.asarray(3*[X]).transpose(1, 2, 3, 0)


def _is_ndarray_a_video(X: np.ndarray):
    return X.ndim >= 3


def _assert_ndarray_is_video(X):
    if not _is_ndarray_a_video():
        raise ValueError("ndarray is not a video")
