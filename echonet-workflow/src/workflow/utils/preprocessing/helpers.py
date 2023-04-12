import functools
import sys

import numpy as np
from skimage.morphology import remove_small_objects, erosion

from ..helpers import int32

__all__ = ['generate']

grayscale = lambda x: np.tensordot(x/255, [.2126, .7152, .0722], axes=((-1, -1)))
normalisation = lambda x: (x - np.min(x))/ (np.max(x) - np.min(x))

def generate(): 
    pass