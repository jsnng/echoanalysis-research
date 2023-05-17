import functools

import numpy as np


def uint8(func=None, astype=None):
    if func is None:
        return functools.partial(uint8, astype=astype)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        result = np.asarray(result)
        return result.astype(np.uint8) if astype is None else result.astype(astype=astype)
    return wrapper
