import functools
import numpy as np

def int32(func = None, *, astype=None):
    if func is None:
        return functools.partial(int32, astype=astype)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = np.array(func(*args, **kwargs))
        return result.astype(np.int32) if astype is None else result.astype(astype)
    return wrapper