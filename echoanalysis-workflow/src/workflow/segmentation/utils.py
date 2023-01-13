import os
import glob

import numpy as np

__all__ = ['get_ds_mean_and_std']


def get_ds_mean_and_std(root):
    pixels = np.empty(3)
    for item in glob.iglob(os.path.join(root, '*.npz')):
        data = np.load(item)
        pc = data['after'][np.where(data['mask'])]
        pc = (pc/255).astype(np.float32)
        pixels = np.vstack((pixels, pc))
    return np.mean(pixels, axis=0), np.std(pixels, axis=0)
