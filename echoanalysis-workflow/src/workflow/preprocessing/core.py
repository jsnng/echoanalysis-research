"""invokes pre-processing of echocardiograms."""

import os

import cv2
import numpy as np

from .utils import generate
from .transforms import (
    compose,
    segmentation,
    square,
    scrub,
    averaging,
    resize,
    stack,
    wtfilter
)


def run(item: os.PathLike, verbose=False) -> None:
    """Performs per-preprocessing.
    
    the preprocessed video will be saved to a uncompressed .npz archive 
    of the same name into the current working directory. The archive will store:
        abspath (os.PathLike): original video's path.
        fps (float): video's number of frames per second.
        before (numpy.ndarray): original video in numpy.ndarray of shape [fc, h, w, c]
        after (numpy.ndarray): video after pre-preprocessing
        mask (numpy.ndarray): binary mask generated by workflow.prepreprocessing.utils.generate()
        
    Arguments:
        item (os.PathLike): path to the video to be pre-processed.
        verbose (bool): enables verbose mode.

    """
    vc = cv2.VideoCapture(item)

    if not vc.isOpened():
        raise IOError(f'{vc.__class__=} object initialisation failed on: {item=}')

    fc = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vc.get(cv2.CAP_PROP_FPS) 

    # i.e.,
    # while vc.isOpened():
    #   ret, frame = vc.read()
    #   if ret:
    #       data.append(frame)

    data = np.asarray([rgb for x, rgb in [vc.read() for _ in range(int(fc))] if x])

    assert len(data) == fc

    vc.release()

    mask = generate(data)

    # use the boundaries of mask to "squeeze" data into a square before resizing
    # to avoid distortion
    y, x = np.sort(np.where(mask))
    x1, x2 = x[[0, -1]]
    y1, y2 = y[[0, -1]]
 
    def transformer(x):
        return compose([
            segmentation(mask),
            square(y1, x1, y2, x2),
            scrub(),
            averaging(),
            wtfilter(),
            resize(112, 112),
            stack()
        ], verbose)(x)

    after = transformer(data)

    assert len(after) == fc

    # resize mask to (112, 112)
    mask = mask.astype(np.uint8)
    
    def transformer(x):
        return compose([
            square(y1, x1, y2, x2),
            resize(112, 112)
        ], verbose)(x)

    # workaround to use preprocessing transformations
    mask = transformer(np.expand_dims(mask, axis=0))
    mask = np.asarray(mask).astype(bool)

    np.savez(
        os.path.join(os.getcwd(), os.path.basename(item).split('.')[0] + '.npz'),
        basename=os.path.basename(item),  # fixme: numpy.ndarray of dtype `U33`?! use np.fromstring?
        fps=fps,
        before=data,
        after=after,
        mask=mask
    )
