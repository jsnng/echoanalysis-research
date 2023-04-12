import os
import glob

import cv2
import numpy as np

import workflow.utils.preprocessing as preprocessing
import workflow.utils.segmentation as segmentation

import matplotlib.pyplot as plt

import hashlib

if __name__ == '__main__':

    for item in glob.iglob(os.path.join(os.getcwd(), '*.avi')):
        vc = cv2.VideoCapture(item)

        if not vc.isOpened():
            os.exit(-1)

        fc = vc.get(cv2.CAP_PROP_FRAME_COUNT)

        item = np.array([y for x, y in [vc.read() for _ in range(int(fc))] if x])

        assert(len(item) == int(fc))
        vc.release()

        mask = preprocessing.helpers.generate(item)

        x, y = np.sort(np.where(mask))
        x1, x2 = x[[0, -1]]
        y1, y2 = y[[0, -1]]

        item = preprocessing.core.compose(
        preprocessing.core.segmentation(mask),
        preprocessing.core.square(y1, x1, y2, x2),
        preprocessing.core.scrub(),
        )