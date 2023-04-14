import os
import glob

import cv2
import numpy as np

import workflow.preprocessing as preprocessing
import workflow.segmentation as segmentation

import matplotlib.pyplot as plt

import warnings

import torch
import torchvision

class averaging(preprocessing.fx):
    def __init__(self):
        pass

    def __call__(self, item):
        item = np.array([np.sum(item[x:x+3], axis=0)/3 for x in range(len(item))])
        return preprocessing.helpers.normalisation(item)
    
class dataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, transforms = None, transform = None, target_transform = None):

        super().__init__(root, transforms, transform, target_transform)

        self.root = root
        self.transform = transform

        self.items = glob.glob(os.path.join(self.root, '*.npy'))

    def __getitem__(self, idx):
        np.load(self.items[idx])

        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.items)
    
if __name__ == '__main__':

    for item in glob.iglob(os.path.join(os.getcwd(), '*.avi')):
        vc = cv2.VideoCapture(item)

        if not vc.isOpened():
            os.exit(-1)

        fc = vc.get(cv2.CAP_PROP_FRAME_COUNT)

        item = np.array([y for x, y in [vc.read() for _ in range(int(fc))] if x])

        if not len(item) == int(fc):
            warnings.warn("frame count property and return ndarray length are not equal.")

        vc.release()

        mask = preprocessing.helpers.generate(item)

        y, x = np.sort(np.where(mask))
        x1, x2 = x[[0, -1]]
        y1, y2 = y[[0, -1]]

        preprocessor = lambda x: preprocessing.compose([
            preprocessing.segmentation(mask),
            preprocessing.square(y1, x1, y2, x2),
            preprocessing.scrub(),
            averaging(),
        ])(x)