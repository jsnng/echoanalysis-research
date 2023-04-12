import os
import glob
import torchvision

import cv2
import numpy as np


class dataset(torchvision.datasets.VisionDataset):
    def __init__(
            self,
            root,
            transforms = None,
            transform = None,
            target_transform = None,
    ):

        super().__init__(root, transforms, transform, target_transform)

        self.root = root
        self.transform = transform

        self.items = glob.glob(os.path.join(self.root, '*.npy'))

    def __getitem__(self, idx):
        
        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.items)
