import os
import glob
import torchvision

import cv2


class toy(torchvision.datasets.VisionDataset):
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

        self.items = glob.glob(os.path.join(self.root, ''))

    def __getitem__(self, idx):
        vc = cv2.VideoCapture(self.items[idx])
        
        if not vc.isOpened():
            return None

        fc = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        item = [vc.read()[1] for _ in range(fc)]
        vc.release()
        
        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.items)
