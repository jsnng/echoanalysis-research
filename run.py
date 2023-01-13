import os
import glob

import numpy as np

import workflow.preprocessing as preprocessing
import workflow.segmentation as segmentation

import torchvision


class dataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.transform = transform

        self.items = glob.glob(os.path.join(self.root, '*.npz'))

        assert len(self.items) > 0

    def __getitem__(self, idx):
        item = self.items[idx]

        data = np.load(item)

        after = data['after'].astype(np.float32)
        after /= 255.0

        metadata = {
            'basename': os.path.basename(item),  # duh-
            'fps': data['fps']
        }

        if self.transforms is not None:
            after = self.transforms(after)

        return after, metadata

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    for item in glob.iglob(os.path.join(os.getcwd(), 'Videos', '*.avi')):
        preprocessing.run(item)

    mean, std = segmentation.utils.get_ds_mean_and_std(os.getcwd())

    ds = dataset(os.getcwd(), transforms=torchvision.transforms.Compose([
        segmentation.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ]))

    bestpt = os.path.join(os.getcwd(), 'best.pt')

    if not os.path.exists(bestpt):
        raise FileNotFoundError(f'invalid location: {bestpt=}')

    segmentation.run(bestpt, ds)
