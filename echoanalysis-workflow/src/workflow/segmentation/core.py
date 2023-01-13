import os

import torch
import torchvision
from torch.utils.data import DataLoader

import numpy as np

import math

__all__ = ['run']


def run(bestpt: os.PathLike, dataset: torchvision.datasets.VisionDataset) -> None:
    """
    runs echonet; a deeplabv3_resnet50 model on the provided torchvision.datasets.VisionDataset.

    Arguments:
        bestpt (os.PathLike): abspath of a saved deeplabv3_resnet50 torch object
        dataset (torchvision.data.Dataset class): a dataset class inheriting torchvision.data.Dataset to evaluate echonet
    
    """
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
    
    # if torch.__version__ < 2.0.0:
    #     model.classifer[-1] = torch.nn.Conv2D(...)
    
    model.get_submodule('classifier')[-1] = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

    device = 'cpu'
    pin_memory = False
    
    if torch.cuda.is_available():
        device = 'cuda'
        pin_memory = True
        model = torch.nn.DataParallel(model)
    
    if torch.backends.mps.is_available():
        device = 'mps'
    
    ckpt = torch.load(bestpt, map_location={'cuda:0': device})
    model.to(device)

    ckpt['state_dict'] = dict(zip(model.state_dict(), list(ckpt['state_dict'].values())))
    model.load_state_dict(ckpt.get('state_dict'))

    # cfg
    # dataset = None
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        # num_workers=1,
        pin_memory=pin_memory
    )

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            item, metadata = batch
            fps = metadata.get('fps').detach().cpu().numpy().squeeze().astype(np.int32)

            item = item.squeeze()

            bc = math.ceil(len(item)/fps)
            prediction = np.empty((len(item), 112, 112))
            for i in range(bc): 
                idx1 = int(i*fps)
                idx2 = min((i+1)*fps, len(item))
                subset = item[idx1:idx2].to(device)

                prediction[idx1:idx2] = model(subset).get('out').detach().squeeze().cpu().numpy()
                
            savenpz = (metadata.get('basename')[0]).split('.')[0] + '.npy'
            savenpz = os.path.join(os.getcwd(), savenpz)
            np.save(savenpz, prediction)
