import torch
import torchvision

from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm, trange

import math

def run(bestpt, dataset):
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifer[-1] = torch.nn.Conv2d(256, 1, (1, 1))
    
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
    model.load(ckpt)

    # cfg
    dataset = None
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=pin_memory
    )

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            for item, metadata in tqdm(batch):
                fps = metadata['fps'].detach.cpu().numpy().squeeze()

                bc = math.ceil(len(item)/fps)
                prediction = np.empty((len(item), 112, 112))
                for i in tqdm.trange(bc):
                    idx1 = int(i*fps)
                    idx2 = min((i+1)*fps, len(item))
                    subset = item[idx1:idx2]

                    prediction[idx1:idx2] = model(subset.to(device))['out'].detach().cpu().numpy().squeeze()

                prediction = prediction > 0.99

                echocardiogram = item.detach().cpu().numpy().squeeze().transpose(0, 2, 3, 1)
                echocardiogram = (echocardiogram - mean) * std
                echocardiogram[echocardiogram > 1.0] = 1.0
                echocardiogram[echocardiogram < 0.0] = 0.0

                segmentation = echocardiogram.copy()
                for i in range(len(prediction)):
                    segmentation[:, :, :, 2][prediction[:] > 0] = 0.5
                segmentation *= 255
                segmentation = segmentation.astype(np.uint8)
                
                echocardiogram *= 255
                echocardiogram = segmentation.astype(np.uint8)

                # np.save('', np.array([    
                #     ('echocardiogram', np.uint8, echocardiogram)
                #     ('prediction', np.bool, prediction),
                #     ('segmentation', np.uint8, segmentation)
                # ]))

                # vw = cv2.VideoWriter('', cv2.VideoWriter_fourcc(*"MJPG"), fps, (112, 112))
                # if vw.isOpened():
                #     (vw.write(x) for x in segmentation)
                # vw.release()

                return echocardiogram
