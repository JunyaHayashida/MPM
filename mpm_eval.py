import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2

def eval_net(net, loader, device, criterion, writer, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_error = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, mpms_gt = batch['img'], batch['mpm']
            imgs = imgs.to(device=device, dtype=torch.float32)
            mpms_gt = mpms_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mpms_pred = net(imgs)

            total_error += criterion(mpms_pred, mpms_gt).item()

            pbar.update()

        # print(imgs.shape)
        writer.add_images('images/1', imgs[:, :1], global_step)
        writer.add_images('images/2', imgs[:, 1:], global_step)

        writer.add_images('mpms/true', mpms_gt, global_step)
        writer.add_images('mpms/pred', mpms_pred, global_step)

        mags_gt = mpms_gt.pow(2).sum(dim=1, keepdim=True).sqrt()
        mags_pred = mpms_pred.pow(2).sum(dim=1, keepdim=True).sqrt()

        writer.add_images('mags/true', mags_gt, global_step)
        writer.add_images('mags/pred', mags_pred, global_step)

    net.train()
    return total_error / n_val
