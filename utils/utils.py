import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from scipy.ndimage.filters import gaussian_filter


def inferenceMPM(model, names, ch=None, max_v=255):
    if ch is None:
        imgs = [cv2.imread(name, -1)[None, ...] for name in names]
    else:
        imgs = [cv2.imread(name, -1).transpose(2, 0, 1) for name in names]
    img = np.concatenate(imgs, axis=0)
    img = img / max_v
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.cuda()
    output = model(img)
    mpm = output[0].cpu().detach().numpy()
    return mpm.transpose(1, 2, 0)


def getIndicatedPoints(acm, mag_max=0.9, mag_min=0.1, z_value=5, gauss=False, sigma=3):
    mag = np.sqrt(np.sum(np.square(acm), axis=-1))
    if gauss:
        mag = gaussian_filter(mag, sigma=sigma)

    mag[mag > mag_max] = mag_max
    map_left = np.zeros(mag.shape)
    map_right = np.zeros(mag.shape)
    map_top = np.zeros(mag.shape)
    map_bottom = np.zeros(mag.shape)
    map_left_top = np.zeros(mag.shape)
    map_right_top = np.zeros(mag.shape)
    map_left_bottom = np.zeros(mag.shape)
    map_right_bottom = np.zeros(mag.shape)
    map_left[1:, :] = mag[:-1, :]
    map_right[:-1, :] = mag[1:, :]
    map_top[:, 1:] = mag[:, :-1]
    map_bottom[:, :-1] = mag[:, 1:]
    map_left_top[1:, 1:] = mag[:-1, :-1]
    map_right_top[:-1, 1:] = mag[1:, :-1]
    map_left_bottom[1:, :-1] = mag[:-1, 1:]
    map_right_bottom[:-1, :-1] = mag[1:, 1:]
    peaks_binary = np.logical_and.reduce((
        mag >= map_left,
        mag >= map_right,
        mag >= map_top,
        mag >= map_bottom,
        mag >= map_left_top,
        mag >= map_left_bottom,
        mag >= map_right_top,
        mag >= map_right_bottom,
        mag > mag_min
    ))
    _, _, _, center = cv2.connectedComponentsWithStats((peaks_binary * 1).astype('uint8'))
    center = center[1:]
    result = []
    for center_cell in center.astype('int'):
        vec = acm[center_cell[1], center_cell[0]]
        mag_value = mag[center_cell[1], center_cell[0]]
        vec = vec / np.linalg.norm(vec)
        # print(vec)
        x = 0 if vec[1] == 0 else z_value * (vec[1] / vec[2])
        y = 0 if vec[0] == 0 else z_value * (vec[0] / vec[2])
        x = int(x)
        y = int(y)
        result.append([center_cell[0], center_cell[1], center_cell[0] + x, center_cell[1] + y, mag_value])
    return np.array(result)
