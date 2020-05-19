import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def get3chImage(src):
    '''
    Args:
        src: input image
    '''
    chk = src.shape
    if len(chk) == 2:
        out = np.concatenate([src[:, :, None], src[:, :, None], src[:, :, None]], axis=-1)
        return out
    elif chk[-1] == 1:
        out = np.concatenate([src, src, src], axis=-1)
        return out
    else:
        return src


def getImageTable(srcs=[], clm=4, save_name=None):
    '''
    Args:
        srcs: image list
        clm: how many columns
        save_name: save the image table with this name if enter, output it if None
    '''
    white_c = np.full((srcs[0].shape[0], 3, 3), 255).astype('uint8')
    white_r = np.full((3, (srcs[0].shape[1] + 3) * clm - 3, 3), 255).astype('uint8')
    black = np.zeros(srcs[0].shape).astype('uint8')
    out = []

    for i in range(len(srcs)):
        srcs[i] = get3chImage(srcs[i])
        srcs[i] = cv2.hconcat([srcs[i], white_c])
    for i in range(len(srcs) % clm):
        srcs.append(black)

    for l in range(int(len(srcs) / clm)):
        c_imgs = cv2.hconcat(srcs[l * clm:l * clm + clm])
        out.append(c_imgs[:, :-3])
        out.append(white_r)
    out = cv2.vconcat(out)

    if save_name is not None:
        cv2.imwrite(save_name, out[:-3])
    else:
        return out


def RandomFlipper4MPM(seed, input, target):
    if seed == 0:
        return input, target
    elif seed == 1:
        inputH = input[:, :, ::-1].copy()
        targetH = target[:, :, ::-1].copy()
        targetH[1] = -targetH[1]
        return inputH, targetH
    elif seed == 2:
        inputV = input[:, ::-1, :].copy()
        targetV = target[:, ::-1, :].copy()
        targetV[0] = -targetV[0]
        return inputV, targetV
    else:
        inputHV = input[:, ::-1, ::-1].copy()
        targetHV = target[:, ::-1, :: -1].copy()
        targetHV[:2] = -targetHV[:2]
        return inputHV, targetHV
