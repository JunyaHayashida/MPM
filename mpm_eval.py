import torch
import numpy as np
import cv2
from utils import getImageTable, get3chImage
import os
from scipy.ndimage.filters import gaussian_filter


def getPeaks_getIndicatedPoints(acm):
    mag = np.sqrt(np.sum(np.square(acm), axis=-1))
    norm = gaussian_filter(mag, sigma=3)
    map_left = np.zeros(norm.shape)
    map_right = np.zeros(norm.shape)
    map_top = np.zeros(norm.shape)
    map_bottom = np.zeros(norm.shape)
    map_left[1:, :] = norm[:-1, :]
    map_right[:-1, :] = norm[1:, :]
    map_top[:, 1:] = norm[:, :-1]
    map_bottom[:, :-1] = norm[:, 1:]
    peaks_binary = np.logical_and.reduce((
        norm >= map_left,
        norm >= map_right,
        norm >= map_top,
        norm >= map_bottom,
        norm > 0.05
    ))
    _, _, _, center = cv2.connectedComponentsWithStats((peaks_binary * 1).astype('uint8'))
    center = center[1:]
    result = []
    for center_cell in center.astype('int'):
        vec = acm[center_cell[1], center_cell[0]]
        mag_value = mag[center_cell[1], center_cell[0]]
        vec = vec / np.linalg.norm(vec)
        x = 0 if vec[1] == 0 else 5 * (vec[1] / vec[2])
        y = 0 if vec[0] == 0 else 5 * (vec[0] / vec[2])
        x = int(x)
        y = int(y)
        result.append([center_cell[0] + x, center_cell[1] + y, center_cell[0], center_cell[1]])
    return np.array(result)


def plotPoints(img1, img2, points):
    for point in points:
        img1 = cv2.drawMarker(img1, (int(point[0]), int(point[1])), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                              markerSize=4)
        img1 = cv2.arrowedLine(img1, tuple(point[:2]), tuple(point[2:]), color=(255, 0, 0), thickness=2)
        img2 = cv2.drawMarker(img2, (int(point[2]), int(point[3])), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                              markerSize=4)
        img2 = cv2.arrowedLine(img2, tuple(point[:2]), tuple(point[2:]), color=(255, 0, 0), thickness=2)
    return img1, img2


def visualize_hsv(flow):
    flow = flow.astype('float32')

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[..., 1] = flow[..., 2] * 255

    mag, ang = cv2.cartToPolar(flow[..., 1], flow[..., 0])

    hsv[..., 0] = ang * 180 / np.pi / 2

    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    hsv = hsv.astype('uint8')
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb


def eval_net(net, dataset, criterion, gpu=False, dir='', cp=0, inputs=[], cmf_data=[]):
    net.eval()
    loss = 0
    save_dir = dir + 'CP{0}/'.format(cp)
    os.makedirs(save_dir, exist_ok=True)
    for i, b in enumerate(dataset):
        img = np.concatenate([inputs[b[0]], inputs[b[1]]], axis=0)[:, :512, :512]
        true_mask = cmf_data[b[2]][:, :512, :512]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        output = net(img)
        loss_pred = criterion(output, true_mask)

        loss += loss_pred[0].item()

        if 20 > i:
            img = ((img.cpu().detach().numpy()) * 255).astype('uint8')
            img1 = get3chImage(img[0, 0])
            img2 = get3chImage(img[0, 1])

            true_mask = true_mask.cpu().detach().numpy()
            true_cmf = true_mask[0]
            true_cmf = np.transpose(true_cmf, axes=[1, 2, 0])
            true_norm = (
                    np.sqrt(true_cmf[:, :, 0] ** 2 + true_cmf[:, :, 1] ** 2 + true_cmf[:, :, 2] ** 2) * 255).astype(
                'uint8')
            true_norm[true_norm > 255] = 255

            true_cmf = visualize_hsv(true_cmf).astype('uint8')

            mpm = output[0].cpu().detach().numpy()
            mpm = np.transpose(mpm, axes=[1, 2, 0])
            norm_img = (np.sqrt(mpm[:, :, 0] ** 2 + mpm[:, :, 1] ** 2 + mpm[:, :, 2] ** 2) * 255)
            norm_img[norm_img > 255] = 255
            norm_img = norm_img.astype('uint8')
            result = getPeaks_getIndicatedPoints(mpm)
            img1, img2 = plotPoints(img1, img2, result)
            mpm = visualize_hsv(mpm).astype('uint8')
            name = save_dir + '{0:03d}.png'.format(i)

            getImageTable([true_cmf, true_norm, img1, mpm, norm_img, img2], clm=3, save_name=name)

        print('\rpred_loss: {0:.6f}'.format(loss_pred[0].item()), end='')

    return loss / (i + 1)
