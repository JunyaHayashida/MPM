'''
Motion and Position Map Generator
'''

import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter


def compute_vector(cur, nxt, same_count, result_l, result):
    img_lm = zeros[:, :, 0].copy()  # likelihood image
    img_lm[nxt[1], nxt[0]] = 255
    img_lm = gaussian_filter(img_lm, sigma=sigma, mode='constant')
    img_lm = img_lm / img_lm.max()
    points = np.where(img_lm > 0)
    img = zeros.copy()
    for y, x in zip(points[0], points[1]):
        v3d = cur - [x, y]
        v3d = np.append(v3d, z_value)
        v3d = v3d / np.linalg.norm(v3d) * img_lm[y, x]
        img[y, x] = np.array([v3d[1], v3d[0], v3d[2]])

    img_i = result_l - img_lm
    result[img_i < 0] = img[img_i < 0]

    img_i = img_lm.copy()
    img_i[img_i == 0] = 100
    img_i = result_l - img_i
    same_count[img_i == 0] += 1
    result[img_i == 0] += img[img_i == 0]

    result_l = np.maximum(result_l, img_lm)
    return same_count, result_l, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tracklet', help='path of tracklet')
    parser.add_argument('target_image', help='path of one sample image of the target')
    parser.add_argument('save_path', default='./data_sample/train/mpm', help='save path for output(s)')
    parser.add_argument('--z_value', type=float, default=5, help='unit of time-axis')
    parser.add_argument('--sigma', type=int, default=6, help='sigma of gaussian filter')
    tp = lambda x: list(map(int, x.split(',')))
    parser.add_argument('--intervals', type=tp, default=[1], help='frame intervals, please split with commas')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    z_value = args.z_value
    sigma = args.sigma
    track_let = np.loadtxt(args.tracklet).astype('int')  # 5 columns [frame, id, x, y, parent_id]
    image_size = cv2.imread(args.target_image).shape
    frames = np.unique(track_let[:, 0])
    ids = np.unique(track_let[:, 1])
    itvs = args.intervals

    zeros = np.zeros((image_size[0], image_size[1], 3))
    ones = np.ones((image_size[0], image_size[1]))

    for itv in itvs:
        print(f'interval {itv}')
        save_dir = os.path.join(args.save_path, f'{itv:03}')
        os.makedirs(save_dir, exist_ok=True)
        output = []

        par_id = -1  # parent id
        for idx, i in enumerate(frames[:-itv]):
            same_count = ones.copy()
            result_lm = zeros[:, :, 0].copy()
            result = zeros.copy()
            bar = tqdm(total=len(ids), position=0)
            for j in ids:
                bar.set_description(f'interval{itv}---frame{i}---id{j}')
                bar.update(1)
                index_current = len(track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)])
                index_next = len(track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)])
                if index_next != 0:
                    par_id = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0, -1]
                if (index_current != 0) & (index_next != 0):
                    data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)][0][2:-1]
                    dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0][2:-1]

                    same_count, result_lm, result = compute_vector(data, dnxt, same_count,
                                                                   result_lm,
                                                                   result)

                elif ((index_current == 0) & (index_next != 0) & (par_id != -1)):
                    try:
                        data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == par_id)][0][2:-1]
                        dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0][2:-1]
                        same_count, result_lm, result = compute_vector(data, dnxt, same_count,
                                                                       result_lm,
                                                                       result)
                    except IndexError:
                        print('Error: no parent')
                        print(track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0])

            result = (result / same_count[:, :, None])
            save_path = os.path.join(save_dir, f'{idx:04}.npy')
            np.save(save_path, result.astype('float32'))
    print('finished')
