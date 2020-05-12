'''
Motion and Position Map Generator
'''

import cv2
import numpy as np
import os
import argparse

def compute_vector(pre, nxt, result, result_l, result_y, result_x, result_z):
    img_l = zeros.copy()  # likelihood image
    img_l[nxt[1] + blur_range, nxt[0] + blur_range] = 255
    img_l = cv2.GaussianBlur(img_l, ksize=(int(blur_range * 2) + 1, int(blur_range * 2) + 1), sigmaX=6)
    img_l = img_l / img_l.max()
    points = np.where(img_l > 0)
    img_y = zeros.copy()
    img_x = zeros.copy()
    img_z = zeros.copy()
    for y, x in zip(points[0], points[1]):
        v3d = pre + [blur_range, blur_range] - [x, y]
        v3d = np.append(v3d, z_value)
        v3d = v3d / np.linalg.norm(v3d) * img_l[y, x]
        img_y[y, x] = v3d[1]
        img_x[y, x] = v3d[0]
        img_z[y, x] = v3d[2]

    img_i = result_l - img_l
    result_y = np.where(img_i < 0, img_y, result_y)
    result_x = np.where(img_i < 0, img_x, result_x)
    result_z = np.where(img_i < 0, img_z, result_z)

    img_i = img_l.copy()
    img_i[img_i == 0] = 2
    img_i = result_l - img_i
    result[img_i == 0] += 1
    result_y += np.where(img_i == 0, img_y, 0)
    result_x += np.where(img_i == 0, img_x, 0)
    result_z += np.where(img_i == 0, img_z, 0)

    result_l = np.maximum(result_l, img_l)
    return result, result_l, result_y, result_x, result_z

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tracklet', help='path of tracklet')
    parser.add_argument('target_image', help='path of one sample image of the target')
    parser.add_argument('save_path', help='save path for output(s)')
    parser.add_argument('--blur_range', type=int, default=50)
    parser.add_argument('--z_value', type=float, default=5)
    tp = lambda x: list(map(int, x.split(',')))
    parser.add_argument('--intervals', type=tp, default=[1], help='frame intervals, please split with commas')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    blur_range = args.blur_range
    z_value = args.z_value
    track_let = np.loadtxt(args.tracklet).astype('int') # 5 columns [frame, id, x, y, parent_id]
    image_size = cv2.imread(args.target_image).shape
    frames = np.unique(track_let[:, 0])
    ids = np.unique(track_let[:, 1])
    itvs = args.intervals

    for itv in itvs:
        print(f'interval {itv}')
        zeros = np.zeros((image_size[0] + blur_range * 2, image_size[1] + blur_range * 2))
        ones = np.ones((image_size[0] + blur_range * 2, image_size[1] + blur_range * 2))
        output = []

        par_id = -1 # parent id
        for idx, i in enumerate(frames[:-itv]):
            result = ones.copy()
            result_lm = zeros.copy()
            result_y = zeros.copy()
            result_x = zeros.copy()
            result_z = zeros.copy()

            for j in ids:
                index_check = len(track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)])
                index_chnxt = len(track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)])
                if index_chnxt != 0:
                    par_id = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0, -1]
                if (index_check != 0) & (index_chnxt != 0):
                    data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)][0][2:-1]
                    dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0][2:-1]

                    result, result_lm, result_y, result_x, result_z = compute_vector(data, dnxt, result,
                                                                                     result_lm,
                                                                                     result_y,
                                                                                     result_x,
                                                                                     result_z)

                elif ((index_check == 0) & (index_chnxt != 0) & (par_id != -1)):
                    try:
                        data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == par_id)][0][2:-1]
                        dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0][2:-1]
                        result, result_lm, result_y, result_x, result_z = compute_vector(data, dnxt, result,
                                                                                         result_lm,
                                                                                         result_y, result_x,
                                                                                         result_z)
                    except IndexError:
                        print('Error: no parent')
                        print(track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0])

            result = result[blur_range:-blur_range, blur_range:-blur_range]
            print(i + 1, 'to', i + itv + 1, result.max())

            result_y = result_y[blur_range:-blur_range, blur_range:-blur_range]
            result_x = result_x[blur_range:-blur_range, blur_range:-blur_range]
            result_z = result_z[blur_range:-blur_range, blur_range:-blur_range]
            result_lm = result_lm[blur_range:-blur_range, blur_range:-blur_range]

            result_x = (result_x / result)
            result_y = (result_y / result)
            result_z = (result_z / result)

            result_vector = np.concatenate((result_y[:, :, None], result_x[:, :, None], result_z[:, :, None]), axis=-1)
            output.append(result_vector)
        output = np.array(output).astype('float16')
        save_path_vector = os.path.join(args.save_path, f'mpm_{itv:02}_{blur_range:03}_{z_value:03}.npy')
        np.save(save_path_vector, output)
    print('finished')