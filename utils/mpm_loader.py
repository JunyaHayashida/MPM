from os.path import splitext, join, basename
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import cv2
import random
from scipy.ndimage import rotate


class MPM_Dataset(Dataset):
    def __init__(self, imgs_dir, mpms_dir, itvs=[1]):
        self.imgs_dir = imgs_dir
        self.mpms_dir = mpms_dir
        self.itvs = itvs
        self.edge = 20
        self.height, self.width = 256, 256

        seqs = sorted(glob(join(imgs_dir, '*')))
        self.img_paths = []
        self.ids = []
        min_itv = itvs[0]
        max_itv = itvs[-1]
        for i, seq in enumerate(seqs):
            seq_name = basename(seq)
            self.img_paths.extend([[path, seq_name] for path in sorted(glob(join(seq, '*')))])
            num_seq = len(listdir(seq))
            self.ids.extend([[i, num_seq - j] for j in range(num_seq)[:-min_itv]])

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def flip_and_rotate(cls, img, mpm, seed):
        img = rotate(img, 90 * (seed % 4))
        mpm = rotate(mpm, 90 * (seed % 4))
        # process for MPM
        ## seed = 1 or 5: 90 degrees counterclockwise
        if seed % 4 == 1:
            mpm[:, :, 1] = -mpm[:, :, 1]
            mpm = mpm[:, :, [1, 0, 2]]
        ## seed = 2 or 6: 180 degrees counterclockwise
        if seed % 4 == 2:
            mpm[:, :, :2] = -mpm[:, :, :2]
        ## seed = 3 or 7: 270 degrees counterclockwise
        if seed % 4 == 3:
            mpm[:, :, 0] = -mpm[:, :, 0]
            mpm = mpm[:, :, [1, 0, 2]]
        ## flip horizontal (4 or more)
        if seed > 3:
            img = np.fliplr(img).copy()
            mpm = np.fliplr(mpm).copy()
            mpm[:, :, 1] = -mpm[:, :, 1]
        return img, mpm

    def __getitem__(self, i):
        idx = self.img_paths[i + self.ids[i][0]]
        itv = self.itvs[random.randrange(len(self.itvs))]
        while self.ids[i][1] - itv <= 0:
            itv = self.itvs[random.randrange(len(self.itvs))]
        nxt_idx = self.img_paths[i + self.ids[i][0] + itv]
        img1_file = glob(idx[0])
        img2_file = glob(nxt_idx[0])
        img_name = splitext(basename(idx[0]))[0]
        mpm_name = join(self.mpms_dir, idx[1], f'{itv:03}', img_name)
        mpm_file = glob(mpm_name + '.*')

        assert len(mpm_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mpm_file}'
        assert len(img1_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img1_file}'
        assert len(img2_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img2_file}'

        mpm = np.load(mpm_file[0])
        img1 = cv2.imread(img1_file[0], -1)
        img2 = cv2.imread(img2_file[0], -1)
        if len(img1.shape) == 2:
            img1 = np.expand_dims(img1, axis=2)
            img2 = np.expand_dims(img2, axis=2)
        img = np.concatenate([img1, img2], axis=-1)
        if img.max() > 1:
            img = img / 255

        # assert img.size[:2] == mpm.size[:2], \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mpm.size}'

        # random crop
        seed1 = np.random.randint(self.edge, img1.shape[0] - self.height - self.edge)
        seed2 = np.random.randint(self.edge, img1.shape[1] - self.width - self.edge)
        img = img[seed1:seed1 + self.height, seed2:seed2 + self.width]
        mpm = mpm[seed1:seed1 + self.height, seed2:seed2 + self.width]

        # random flip and rotate
        seed = random.randrange(8)
        img_r, mpm_r = self.flip_and_rotate(img, mpm, seed)

        img_trans = img_r.transpose((2, 0, 1))
        mpm_trans = mpm_r.transpose((2, 0, 1))

        return {
            'img': torch.from_numpy(img_trans).type(torch.FloatTensor),
            'mpm': torch.from_numpy(mpm_trans).type(torch.FloatTensor)
        }
