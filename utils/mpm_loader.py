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
from hydra.utils import to_absolute_path as abs_path

class MPM_Dataset(Dataset):
    def __init__(self, cfg_type, cfg_data):
        self.imgs_dir = abs_path(cfg_type.imgs)
        self.mpms_dir = abs_path(cfg_type.mpms)
        self.itvs = cfg_type.itvs
        self.edge = cfg_data.edge
        self.height, self.width = cfg_data.height, cfg_data.width

        seqs = sorted(glob(join(self.imgs_dir, '*')))
        self.img_paths = []
        self.ids = []
        min_itv = self.itvs[0]
        max_itv = self.itvs[-1]
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

class MPM_Dataset3D(Dataset):
    def __init__(self, cfg_type, cfg_data):
        self.imgs_dir = abs_path(cfg_type.imgs)
        self.mpms_dir = abs_path(cfg_type.mpms)
        self.itvs = cfg_type.itvs
        self.edge = cfg_data.edge
        self.height, self.width = cfg_data.height, cfg_data.width
        self.depth = cfg_data.depth

        seqs = sorted(glob(join(self.imgs_dir, '*')))
        self.img_paths = []
        self.ids = []
        min_itv = self.itvs[0]
        max_itv = self.itvs[-1]
        for i, seq in enumerate(seqs):
            seq_name = basename(seq)
            self.img_paths.extend([[path, seq_name] for path in sorted(glob(join(seq, '*')))])
            num_seq = len(listdir(seq))
            self.ids.extend([[i, num_seq - j] for j in range(num_seq)[:-min_itv*self.depth]])
        self.org_shape = cv2.imread(self.img_paths[0][0]).shape

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def flip_and_rotate(cls, mpm_list, seed):
        mpms = []
        for mpm in mpm_list:
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
                mpm = np.fliplr(mpm).copy()
                mpm[:, :, 1] = -mpm[:, :, 1]
            mpms.append(mpm)
        return np.concatenate(mpms, axis=-1)

    def __getitem__(self, i):
        idx = self.img_paths[i + self.ids[i][0]]
        itvs = [self.itvs[random.randrange(len(self.itvs))] for j in range(self.depth)]
        sum_itvs = sum(itvs)
        while self.ids[i][1] - sum_itvs <= 0:
            itvs = [self.itvs[random.randrange(len(self.itvs))] for j in range(self.depth)]
            sum_itvs = sum(itvs)
        img_names = [idx[0]]
        mpm_names = []
        nxt_id = i + self.ids[i][0]
        nxt_idx = idx[0]
        for itv in itvs:
            mpm_name = join(self.mpms_dir, idx[1], f'{itv:03}', splitext(basename(nxt_idx))[0])
            mpm_name = glob(mpm_name + '.*')[0]
            mpm_names.append(mpm_name)
            nxt_id = nxt_id + itv
            nxt_idx = self.img_paths[nxt_id][0]
            img_names.append(nxt_idx)

        print(img_names)
        print(mpm_names)

        # random crop
        seed1 = np.random.randint(self.edge, self.org_shape[0] - self.height - self.edge)
        seed2 = np.random.randint(self.edge, self.org_shape[1] - self.width - self.edge)

        mpms = [np.load(mpmname)[seed1:seed1 + self.height, seed2:seed2 + self.width] for mpmname in mpm_names]
        imgs = [cv2.imread(imgname, -1)[seed1:seed1 + self.height, seed2:seed2 + self.width] for imgname in img_names]
        if len(imgs[0].shape) == 2:
            ch = 1
            imgs = np.concatenate([np.expand_dims(img, axis=2) for img in imgs], axis=-1)
        else:
            ch = imgs.shape[-1]
            imgs = np.concatenate(imgs, axis=-1)
        if imgs.max() > 1:
            imgs = imgs / 255

        # assert img.size[:2] == mpm.size[:2], \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mpm.size}'



        # random flip and rotate
        seed = random.randrange(8)
        img_rs = rotate(imgs, 90 * (seed % 4))
        if seed > 3:
            img_rs = np.fliplr(img_rs).copy()
        mpm_rs = self.flip_and_rotate(mpms, seed)

        img_transes = img_rs.transpose((2, 0, 1))
        mpm_transes = mpm_rs.transpose((2, 0, 1))
        img_transes = img_transes.reshape((ch, self.depth+1, self.height, self.width))
        mpm_transes = mpm_transes.reshape((3, self.depth, self.height, self.width))
        print(mpm_transes.shape)

        return {
            'img': torch.from_numpy(img_transes).type(torch.FloatTensor),
            'mpm': torch.from_numpy(mpm_transes).type(torch.FloatTensor)
        }
