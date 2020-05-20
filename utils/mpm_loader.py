import numpy as np
from torch.utils.data import Dataset
import os
import random
import sys
import glob


class MPMLoader(Dataset):
    """
    class for dataset of Mitosis Detection CNN

    Attributes
    ----------
    eval_rate: float
        rate of data for evaluation
    mode: str
        train or eval
    transform: function
        functions of data augmentation
        use transforms.Compose (input shape of first function is (H,W,3))
    p_names & n_names: list
        name list of positive or negative dataset
    dataset: dict
        check dataset['train'] & dataset['eval']
    train_mode: function
        change from eval to train
    eval_mode: function
        change from train to eval
    """

    def __init__(self, dataset_dir, itv, input_name='input', mpm_name='mpm', transform=None, eval_rate=0.025):
        self.transform = transform
        self.eval_rate = eval_rate
        self.in_names = os.listdir(os.path.join(dataset_dir, input_name))
        self.mpm_names = os.listdir(os.path.join(dataset_dir, mpm_name))
        self.cell, self.mpm, self.ids = self._TrainEvalSplitter_MultiFPS(dataset_dir, itv)
        self.shape = self.cell.shape

    def _TrainEvalSplitter_MultiFPS(self, cell_name, mpm_names):
        bn_in = 0
        bn_mpm = 0
        inputs = []
        mpms = []
        ids = []
        for name1, name2s in zip(cell_name, mpm_names):
            print('load: ', os.path.split(name1)[1])
            input = np.load(name1).astype('float32')
            input = input / 255.
            inputs.append(input)
            for name2 in zip(name2s):
                print('load: ', os.path.basename(name2))
                mpm = np.load(name2).astype('float32').transpose(0, 3, 1, 2)
                itv = int(os.path.basename(name2).split('_')[-1])
                if (len(input) != len(mpm) + itv):
                    print('ERROR: Inconsistency between input and mpm shape')
                    print(os.path.split(name1), input.shape)
                    print(os.path.split(name2), mpm.shape)
                    sys.exit()

                id = [[i + bn_in, i + itv + bn_in, i + bn_mpm] for i in range(len(input) - itv)]
                bn_mpm = bn_mpm + len(mpm)
                ids.extend(id)
                mpms.append(mpm)
            bn_in = bn_in + len(input)

        input = np.concatenate(inputs, axis=0)[:, None, :, :]
        mpm = np.concatenate(mpms, axis=0)

        random.seed(0)
        random.shuffle(ids)
        n = int(len(ids) * self.eval_rate)

        return input, mpm, {'val': ids[:n], 'train': ids[n:]}

    def __len__(self):
        return (len(self.ids['train']))

    def __getitem__(self, idx):
        seed1 = np.random.randint(30, self.shape[-2] - 256)
        seed2 = np.random.randint(30, self.shape[-1] - 256)
        cell1 = self.cell[self.ids['train'][idx][0]][:, seed1:seed1 + 256, seed2:seed2 + 256]
        cell2 = self.cell[self.ids['train'][idx][1]][:, seed1:seed1 + 256, seed2:seed2 + 256]
        mpm = self.mpm[self.ids['train'][idx][2]][:, seed1:seed1 + 256, seed2:seed2 + 256]
        cells = np.concatenate([cell1, cell2], axis=0)
        # if self.transform:
        #     cell1 = self.transform(cell1)
        #     cell2 = self.transform(cell2)
        #     cells = np.concatenate([cell1, cell2], axis=0)
        #     cmf = self.transform(cmf)

        return {'cell': cells, 'mpm': mpm}
