import os
import numpy as np
import torch
from torch import optim
import datetime
from mpm_eval import eval_net
from utils.utils import RandomFlipper4MPM
from utils.losses import RMSE_Q_NormLoss
from utils.mpm_loader import MPMLoader
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from NN_model import MPMNet

def writeDetail(dir_checkpoint, net, batch_size, optimizer, lr, dataset):
    os.makedirs(dir_checkpoint, exist_ok=True)
    path = os.path.join(dir_checkpoint, 'detail.txt')
    with open(path, mode='a') as f:
        f.write('MODEL: {}\n'.format(net.__class__.__name__))
        f.write('batch size: {}\n'.format(batch_size))
        f.write('optimizer: {}\n'.format(optimizer.__class__.__name__))
        f.write('lr: {}\n'.format(lr))
        f.write('input_names: \n')
        f.write(os.listdir(os.path.join(dataset, 'input')))
        f.write('mpm_names: \n')
        f.write(os.listdir(os.path.join(dataset, 'mpm')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', help='number of epochs')
    parser.add_argument('batch_size', help='batch size')
    parser.add_argument('dataset', type=str, help='dataset name')
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--crop_size', type=tuple, default=(256, 256))
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    dataset_dir = args.dataset
    gpu = args.gpu
    lr = args.lr

    net = MPMNet(2, 3)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = RMSE_Q_NormLoss(0.8)
    dir_checkpoint = 'checkpoints/train_{0:%Y%m%d%H%M}/'.format(datetime.datetime.now())

    loader = MPMLoader(dataset_dir=dataset_dir,
                    transform=transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.RandomCrop(args.crop_size),
                            transforms.ToTensor()]),
                    )

    train_loader = DataLoader(MPMLoader, batch_size=batch_size, shuffle=True, num_workers=1)
    pre_val_loss = 1

    path0 = dir_checkpoint + 'train_all.txt'
    path1 = dir_checkpoint + 'train.txt'
    path2 = dir_checkpoint + 'eval.txt'

    net.train()
    for epoch in range(epochs):
        f0 = open(path0, mode='a')
        f1 = open(path1, mode='a')
        f2 = open(path2, mode='a')
        epoch_loss = 0
        print('epoch {}/{}'.format(epoch + 1, epochs))
        for i, b in enumerate(train_loader):
            imgs = []
            targets = []
            b['cell'] = b['cell'].detach().cpu().numpy()
            b['mpm'] = b['mpm'].detach().cpu().numpy()
            for cell, mpm in zip(b['cell'], b['mpm']):
                seed = np.random.randint(0, 4)
                img, target = RandomFlipper4MPM(seed, cell, mpm)
                imgs.append(img)
                targets.append(target)
            imgs = torch.from_numpy(np.array(imgs))
            targets = torch.from_numpy(np.array(targets))

            if gpu:
                imgs = imgs.to(torch.device('cuda:0'))
                targets = targets.to(torch.device('cuda:0'))

            outputs = net(imgs)
            loss = criterion(outputs, targets)
            epoch_loss += loss[0].item()
            f0.write('{}\n'.format(loss[0].item()))
            N_train = len(loader.ids['train'])
            print('\rTraining...[{0}/{1}] --- MSE : {2:.6f}'.format(i * batch_size, N_train, loss[0].item()), end='')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = epoch_loss / (i + 1)
        print('\nEpoch finished ! Loss: {}'.format(loss))
        f1.write('{}\t{}\n'.format(epoch + 1, loss))

        val_loss = eval_net(net, loader.ids['val'], criterion, gpu,
                            dir_checkpoint, epoch + 1, loader.cell, loader.mpm)
        print('\nvalidation MSE Loss: {}'.format(val_loss))
        f2.write('{}\t{}\n'.format(epoch + 1, val_loss))

        torch.save(net.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
        if pre_val_loss < val_loss:
            os.remove(dir_checkpoint + 'CP{}.pth'.format(epoch))
        pre_val_loss = val_loss
        print('Checkpoint {} saved !'.format(epoch + 1))
