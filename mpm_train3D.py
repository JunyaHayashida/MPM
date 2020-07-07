import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from mpm_eval import eval_net
from MPM_Net3D import MPMNet3D
from torch.utils.tensorboard import SummaryWriter
from utils.mpm_loader import MPM_Dataset3D
from torch.utils.data import DataLoader, random_split

from utils.losses import RMSE_Q_NormLoss

import hydra


def train_net(net,
              device,
              cfg):

    if cfg.eval.imgs is not None:
        train = MPM_Dataset3D(cfg.train, cfg.dataloader)
        val = MPM_Dataset3D(cfg.eval, cfg.dataloader)
        n_train = len(train)
        n_val = len(val)
    else:
        dataset = MPM_Dataset3D(cfg.train, cfg.dataloader)
        n_val = int(len(dataset) * cfg.eval.rate)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])

    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{cfg.train.lr}_BS_{batch_size}')
    global_step = 0

    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = RMSE_Q_NormLoss(0.8)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {cfg.train.lr}
        Training size:   {len(train)}
        Validation size: {len(val)}
        Checkpoints:     {cfg.output.save}
        Device:          {device.type}
        Intervals        {cfg.train.itvs}
        Optimizer        {optimizer.__class__.__name__}
        Criterion        {criterion.__class__.__name__}
    ''')

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['img']
                mpms_gt = batch['mpm']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mpms_gt = mpms_gt.to(device=device, dtype=torch.float32)

                mpms_pred = net(imgs)
                loss = criterion(mpms_pred, mpms_gt)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_loss = eval_net(net, val_loader, device, criterion, writer, global_step)
                    # scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation loss: {}'.format(val_loss))
                    writer.add_scalar('Loss/test', val_loss, global_step)

        if cfg.output.save:
            try:
                os.mkdir(cfg.output.dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(cfg.output.dir, f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

@hydra.main(config_path='config/mpm_train3D.yaml')
def main(cfg):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = MPMNet3D()
    # logging.info(f'Network:\n'
    #              f'\t{net.n_channels} input channels\n'
    #              f'\t{net.n_classes} output channels\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if cfg.load:
        net.load_state_dict(
            torch.load(cfg.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  device=device,
                  cfg=cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
    main()
