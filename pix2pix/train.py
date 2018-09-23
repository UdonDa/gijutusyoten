import os
import argparse

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from model import get_G, get_D
from data_loader import get_dataloader
from utils import make_dir, save_checkpoint, save_image
from solver import set_requires_grad, GANLoss
from tqdm import tqdm
import cv2
import numpy as np
import torch.backends.cudnn as cudnn


def main(config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  data_loader = get_dataloader(config)
  G = get_G()
  D = get_D()
  G.to(device)
  D.to(device)
  G = nn.DataParallel(G)
  D = nn.DataParallel(D)
  optimizer_G = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(0.5, 0.999))
  optimizer_D = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(0.5, 0.999))

  criterionGAN = GANLoss(use_lsgan=True).to(device)
  criterionL1 = nn.L1Loss()

  make_dir(config.result_dir)
  make_dir(config.sample_dir)
  make_dir(config.model_dir)
  make_dir(config.log_dir)
  total_steps = 0

  for epoch in range(config.epoch_count, config.niter + config.niter_decay + 1):
    SAVE_IMAGE_DIR = "{}/{}".format(config.sample_dir, epoch)
    make_dir(SAVE_IMAGE_DIR)

    for i, (real_A, real_B) in enumerate(DataLoader(
      data_loader,
      batch_size=config.batch_size,
      shuffle=True)):

      real_A = real_A.to(device)
      real_B = real_B.to(device)

      ### Making fake B image
      fake_B = G(real_A)

      ### Update D
      ## Set gradients
      set_requires_grad(D, True)
      ## Optimizer D
      optimizer_D.zero_grad()
      ## Backward
      # Fake
      pred_fake = D(fake_B.detach())
      loss_D_fake = criterionGAN(pred_fake, False)
      # Real
      pred_real = D(real_B.detach())
      loss_D_real = criterionGAN(pred_real, True)
      # Conbined loss
      loss_D = (loss_D_fake + loss_D_real) * 0.5
      loss_D.backward()
      ## Optimizer step
      optimizer_D.step()

      ### Update G
      ## Set gradients
      set_requires_grad(D, False)
      ## Optimizer G
      optimizer_G.zero_grad()
      ## Backward
      pred_fake = D(fake_B)
      loss_G_GAN = criterionGAN(pred_fake, True)
      loss_G_L1 = criterionL1(fake_B, real_B) * config.lambda_L1
      loss_G = loss_G_GAN + loss_G_L1
      loss_G.backward()
      ## Optimizer step
      optimizer_G.step()

      if total_steps % config.print_freq == 0:
      # if total_steps % 1 == 0:
        # Print
        print("Loss D Fake:{:.4f}, D Real:{:.4f}, D Total:{:.4f}, G GAN:{:.4f}, G L1:{:.4f}. G Total:{:.4f}"
          .format(loss_D_fake, loss_D_real, loss_D, loss_G_GAN, loss_G_L1, loss_G))
        # Save image
        save_image(fake_B, "{}/{}.png".format(SAVE_IMAGE_DIR, i))
      total_steps += 1

    
    if epoch % config.save_epoch_freq == 0:
      # Save model
      print("Save models in {} epochs".format(epoch))
      save_checkpoint("{}/D_{}.pth".format(config.model_dir, epoch), D, optimizer_D)
      save_checkpoint("{}/G_{}.pth".format(config.model_dir, epoch), G, optimizer_G)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
  parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
  parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
  parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
  parser.add_argument('--save_epoch_freq', type=int, default=25, help='frequency of saving checkpoints at the end of epochs')
  parser.add_argument('--continue_train', default=True, action='store_true', help='continue training: load the latest model')
  parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
  parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
  parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
  parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
  parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
  parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
  parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
  parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
  parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
  parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
  parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
  parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
  parser.add_argument('--batch_size', type=int, default=8, help='multiply by a gamma every lr_decay_iters iterations')
  parser.add_argument('--lambda_L1', type=int, default=100.0, help='multiply by a gamma every lr_decay_iters iterations')

  parser.add_argument('--result_dir', type=str, default='./results')
  parser.add_argument('--sample_dir', type=str, default='./results/samples')
  parser.add_argument('--model_dir', type=str, default='./results/models')
  parser.add_argument('--log_dir', type=str, default='./results/logs')
  parser.add_argument('--A_path', type=str, default='')



  config = parser.parse_args()
  print(config)

  main(config)