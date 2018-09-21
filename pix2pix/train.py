import os
import argparse

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from model import get_G, get_D
from data_loader import get_dataset_train_and_val
from utils import make_dir, save_checkpoint, save_image
from tqdm import tqdm
import cv2
import numpy as np
import torch.backends.cudnn as cudnn


def main(config):






if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
  parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
  parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
  parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
  parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
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

  parser.add_argument('--result_dir', type=str, default='./results')
  parser.add_argument('--sample_dir', type=str, default='./results/samples')
  parser.add_argument('--model_dir', type=str, default='./results/models')
  parser.add_argument('--log_dir', type=str, default='./results/logs')
  parser.add_argument('--A_path', type=str, default='')



  config = parser.parse_args()
  print(config)

  main(config)