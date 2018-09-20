import os
import argparse

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from model import get_unet
from data_loader import get_dataset_train_and_val
from utils import make_dir, save_checkpoint, save_image
from tqdm import tqdm
import cv2
import numpy as np
import torch.backends.cudnn as cudnn


def test(config):
  pass


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int,
                        default=30, help='training epochs')
  parser.add_argument('--batch_size', type=int,
                        default=1, help='mini-batch size')
  parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

  parser.add_argument('--result_dir', type=str, default='./results')
  parser.add_argument('--test_dir', type=str, default='./results/test')
  parser.add_argument('--model', type=str, default='./results/models/30.pth')
  


  config = parser.parse_args()
  print(config)
  test(config)