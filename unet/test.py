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


import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
import os

from model import get_unet
from utils import make_dir, save_checkpoint, save_image
from data_loader import get_dataset_test
from sys import exit

def test(config):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  cudnn.benchmark = True

  dataset_test = get_dataset_test()
  model = get_unet()
  model.to(device)
  model.load_state_dict(torch.load(config.model)['state_dict'])

  for i, (image, image_path) in enumerate(data.DataLoader(dataset_test, batch_size = config.batch_size, shuffle = False)):
    image = image.to(device)
    FILE_NAME = image_path[0].split("/")[-1]

    y_pred = model(image)
    save_image(y_pred, "{}/{}".format(config.test_dir, FILE_NAME))

    if i % 10 == 0:
      print("Saved... {}".format(i))

  print("Finished...!")


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
  parser.add_argument('--model', type=str, default='./results/models/10.pth')
  


  config = parser.parse_args()
  print(config)
  test(config)