import os
import argparse

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from model import get_unet
from data_loader import get_dataset_train
from utils import make_dir, save_checkpoint, save_image
from tqdm import tqdm
import cv2
import numpy as np
import torch.backends.cudnn as cudnn

def main(config):
  dataset_train = get_dataset_train()
  model = get_unet()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
  loss_fn = nn.MSELoss()

  make_dir(config.result_dir)
  make_dir(config.sample_dir)
  make_dir(config.model_dir)
  make_dir(config.log_dir)

  print("Start training...")

  for epoch in range(config.epochs):
    SAVE_IMAGE_DIR = "{}/{}".format(config.sample_dir, epoch)
    make_dir(SAVE_IMAGE_DIR)
    train_loss = []

    for i, (image, mask) in enumerate(DataLoader(
      dataset_train,
      batch_size=config.batch_size,
      shuffle=True)):

      image = image.to(device)
      mask = mask.to(device)

      y_pred = model(image)
      loss = loss_fn(y_pred, mask)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())
      
      if i % 50 == 0:
        save_image(y_pred, "{}/{}.png".format(SAVE_IMAGE_DIR, i))
        print(train_loss[-1])
    
    print("Epoch: %d, Train: %.3f" % (epoch, np.mean(train_loss)))
    if epoch % 5 == 0:
      print("Saved model... {}.pth".format(epoch))
      save_checkpoint("{}/{}.pth".format(config.model_dir, epoch), model, optimizer)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int,
                        default=30, help='training epochs')
  parser.add_argument('--batch_size', type=int,
                        default=16, help='mini-batch size')
  parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
  parser.add_argument('--result_dir', type=str, default='./results')
  parser.add_argument('--sample_dir', type=str, default='./results/samples')
  parser.add_argument('--model_dir', type=str, default='./results/models')
  parser.add_argument('--log_dir', type=str, default='./results/logs')


  config = parser.parse_args()
  print(config)

  main(config)