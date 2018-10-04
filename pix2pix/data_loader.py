from glob import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms as T
from torch.autograd import Variable


def get_dataloader(config):
  data_loader = DataLoader(config)
  return data_loader


def load_image(path):
  img = cv2.imread(str(path))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (256, 256))
  return img

class DataLoader(torch.utils.data.Dataset):
  def __init__(self, config):
    self.A_images = glob("../datasets/non_renge_short_paste_mask/*")
    self.B_images = glob("../datasets/non_renge_short/*")

    self.transform = T.Compose([
      T.ToPILImage(),
      T.ToTensor(),
      T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

  def __len__(self):
    return len(self.A_images)

  def __getitem__(self, index):
    mini = min(len(self.A_images), len(self.B_images))
    if index not in range(mini):
      return self.__getitem__(np.random.randint(0, self.__len__()))

    A = self.A_images[index]
    B = self.B_images[index]

    A = self.transform(load_image(A))
    B = self.transform(load_image(B))
    return A, B
