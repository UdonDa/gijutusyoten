path = './'
from glob import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms as T
from torch.autograd import Variable
from PIL import Image


class Dataset(torch.utils.data.Dataset):
  def __init__(self, is_test=False):
    self.images = glob("../datasets/non_renge_short_paste_mask/*")
    self.masks = glob("../datasets/non_renge_short/*")
    self.is_test = is_test

    self.transform = T.Compose([
      T.ToTensor(),
      T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, index):
    if index not in range(len(self.images)):
      return self.__getitem__(np.ramdom.randint(0, self.__len__()))

    if self.is_test:
      image_path = self.images[index]
      image = Image.open(image_path).convert('RGB')
      image = self.transform(image)
      return image, image_path
    else:
      # image = self.transform(Image.open(self.images[index]).convert('RGB'))
      # mask = self.transform(Image.open(self.masks[index]).convert('RGB'))

      image = self.transform(load_image(self.images[index]))
      mask = self.transform(load_image(self.masks[index]))
      return image, mask

def load_image(path):
  img = cv2.imread(str(path))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (256, 256))
  return img


def get_dataset_train():
  dataset_train = Dataset()
  print("Loaded dataset...")
  return dataset_train