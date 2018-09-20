path = './'

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms as T
from torch.autograd import Variable

def load_image(path, mask=False):
  img = cv2.imread(str(path))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (256, 256))
  return img

class Dataset(torch.utils.data.Dataset):
  def __init__(self, image_path, mask_path, is_test=False):
    self.IMAGE_PATH = image_path
    self.MASK_PATH = mask_path
    self.images = self._get_data_ary_from_txt(self.IMAGE_PATH)
    self.masks = self._get_data_ary_from_txt(self.MASK_PATH)
    self.is_test = is_test
  
  def _get_data_ary_from_txt(self, txt_path):
    FILE = open(txt_path)
    FILES = FILE.readlines()
    FILE.close()
    FILES = [f.replace("\n", "") for f in FILES]
    return FILES
  
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, index):
    if index not in range(len(self.images)):
      return self.__getitem__(np.ramdom.randint(0, self.__len__()))
    
    transform = T.Compose([
      T.ToPILImage(),
      T.ToTensor(),
      T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if self.is_test:
      image_path = self.images[index]
      image = load_image(image_path)
      image = transform(image)
      return image, image_path
    else:
      image_path = self.images[index]
      mask_path = self.masks[index]

      image = transform(load_image(image_path))
      mask = transform(load_image(mask_path))
      return image, mask





