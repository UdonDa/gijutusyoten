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
    # self.A_path = config.A_path
    # self.B_path = config.B_path
    BASE = "/host/space/horita-d/programing/python/conf/cvpr2018/renge/unet_pytorch_inpainting/txt"
    self.A_path = "{}/non_renge_short_paste_mask_train_train.txt".format(BASE)
    self.B_path = "{}/non_renge_short_train.txt".format(BASE)

    self.A_images = self._get_data_ary_from_txt(self.A_path)
    self.B_images = self._get_data_ary_from_txt(self.B_path)

    self.transform = T.Compose([
      T.ToPILImage(),
      T.ToTensor(),
      T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

  def _get_data_ary_from_txt(self, txt_path):
    FILE = open(txt_path)
    FILES = FILE.readlines()
    FILE.close()
    FILES = [f.replace("\n", "") for f in FILES]
    return FILES

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