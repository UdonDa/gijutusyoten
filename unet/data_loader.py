path = './'

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms as T
from torch.autograd import Variable
from PIL import Image


def get_data_ary_from_txt(txt_path):
  FILE = open(txt_path)
  FILES = FILE.readlines()
  FILE.close()
  FILES = [f.replace("\n", "") for f in FILES]
  return FILES
class Dataset(torch.utils.data.Dataset):
  def __init__(self, image_path, mask_path, is_test=False):
    self.images = get_data_ary_from_txt(image_path)
    self.masks = get_data_ary_from_txt(mask_path)
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
      image = self.transform(Image.open(self.images[index]).convert('RGB'))
      mask = self.transform(Image.open(self.masks[index]).convert('RGB'))
      return image, mask


def get_dataset_train_and_val():
  BASE = "/host/space/horita-d/programing/python/conf/cvpr2018/renge/unet_pytorch_inpainting/txt/"

  TRAIN_IMAGE_PATH = BASE + "non_renge_short_paste_mask_train_train.txt"
  TRAIN_MASK_PATH = BASE + "non_renge_short_train.txt"
  TEST_IMAGE_PATH = BASE + "non_renge_short_paste_mask_train_test.txt" 
  TEST_MASK_PATH = BASE + "non_renge_short_test.txt"

  dataset_train = Dataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH)
  dataset_val = Dataset(TEST_IMAGE_PATH, TEST_MASK_PATH)

  print("Loaded dataset...")
  return dataset_train, dataset_val

def get_dataset_test():
  BASE = "/host/space/horita-d/programing/python/conf/cvpr2018/renge/unet_pytorch_inpainting/txt/"
  TRAIN_IMAGE_PATH = BASE + "non_renge_short_paste_mask_train_train.txt"

  IMAGE_PATH = "/host/space/horita-d/dataset/RENGE11k/txt/renge.txt"
  # dataset_test = Dataset(IMAGE_PATH, IMAGE_PATH, True)
  dataset_test = Dataset(TRAIN_IMAGE_PATH, IMAGE_PATH, True)
  print("Loaded dataset...")
  return dataset_test