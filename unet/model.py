import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def conv3x3(in_channels, out_channels, stride=1,
  padding=1, bias=True)