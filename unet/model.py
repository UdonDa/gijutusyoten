import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def conv3x3(in_channels, out_channels, stride=1,
  padding=1, bias=True, groups=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3,
    stride=stride, padding=padding, bias=bias, groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
  if mode == 'transpose':
    return nn.ConvTranspose2d(in_channels, out_channels,
      kernel_size=2, stride=2)
  else:
    return nn.Sequential(
      nn.Upsample(mode='bilinear', scale_factor=2),
      conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
  return nn.Conv2d(in_channels, out_channels,
    kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
  def __init__(self, in_channels, out_channels, pooling=True):
    super(DownConv, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.pooling = pooling
    self.bn = nn.BatchNorm2d(out_channels)
    
    self.conv1 = conv3x3(self.in_channels, self.out_channels)
    self.conv2 = conv3x3(self.out_channels, self.out_channels)

    if self.pooling:
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = F.relu(self.bn(self.conv1(x)))
    x = F.relu(self.bn(self.conv2(x)))
    before_pool = x
    if self.pooling:
      x = self.pool(x)
    return x, before_pool


class UpConv(nn.Module):
  def __init__(self, in_channels, out_channels,
    merge_mode='concat', up_mode='transpose'):
    super(UpConv, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.up_mode = up_mode
    self.bn = nn.BatchNorm2d(out_channels)

    self.upconv = upconv2x2(self.in_channels, self.out_channels,
      mode=self.up_mode)
    
    if self.merge_mode == 'concat':
      self.conv1 = conv3x3(2*self.out_channels, self.out_channels)
    else:
      self.conv1 = conv3x3(self.out_channels, self.out_channels)
    self.conv2 = conv3x3(self.out_channels, self.out_channels)

  def forward(self, down, up):
    up = self.upconv(up)
    if self.merge_mode == 'concat':
      x = torch.cat((up, down), 1)
    else:
      x = up + down
    x = F.relu(self.bn(self.conv1(x)))
    x = F.relu(self.bn(self.conv2(x)))
    return x


class UNet(nn.Module):
  def __init__(self, num_classes, in_channels=3, depth=5,
    start_filts=64, up_mode='transpose', merge_mode='concat'):
    super(UNet, self).__init__()

    self.up_mode = up_mode # available for 'transpose', 'upsample'
    self.merge_mode = merge_mode # available for 'concat', 'add'
    
    self.num_classes = num_classes
    self.in_channels = in_channels
    self.depth = depth
    self.start_filts = start_filts

    self.down_convs = []
    self.up_convs = []

    for i in range(depth):
      # Create Encoder
      ins = self.in_channels if i == 0 else outs
      outs = self.start_filts * (2**i)
      pooling = True if i < depth-1 else False
      down_conv = DownConv(ins, outs, pooling=pooling)
      self.down_convs.append(down_conv)
    self.down_convs = nn.ModuleList(self.down_convs)

    for i in range(depth-1):
      # Create Decoder
      ins = outs
      outs = ins // 2
      up_conv = UpConv(ins, outs,
        up_mode=up_mode, merge_mode=merge_mode)
      self.up_convs.append(up_conv)
    self.up_convs = nn.ModuleList(self.up_convs)
    
    self.conv_final = conv1x1(outs, self.num_classes)
    self.reset_params()


  @staticmethod
  def weight_init(module):
    if isinstance(module, nn.Conv2d):
      init.xavier_normal_(module.weight)
      init.constant_(module.bias, 0)


  def reset_params(self):
    for module in self.modules():
      self.weight_init(module)
  
  
  def forward(self, x):
    encoder_outs = []
    for module in self.down_convs:
      x, before_pool = module(x)
      encoder_outs.append(before_pool)
    
    for module in self.up_convs:
      before_pool = encoder_outs[-(i+2)]
      x = module(before_pool, x)
    x = self.conv_final(x)
    return x