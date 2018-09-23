import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import functools
from torch.autograd import Variable

class UnetBlock(nn.Module):
  def __init__(self, outer_nc, inner_nc, input_nc=None,
        submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
    super(UnetBlock, self).__init__()
    self.outermost = outermost
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d
    if input_nc is None:
      input_nc = outer_nc
    downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                          stride=2, padding=1, bias=use_bias)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(inner_nc)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(outer_nc)

    if outermost:
      upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                  kernel_size=4, stride=2,
                                  padding=1)
      down = [downconv]
      up = [uprelu, upconv, nn.Tanh()]
      model = down + [submodule] + up
    elif innermost:
      upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                  kernel_size=4, stride=2,
                                  padding=1, bias=use_bias)
      down = [downrelu, downconv]
      up = [uprelu, upconv, upnorm]
      model = down + up
    else:
      upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                  kernel_size=4, stride=2,
                                  padding=1, bias=use_bias)
      down = [downrelu, downconv, downnorm]
      up = [uprelu, upconv, upnorm]

      if use_dropout:
          model = down + [submodule] + up + [nn.Dropout(0.5)]
      else:
          model = down + [submodule] + up

    self.model = nn.Sequential(*model)

  def forward(self, x):
    if self.outermost:
      return self.model(x)
    else:
      return torch.cat([x, self.model(x)], 1)

class Generatror(nn.Module):
  def __init__(self):
    super(Generatror, self).__init__()
    self.ngf = 64
    self.norm_layer = nn.BatchNorm2d
    self.num_down = 8
    self.input_nc = 3
    self.output_nc = 3

    
    block = UnetBlock(self.ngf*8, self.ngf*8, innermost=True)
    for i in range(self.num_down - 5):
      block = UnetBlock(self.ngf*8, self.ngf*8, submodule=block)
    block = UnetBlock(self.ngf*4, self.ngf*8, submodule=block)
    block = UnetBlock(self.ngf*2, self.ngf*4, submodule=block)
    block = UnetBlock(self.ngf, self.ngf*2, submodule=block)
    block = UnetBlock(self.output_nc, self.ngf, input_nc=self.input_nc ,submodule=block, outermost=True)
    
    self.model = block
  
  def forward(self, x):
    return self.model(x)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.input_nc = 3
    self.output_nc = 3
    self.ndf = 64
    self.n_layer = 3
    self.norm_layer = nn.BatchNorm2d
    self.use_sigmoid = True

    kw = 4
    padw = 1
    sequence = [
      nn.Conv2d(self.input_nc, self.ndf, kernel_size=kw, stride=2, padding=padw),
      nn.LeakyReLU(0.2, True)
    ]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, self.n_layer):
      nf_mult_prev = nf_mult
      nf_mult = min(2**n, 8)
      sequence += [
        nn.Conv2d(self.ndf*nf_mult_prev, self.ndf*nf_mult, kernel_size=kw, stride=2, padding=padw),
        self.norm_layer(self.ndf * nf_mult),
        nn.LeakyReLU(0.2, True)
      ]
    
    nf_mult_prev = nf_mult
    nf_mult = min(2**self.n_layer, 8)
    sequence += [
      nn.Conv2d(self.ndf*nf_mult_prev, self.ndf*nf_mult, kernel_size=kw, stride=1, padding=padw)
    ]
    
    self.model = nn.Sequential(*sequence)
  
  def forward(self, x):
    return self.model(x)

def get_G():
  G = Generatror()
  return init_net(G)


def get_D():
  D = Discriminator()
  return init_net(D)


def init_net(net, init_type='normal', init_gain=0.02):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  net.to(device)
  init_weight(net, gain=init_gain)
  return net


def init_weight(net, gain=0.02):
  def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
      init.normal_(m.weight.data, 0.0, gain)
        
    if hasattr(m, 'bias') and m.bias is not None:
      init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
      init.normal_(m.weight.data, 1.0, gain)
      init.constant_(m.bias.data, 0.0)
  net.apply(init_func)