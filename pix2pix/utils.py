import os
import numpy as np
from PIL import Image
import torch


def make_dir(path):
  if os.path.exists(path):
    return True
  else:
    if not os.path.isdir(path):
      os.makedirs(path)
    print("Making {}".format(path))

def tensor2im(input_image, imtype=np.uint8):
  if isinstance(input_image, torch.Tensor):
    image_tensor = input_image.data
  else:
    return input_image
  image_numpy = image_tensor[0].cpu().float().numpy()
  if image_numpy.shape[0] == 1:
    image_numpy = np.tile(image_numpy, (3, 1, 1))
  image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
  return image_numpy.astype(imtype)


def save_image(input_image, image_path):
  image_numpy = tensor2im(input_image)
  image_pil = Image.fromarray(image_numpy)
  image_pil.save(image_path)

def save_checkpoint(checkpoint_path, model, optimizer):
  state = {
    'state_dict' : model.state_dict(),
    'optimizer'   : optimizer.state_dict()}
  torch.save(state, checkpoint_path)