import os
import torch
import torchvision as tv


def alexnet(pretrained_path=None):
  if pretrained_path and os.path.exists(pretrained_path):
    weights = None
  else:
    weights = tv.models.AlexNet_Weights.DEFAULT
  model = tv.models.alexnet(weights=weights)
  if not weights:
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
  return model
