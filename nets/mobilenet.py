import os
import torch
import torchvision as tv


def mobilenet_v2(pretrained_path=None):
  if pretrained_path and os.path.exists(pretrained_path):
    weights = None
  else:
    weights = tv.models.MobileNet_V2_Weights.DEFAULT
  model = tv.models.mobilenet_v2(weights=weights)
  if not weights:
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
  return model


def mobilenet_v3_small(pretrained_path=None):
  if pretrained_path and os.path.exists(pretrained_path):
    weights = None
  else:
    weights = tv.models.MobileNet_V3_Small_Weights.DEFAULT
  model = tv.models.mobilenet_v3_small(weights=weights)
  if not weights:
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
  return model
