import os
import torch
import torchvision as tv


def resnet_50(pretrained_path=None):
  if pretrained_path and os.path.exists(pretrained_path):
    weights = None
  else:
    weights = tv.models.resnet.ResNet50_Weights.DEFAULT
  model = tv.models.resnet50(weights=weights)
  if not weights:
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
  return model
