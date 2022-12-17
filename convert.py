import copy
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from typing import List
from tensorflow.core.framework import graph_pb2

import graph


def cosin_similarity(a: np.ndarray, b: np.ndarray):
  a = a.flatten()
  b = b.flatten()
  sim = np.sum(a * b) / (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
  return sim


class Converter(object):
  """Convert PyTorch model to TensorFlow pb file."""

  def __init__(self, model: nn.Module) -> None:
    self.model = model.eval()
    self.graph_config = graph.GraphConfig()
    self.config = self.graph_config.generate(model=self.model)
    self.graph = graph.Graph(config=self.config)
    self.graph_def = self.graph.graph

  def save(self, output_path: str):
    self.graph.save(output_path=output_path)

  def check(self, threshold=0.999, verbose=True) -> bool:
    """Check precision by layerwise comparison."""
    low_precision_modules = []
    graph_pth: torch.fx.Graph = torch.fx.Tracer().trace(self.model)
    for node in graph_pth.nodes:
      node_dicts = []
      if node.op == 'call_module':
        node_dicts = self.graph_config.module_config(node)
        input_name = self.graph_config.alias_map.get(
            node.args[0].name, node.args[0].name)
        node_dicts.insert(0, {'op': 'placeholder', 'name': input_name})
        module_name = self.graph_config._get_module_name(node.name)
        module = self.model.get_submodule(module_name)
        if hasattr(module, 'in_channels'):
          shape = (1, 224, 224, module.in_channels)
        elif hasattr(module, 'num_features'):
          shape = (1, 224, 224, module.num_features)
        elif hasattr(module, 'in_features'):
          shape = (1, module.in_features)
        else:
          shape = (1, 224, 224, 3)

        graph_def = self.graph.create(node_dicts)
        inputs_np = np.random.randn(*shape).astype(np.float32)
        output_name = self.graph_config.alias_map.get(node.name, node.name)
        sim = self._calculate_similarity(
            module, graph_def, input_name, output_name, inputs_np)
        line = 'Module: {}\nNode name: {}\nCosine similarity: {}'.format(
            module, node.name, sim)
        if sim < threshold:
          low_precision_modules.append(line)
        if verbose:
          print('-' * 100)
          print(line)

    if low_precision_modules:
      print('\n')
      print('Unexpected convertion (with low precision):')
      for line in low_precision_modules:
        print('-' * 100)
        print('\033[1;31m{}\033[0m'.format(line))

    shape = (1, 224, 224, 3)
    inputs_np = np.random.randn(*shape).astype(np.float32)
    input_node_name = self.config[0]['name']
    output_node_name = self.config[-1]['name']
    sim = self._calculate_similarity(
        self.model, copy.deepcopy(self.graph.graph), input_node_name,
        output_node_name, inputs_np)
    print('\nCosine similarity of total graph: ', sim)
    if sim < threshold:
      return False
    return True

  def _calculate_similarity(self,
                            module: nn.Module,
                            graph_def: graph_pb2.GraphDef,
                            input_node_name: str,
                            output_node_name: str,
                            inputs_np: np.ndarray) -> float:
    if inputs_np.ndim == 4:
      inptus_pth = torch.from_numpy(inputs_np.transpose(0, 3, 1, 2))
    else:
      inptus_pth = torch.from_numpy(inputs_np)
    with torch.no_grad():
      outputs_pth = module(inptus_pth).data.cpu().numpy()
    if outputs_pth.ndim == 4:
      outputs_pth = outputs_pth.transpose(0, 2, 3, 1)

    graph = tf.Graph()
    with graph.as_default():
      tf.import_graph_def(graph_def=graph_def, name='')
    inputs = graph.get_tensor_by_name('{}:0'.format(input_node_name))
    outputs = graph.get_tensor_by_name('{}:0'.format(output_node_name))
    with tf.compat.v1.Session(graph=graph) as sess:
      outputs_tf = sess.run(outputs, feed_dict={inputs: inputs_np})
    sim = cosin_similarity(outputs_pth, outputs_tf)
    return sim


if __name__ == '__main__':
  # MobileNet
  from nets import mobilenet
  # pretrained_path = './models/mobilenet/mobilenet_v2-7ebf99e0.pth'
  # output_path = './models/mobilenet_v2.pb'
  # net = mobilenet.mobilenet_v2(pretrained_path=pretrained_path)
  # pretrained_path = './models/mobilenet/mobilenet_v3_small-047dcff4.pth'
  # output_path = './models/mobilenet_v3_small.pb'
  # net = mobilenet.mobilenet_v3_small(pretrained_path=pretrained_path)
  # ResNet
  from nets import resnet
  pretrained_path = './models/resnet/resnet50-0676ba61.pth'
  output_path = './models/resnet50.pb'
  net = resnet.resnet_50(pretrained_path=pretrained_path)

  converter = Converter(model=net)
  converter.save(output_path=output_path)
  converter.check(verbose=False)
