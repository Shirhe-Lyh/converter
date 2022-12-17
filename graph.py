import tensorflow as tf
import torch
import torch.nn as nn

from typing import List, Optional
from tensorflow.core.framework import graph_pb2

from nodes import node


class GraphConfig(object):
  """Translate nodes in PyTorch model to configurations.

  Note:
    The configuration list is as follows:
      [
        {
          'op': xxx,
          'name': xxx,
          'kwargs': {...}
        },
        {
          'op': xxx,
          'name': xxx,
          'kwargs': {...}
        },
        ...
      ]
  """

  def __init__(self) -> None:
    self.model = None
    self.module_names = []
    self.alias_map = {}

  def _get_module_names(self, module: nn.Module):
    module_names = []
    for name, _ in module.named_modules():
      module_names.append(name)
    return module_names

  def _get_module_name(self, name: str):
    module_name = ''
    name_lenght = 0
    for n in self.module_names:
      n_ = n.replace('.', '_')
      if name.startswith(n_):
        if len(n_) > name_lenght:
          module_name = n
          name_lenght = len(n_)
    return module_name

  def generate(self, model: Optional[nn.Module] = None) -> List[dict]:
    """Generate configurations of a PyTorch model."""
    if model is None or not isinstance(model, nn.Module):
      raise ValueError('`model` is invalid, expected a nn.Module object.')

    self.model = model
    self.alias_map.clear()
    self.module_names = self._get_module_names(model)

    config = []
    graph: torch.fx.Graph = torch.fx.Tracer().trace(model)
    for node in graph.nodes:
      node_dicts = []
      if node.op == 'placeholder':
        node_dicts = [{'op': 'placeholder', 'name': node.name}]
      elif node.op == 'call_function':
        node_dicts = self.function_config(node)
      elif node.op == 'call_module':
        node_dicts = self.module_config(node)
      config.extend(node_dicts)
    return config

  def _get_input_names(self, node: torch.fx.node.Node) -> List[str]:
    inputs = []
    for input_node in node.args:
      if isinstance(input_node, torch.fx.node.Node):
        name = self.alias_map.get(input_node.name, input_node.name)
        inputs.append(name)
    return inputs

  def _get_input_ints(self, node: torch.fx.node.Node) -> List[int]:
    ints = []
    for input_node in node.args:
      if isinstance(input_node, int):
        ints.append(input_node)
    return ints

  def function_config(self, node: torch.fx.node.Node) -> List[dict]:
    """Convert torch.function to NodeDef in TensorFlow.

    Note: One torch.fx.node.Node may be corresponding to multiple NodeDef. 
      For example, torch.flatten -> Reshape + reshape_dims.
    """
    op = node.target.__name__
    if op == 'flatten':
      node_dicts = self._flatten_config(node)
    elif op == 'adaptive_avg_pool2d':
      output_size = (1, 1)
      for input_node in node.args:
        if isinstance(input_node, tuple):
          output_size = input_node
      module = nn.AdaptiveAvgPool2d(output_size=output_size)
      node_dicts = self._adaptive_pool_config(node, module)
    else:
      inputs, kwargs, dims = [], {}, []
      inputs = self._get_input_names(node)
      if inputs:
        kwargs['inputs'] = inputs
      dims = self._get_input_ints(node)
      if dims:
        kwargs['{}_dims'.format(op)] = dims
      node_dict = {
          'op': op,
          'name': node.name,
          'kwargs': kwargs
      }
      node_dicts = [node_dict]
    return node_dicts

  def _flatten_config(self, node: torch.fx.node.Node) -> List[dict]:
    op, kwargs = 'flatten', {}
    inputs = self._get_input_names(node)
    if inputs:
      kwargs['inputs'] = inputs
    dims = self._get_input_ints(node)
    if len(dims) == 1 and dims[0] == 1:
      node_dicts = [{'op': op, 'name': node.name, 'kwargs': kwargs}]
      # Shape
      name = '{}/{}'.format(node.name, 'shape')
      value = [1, -1]
      kwargs = {'value': value, 'dtype': 3}
      node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
      node_dicts[0]['kwargs']['inputs'].append(name)
    else:
      raise ValueError('Unsupported flatten with dims: {}'.format(dims))
    return node_dicts

  def module_config(self, node: torch.fx.node.Node) -> List[dict]:
    """Convert nn.Module to NodeDef in TensorFlow.

    Note: One torch.fx.node.Node may be corresponding to multiple NodeDef. 
      For example, nn.Conv2d -> Conv2d + BiasAdd. And then, we may insert 
      additional NodeDefs, hence we must keep the node names consistent
      between PyTorch and TensorFlow carefully.
    """
    module_name = self._get_module_name(node.name)
    module = self.model.get_submodule(module_name)
    if isinstance(module, nn.Conv2d):
      node_dicts = self._conv_config(node, module)
    elif isinstance(module, nn.Linear):
      node_dicts = self._linear_config(node, module)
    elif isinstance(module, nn.Dropout):
      node_dicts = self._dropout_config(node, module)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      node_dicts = self._batch_norm_config(node, module, module_name)
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Softmax, nn.Sigmoid)):
      op = module.__class__.__name__.lower()
      inputs = self._get_input_names(node)
      kwargs = {}
      if inputs:
        kwargs['inputs'] = inputs
      node_dicts = [{'op': op, 'name': node.name, 'kwargs': kwargs}]
    elif isinstance(module, nn.Hardswish):
      node_dicts = self._hardswish_config(node)
    elif isinstance(module, nn.Hardsigmoid):
      node_dicts = self._hardsigmoid_config(node)
    elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                             nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
      node_dicts = self._pool_config(node, module)
    elif isinstance(module, (nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
      node_dicts = self._adaptive_pool_config(node, module)
    else:
      print(node.name)
      raise ValueError('Unkown module type: {}'.format(type(module)))

    return node_dicts

  def _conv_config(self, node: torch.fx.node.Node,
                   module: nn.Module) -> List[dict]:
    op, kwargs = 'conv2d', {}
    if (module.in_channels == module.out_channels and
            module.groups == module.out_channels):
      op = 'depthwise_conv2d'
    inputs = self._get_input_names(node)
    if inputs:
      kwargs['inputs'] = inputs
    kwargs['dilation'] = module.dilation[0]
    kwargs['stride'] = module.stride[0]
    kwargs['padding'] = 'VALID'
    node_dicts = [{'op': op, 'name': node.name, 'kwargs': kwargs}]

    # Additional constant nodes (weights, bias, ...)
    if isinstance(module, nn.Conv2d):
      # Explicit padding
      if module.padding[0] > 0:
        name = '{}/{}'.format(node.name, 'Pad/padding')
        value = [[0, 0], list(module.padding), list(module.padding), [0, 0]]
        kwargs = {'value': value, 'dtype': 3}
        node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
        inputs.append(name)
        kwargs = {'inputs': inputs}
        name = '{}/{}'.format(node.name, 'Pad')
        node_dicts.append({'op': 'pad', 'name': name, 'kwargs': kwargs})
        node_dicts[0]['kwargs']['inputs'] = [name]
      # Weight/Bias
      weights = module.weight.data.cpu().numpy().transpose(2, 3, 1, 0)
      if op == 'depthwise_conv2d':
        weights = weights.transpose(0, 1, 3, 2)
      name = '{}/{}'.format(node.name, 'weights')
      kwargs = {'value': weights}
      node_dicts[0]['kwargs']['inputs'].append(name)
      node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
      if module.bias is not None:
        weights = module.bias.data.cpu().numpy()
        name = '{}/{}'.format(node.name, 'bias')
        kwargs = {'value': weights}
        node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
        kwargs = {'inputs': [node.name, name]}
        name = '{}/{}'.format(node.name, 'BiasAdd')
        node_dicts.append({'op': 'bias_add', 'name': name, 'kwargs': kwargs})
        self.alias_map[node.name] = name
    return node_dicts

  def _linear_config(self, node: torch.fx.node.Node,
                     module: nn.Linear) -> List[dict]:
    op, kwargs = 'matmul', {}
    inputs = self._get_input_names(node)
    if inputs:
      kwargs['inputs'] = inputs
    node_dicts = [{'op': op, 'name': node.name, 'kwargs': kwargs}]

    # Weight and Bias
    weights = module.weight.data.cpu().numpy().transpose()
    name = '{}/{}'.format(node.name, 'weights')
    kwargs = {'value': weights}
    node_dicts[0]['kwargs']['inputs'].append(name)
    node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
    weights = module.bias.data.cpu().numpy()
    name = '{}/{}'.format(node.name, 'bias')
    kwargs = {'value': weights}
    node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
    kwargs = {'inputs': [node.name, name]}
    name = '{}/{}'.format(node.name, 'BiasAdd')
    node_dicts.append({'op': 'bias_add', 'name': name, 'kwargs': kwargs})
    self.alias_map[node.name] = name
    return node_dicts

  def _dropout_config(self, node: torch.fx.node.Node,
                      module: nn.Module,
                      training=False) -> List[dict]:
    """The convertion flow is (In training mode):
                  /-> Shape -> RandomUniform -> GreatEqual ->\ 
      nn.Dropout                                               -> SelectV2
                  \-> Mul(., 1 /(1 - p)) ------------------->/
    """
    inputs = self._get_input_names(node)

    if not training:
      node_dicts = [{
          'op': 'identity',
          'name': node.name,
          'kwargs': {'inputs': inputs}
      }]
      return node_dicts

    drop_prob = module.p
    inv_p = 1. / (1 - drop_prob)
    name = '{}/InverseP'.format(node.name)
    node_dicts = [{'op': 'constant', 'name': name, 'kwargs': {'value': inv_p}}]
    op, kwargs = 'mul', {}
    if inputs:
      kwargs['inputs'] = [*inputs, name]
    mul_name = '{}/Mul'.format(node.name)
    node_dicts.append({'op': op, 'name': mul_name, 'kwargs': kwargs})

    op, kwargs = 'shape', {}
    if inputs:
      kwargs['inputs'] = inputs
    shape_name = '{}/Shape'.format(node.name)
    node_dicts.append({'op': op, 'name': shape_name, 'kwargs': kwargs})
    op = 'random_uniform'
    kwargs = {'inputs': [shape_name]}
    uniform_name = '{}/RandomUniform'.format(node.name)
    node_dicts.append({'op': op, 'name': uniform_name, 'kwargs': kwargs})
    p_name = '{}/DropoutProb'.format(node.name)
    kwargs = {'value': drop_prob}
    node_dicts.append({'op': 'constant', 'name': p_name, 'kwargs': kwargs})
    op = 'great_equal'
    kwargs = {'inputs': [uniform_name, p_name]}
    great_euql_name = '{}/GreatEqual'.format(node.name)
    node_dicts.append({'op': op, 'name': great_euql_name, 'kwargs': kwargs})
    e_name = '{}/E'.format(node.name)
    kwargs = {'value': 0.}
    node_dicts.append({'op': 'constant', 'name': e_name, 'kwargs': kwargs})
    op = 'select_v2'
    kwargs = {'inputs': [great_euql_name, mul_name, e_name]}
    node_dicts.append({'op': op, 'name': node.name, 'kwargs': kwargs})
    return node_dicts

  def _batch_norm_config(self, node: torch.fx.node.Node,
                         module: nn.Module, module_name: str) -> List[dict]:
    op, kwargs = 'fused_batch_norm', {}
    inputs = self._get_input_names(node)
    if inputs:
      kwargs['inputs'] = inputs
    kwargs['epsilon'] = module.eps
    node_dicts = [{'op': op, 'name': node.name, 'kwargs': kwargs}]

    # Scale, Offset, Moving_mean, Moving_variance
    name = '{}/{}'.format(node.name, 'gamma')
    kwargs = {'value': module.weight.data.cpu().numpy()}
    node_dicts[0]['kwargs']['inputs'].append(name)
    node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
    name = '{}/{}'.format(node.name, 'beta')
    kwargs = {'value': module.bias.data.cpu().numpy()}
    node_dicts[0]['kwargs']['inputs'].append(name)
    node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
    for sub_name in ['running_mean', 'running_var']:
      name = '{}/{}'.format(node.name, sub_name)
      value = self.model.get_buffer('{}.{}'.format(module_name, sub_name))
      kwargs = {'value': value.data.cpu().numpy()}
      node_dicts[0]['kwargs']['inputs'].append(name)
      node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
    return node_dicts

  def _hardswish_config(self, node: torch.fx.node.Node) -> List[dict]:
    """hardswish(x) = x * relu6(x + 3) / 6."""
    inputs = self._get_input_names(node)
    node_dicts = self._hardsigmoid_config(node)
    name = '{}/Mul'.format(node.name)
    node_dicts[-1]['name'] = name
    kwargs = {'inputs': [*inputs, name]}
    node_dicts.append({'op': 'mul', 'name': node.name, 'kwargs': kwargs})
    return node_dicts

  def _hardsigmoid_config(self, node: torch.fx.node.Node) -> List[dict]:
    """hardsigmoid(x) = relu6(x + 3) / 6."""
    inputs = self._get_input_names(node)

    constant_name = '{}/Const_1'.format(node.name)
    kwargs = {'value': 3.}
    node_dicts = [{'op': 'constant', 'name': constant_name, 'kwargs': kwargs}]
    add_name = '{}/Add'.format(node.name)
    kwargs = {'inputs': [*inputs, constant_name]}
    node_dicts.append({'op': 'add', 'name': add_name, 'kwargs': kwargs})
    relu6_name = '{}/Relu6'.format(node.name)
    kwargs = {'inputs': [add_name]}
    node_dicts.append({'op': 'relu6', 'name': relu6_name, 'kwargs': kwargs})
    constant_name = '{}/Const_2'.format(node.name)
    node_dicts.append({'op': 'constant', 'name': constant_name,
                       'kwargs': {'value': 1 / 6.}})
    kwargs = {'inputs': [relu6_name, constant_name]}
    node_dicts.append({'op': 'mul', 'name': node.name, 'kwargs': kwargs})
    return node_dicts

  def _pool_config(self, node: torch.fx.node.Node,
                   module: nn.Module) -> List[dict]:
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
      op = 'max_pool'
    elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
      op = 'avg_pool'
    else:
      raise ValueError('Unsupported pool type: {}'.format(type(module)))

    kwargs = {}
    inputs = self._get_input_names(node)
    if inputs:
      kwargs['inputs'] = inputs
    kwargs['kernel_size'] = module.kernel_size
    kwargs['stride'] = module.stride
    kwargs['padding'] = 'VALID'
    dim = int(module.__class__.__name__[-2])
    kwargs['dim'] = dim
    node_dicts = [{'op': op, 'name': node.name, 'kwargs': kwargs}]

    # Explicit padding
    if module.padding > 0:
      padding = module.padding
      name = '{}/{}'.format(node.name, 'Pad/padding')
      if dim == 1:
        value = [[0, 0], [padding, padding], [0, 0]]
      elif dim == 2:
        value = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
      else:
        value = [[0, 0], [padding, padding], [padding, padding],
                 [padding, padding], [0, 0]]
      kwargs = {'value': value, 'dtype': 3}
      node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
      inputs.append(name)
      kwargs = {'inputs': inputs}
      name = '{}/{}'.format(node.name, 'Pad')
      node_dicts.append({'op': 'pad', 'name': name, 'kwargs': kwargs})
      node_dicts[0]['kwargs']['inputs'] = [name]
    return node_dicts

  def _adaptive_pool_config(self, node: torch.fx.node.Node,
                            module: nn.Module) -> List[dict]:
    kwargs = {}
    inputs = self._get_input_names(node)
    if inputs:
      kwargs['inputs'] = inputs

    if isinstance(module, nn.AdaptiveAvgPool2d):
      if module.output_size == 1 or module.output_size == (1, 1):
        op = 'mean'
        kwargs['keep_dims'] = True
        node_dicts = [{'op': op, 'name': node.name, 'kwargs': kwargs}]

        reduce_indices = [1, 2]
        name = '{}/{}'.format(node.name, 'reduction_indices')
        kwargs = {'value': reduce_indices, 'dtype': 3}
        node_dicts[0]['kwargs']['inputs'].append(name)
        node_dicts.append({'op': 'constant', 'name': name, 'kwargs': kwargs})
    if isinstance(module, nn.AdaptiveMaxPool2d):
      pass
    return node_dicts


class Graph(object):
  """Create TensorFlow's GraphDef object."""

  def __init__(self, config: List[dict]) -> None:
    """Constructor.

    Args:
      config: A list of dictionary. Each dict looks like:
        {
          'op': 'conv2d',
          'name': 'layers.conv1',
          'kwargs': {'stride': 2, 'inputs': [...], ...}
        }
    """
    self.graph = self.create(config=config)

  def create(self, config: List[dict]) -> graph_pb2.GraphDef:
    """Create graph by node->node translation."""
    graph_nodes = []
    for node_dict in config:
      op = node_dict.get('op', None)
      name = node_dict.get('name', None)
      kwargs = node_dict.get('kwargs', {})
      graph_node = node.create(op=op, name=name, **kwargs)
      graph_nodes.append(graph_node)

    graph = graph_pb2.GraphDef()
    graph.node.extend(graph_nodes)
    return graph

  def save(self, output_path: str):
    with tf.io.gfile.GFile(output_path, 'wb') as writer:
      writer.write(self.graph.SerializeToString())
