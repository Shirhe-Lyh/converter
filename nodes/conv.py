from tensorflow.core.framework import node_def_pb2
from typing import List, Optional, Tuple

from nodes import ops


def conv_(name: str,
          op: str,
          inputs: List[str],
          # outputs: Optional[List[str]] = None,
          data_format: str = 'NHWC',
          dilations: Tuple[int] = (1, 1, 1, 1),
          explicit_paddings: Optional[Tuple[int]] = None,
          padding: str = 'SAME',
          strides: Tuple[int] = (1, 2, 2, 1),
          dtype: int = 1,
          use_cudnn_on_gpu: bool = True
          ) -> node_def_pb2.NodeDef:
  node = node_def_pb2.NodeDef()
  node.name = name
  node.op = op
  node.input.extend(inputs)
  ops.set_attr_type(node, key='T', dtype=dtype)
  ops.set_attr_str(node, key='data_format', value=data_format)
  ops.set_attr_list(node, key='dilations', value=dilations)
  if explicit_paddings:
    ops.set_attr_list(node, key='explicit_paddings', value=explicit_paddings)
  ops.set_attr_str(node, key='padding', value=padding)
  ops.set_attr_list(node, key='strides', value=strides)
  if use_cudnn_on_gpu:
    ops.set_attr_boolean(node, key='use_cudnn_on_gpu', value=use_cudnn_on_gpu)
  return node


def conv2d(name: str,
           inputs: List[str],
           data_format: str = 'NHWC',
           dilation: int = 1,
           explicit_paddings: Optional[Tuple[int]] = None,
           padding: str = 'SAME',
           stride: int = 1,
           dtype: int = 1,
           use_cudnn_on_gpu: bool = True
           ) -> node_def_pb2.NodeDef:
  dilations = (1, dilation, dilation, 1)
  strides = (1, stride, stride, 1)
  return conv_(name=name, op='Conv2D', inputs=inputs,
               data_format=data_format, dilations=dilations,
               explicit_paddings=explicit_paddings,
               padding=padding, strides=strides, dtype=dtype,
               use_cudnn_on_gpu=use_cudnn_on_gpu)


def depthwise_conv2d(name: str,
                     inputs: List[str],
                     data_format: str = 'NHWC',
                     dilation: int = 1,
                     explicit_paddings: Optional[Tuple[int]] = None,
                     padding: str = 'SAME',
                     stride: int = 1,
                     dtype: int = 1
                     ) -> node_def_pb2.NodeDef:
  dilations = (1, dilation, dilation, 1)
  strides = (1, stride, stride, 1)
  return conv_(name=name, op='DepthwiseConv2dNative', inputs=inputs,
               data_format=data_format, dilations=dilations,
               explicit_paddings=explicit_paddings,
               padding=padding, strides=strides, dtype=dtype,
               use_cudnn_on_gpu=False)


def bias_add(name: str,
             inputs: List[str],
             data_format: str = 'NHWC',
             dtype: int = 1) -> node_def_pb2.NodeDef:
  node = node_def_pb2.NodeDef()
  node.name = name
  node.op = 'BiasAdd'
  node.input.extend(inputs)
  ops.set_attr_type(node, key='T', dtype=dtype)
  ops.set_attr_str(node, key='data_format', value=data_format)
  return node
