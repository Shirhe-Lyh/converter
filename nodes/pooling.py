from tensorflow.core.framework import node_def_pb2
from typing import List, Tuple

from nodes import ops


def pool_(name: str,
          op: str,
          inputs: List[str],
          data_format: str = 'NHWC',
          kernel_size: Tuple[int] = (1, 3, 3, 1),
          padding: str = 'SAME',
          strides: Tuple[int] = (1, 2, 2, 1),
          dtype: int = 1
          ) -> node_def_pb2.NodeDef:
  node = node_def_pb2.NodeDef()
  node.name = name
  node.op = op
  node.input.extend(inputs)
  ops.set_attr_str(node, key='data_format', value=data_format)
  ops.set_attr_list(node, key='ksize', value=kernel_size)
  ops.set_attr_str(node, key='padding', value=padding)
  ops.set_attr_list(node, key='strides', value=strides)
  ops.set_attr_type(node, key='T', dtype=dtype)
  return node


def max_pool(name: str,
             inputs: List[str],
             data_format: str = 'NHWC',
             kernel_size: int = 3,
             padding: str = 'SAME',
             stride: int = 2,
             dtype: int = 1,
             dim: int = 2
             ) -> node_def_pb2.NodeDef:
  kernel_size = (1, *(dim*[kernel_size]), 1)
  strides = (1, *(dim*[stride]), 1)
  return pool_(name=name, op='MaxPool', inputs=inputs, data_format=data_format,
               kernel_size=kernel_size, padding=padding, strides=strides,
               dtype=dtype)


def avg_pool(name: str,
             inputs: List[str],
             data_format: str = 'NHWC',
             kernel_size: int = 3,
             padding: str = 'SAME',
             stride: int = 2,
             dtype: int = 1,
             dim: int = 2
             ) -> node_def_pb2.NodeDef:
  kernel_size = (1, *(dim*[kernel_size]), 1)
  strides = (1, *(dim*[stride]), 1)
  return pool_(name=name, op='AvgPool', inputs=inputs, data_format=data_format,
               kernel_size=kernel_size, padding=padding, strides=strides,
               dtype=dtype)
