from tensorflow.core.framework import node_def_pb2
from typing import List

from nodes import ops


def norm_(name: str,
          op: str,
          inputs: List[str],
          data_format: str = 'NHWC',
          epsilon: float = 0.001,
          is_training: bool = False,
          dtype: int = 1
          ) -> node_def_pb2.NodeDef:
  node = node_def_pb2.NodeDef()
  node.name = name
  node.op = op
  node.input.extend(inputs)
  ops.set_attr_str(node, key='data_format', value=data_format)
  ops.set_attr_float(node, key='epsilon', value=epsilon)
  ops.set_attr_boolean(node, key='is_training', value=is_training)
  ops.set_attr_type(node, key='T', dtype=dtype)
  return node


def batch_norm(name: str,
               inputs: List[str],
               data_format: str = 'NHWC',
               epsilon: float = 0.001,
               is_training: bool = False,
               dtype: int = 1
               ) -> node_def_pb2.NodeDef:
  return norm_(name=name, op='BatchNorm', inputs=inputs,
               data_format=data_format, epsilon=epsilon,
               is_training=is_training, dtype=dtype)


def fused_batch_norm(name: str,
                     inputs: List[str],
                     data_format: str = 'NHWC',
                     epsilon: float = 0.001,
                     is_training: bool = False,
                     dtype: int = 1
                     ) -> node_def_pb2.NodeDef:
  return norm_(name=name, op='FusedBatchNorm', inputs=inputs,
               data_format=data_format, epsilon=epsilon,
               is_training=is_training, dtype=dtype)
