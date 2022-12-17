from tensorflow.core.framework import node_def_pb2
from typing import List

from nodes import ops


def activation_(name: str,
                op: str,
                inputs: List[str],
                dtype: int = 1
                ) -> node_def_pb2.NodeDef:
  node = node_def_pb2.NodeDef()
  node.name = name
  node.op = op
  node.input.extend(inputs)
  ops.set_attr_type(node, key='T', dtype=dtype)
  return node


def relu(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  return activation_(name=name, op='Relu', inputs=inputs, dtype=dtype)


def relu6(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  return activation_(name=name, op='Relu6', inputs=inputs, dtype=dtype)


def softmax(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  return activation_(name=name, op='Softmax', inputs=inputs, dtype=dtype)


def sigmoid(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  return activation_(name=name, op='Sigmoid', inputs=inputs, dtype=dtype)
