from tensorflow.core.framework import node_def_pb2
from typing import Any, List, Optional, Tuple, Union

from nodes import ops


def placeholder(name: str,
                shape: Optional[Union[List[int], Tuple[int]]] = None,
                dtype: int = 1) -> node_def_pb2.NodeDef:
  node = node_def_pb2.NodeDef()
  node.name = name
  node.op = 'Placeholder'
  ops.set_attr_type(node, key='dtype', dtype=dtype)
  if shape is not None:
    ops.set_shape(node, shape=shape)
  return node


def fake_quant(name: str,
               inputs: List[str],
               narrow_range: bool = False,
               num_bits: int = 8):
  node = node_def_pb2.NodeDef()
  node.name = name
  node.op = 'FakeQuantWithMinMaxVars'
  node.input.extend(inputs)
  ops.set_attr_boolean(node, key='narrow_range', value=narrow_range)
  ops.set_attr_int(node, key='num_bits', value=num_bits)
  return node


def constant(name: str,
             value: Any,
             shape: Optional[Union[List[int], Tuple[int]]] = None,
             dtype: int = 1
             ) -> node_def_pb2.NodeDef:
  node = node_def_pb2.NodeDef()
  node.name = name
  node.op = 'Const'
  if isinstance(value, (int, float)):
    ops.set_constant_scalar(node, value=value, dtype=dtype)
  else:
    ops.set_constant_value(node, value=value, shape=shape, dtype=dtype)
  return node


def function_(name: str,
              op: str,
              inputs: List[str],
              dtype: int = 1) -> node_def_pb2.NodeDef:
  node = node_def_pb2.NodeDef()
  node.name = name
  node.op = op
  node.input.extend(inputs)
  ops.set_attr_type(node, key='T', dtype=dtype)
  return node


def shape(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  node = function_(name=name, op='Shape', inputs=inputs, dtype=dtype)
  ops.set_attr_type(node, key='out_type', dtype=3)
  return node


def reshape(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  node = function_(name=name, op='Reshape', inputs=inputs, dtype=dtype)
  ops.set_attr_type(node, key='Tshape', dtype=3)
  return node


def squeeze(name: str, inputs: List[str],
            squeeze_dims: Union[List[int], Tuple[int]],
            dtype: int = 1) -> node_def_pb2.NodeDef:
  node = function_(name=name, op='Squeeze', inputs=inputs, dtype=dtype)
  ops.set_attr_list(node, key='squeeze_dims', value=squeeze_dims)
  return node


def flatten(name: str, inputs: List[str], Tshape: int = 3,
            dtype: int = 1) -> node_def_pb2.NodeDef:
  node = function_(name=name, op='Reshape', inputs=inputs, dtype=dtype)
  ops.set_attr_type(node, key='Tshape', dtype=Tshape)
  return node


def identity(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  return function_(name=name, op='Identity', inputs=inputs, dtype=dtype)


def add(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  return function_(name=name, op='Add', inputs=inputs, dtype=dtype)


def sub(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  return function_(name=name, op='Sub', inputs=inputs, dtype=dtype)


def mul(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  return function_(name=name, op='Mul', inputs=inputs, dtype=dtype)


def rsqrt(name: str, inputs: List[str], dtype: int = 1) -> node_def_pb2.NodeDef:
  return function_(name=name, op='Rsqrt', inputs=inputs, dtype=dtype)


def mean(name: str, inputs: List[str], keep_dims: bool = False, Tidx: int = 3,
         dtype: int = 1) -> node_def_pb2.NodeDef:
  node = function_(name=name, op='Mean', inputs=inputs, dtype=dtype)
  ops.set_attr_boolean(node, key='keep_dims', value=keep_dims)
  ops.set_attr_type(node, key='Tidx', dtype=Tidx)
  return node


def matmul(name: str, inputs: List[str], transpose_a: bool = False,
           transpose_b: bool = False, dtype: int = 1) -> node_def_pb2.NodeDef:
  node = function_(name=name, op='MatMul', inputs=inputs, dtype=dtype)
  ops.set_attr_boolean(node, key='transpose_a', value=transpose_a)
  ops.set_attr_boolean(node, key='transpose_b', value=transpose_b)
  return node


def pad(name: str, inputs: List[str], Tpaddings: int = 3,
        dtype: int = 1) -> node_def_pb2.NodeDef:
  node = function_(name=name, op='Pad', inputs=inputs, dtype=dtype)
  ops.set_attr_type(node, key='Tpaddings', dtype=Tpaddings)
  return node


def random_uniform(name: str, inputs: List[str], seed: int = 0, seed2: int = 0,
                   dtype: int = 1) -> node_def_pb2.NodeDef:
  node = function_(name=name, op='RandomUniform', inputs=inputs, dtype=3)
  ops.set_attr_type(node, key='dtype', dtype=dtype)
  ops.set_attr_int(node, key='seed', value=seed)
  ops.set_attr_int(node, key='seed2', value=seed2)
  return node


def great_equal(name: str, inputs: List[str], dtype: int = 1):
  return function_(name=name, op='GreaterEqual', inputs=inputs, dtype=dtype)


def select_v2(name: str, inputs: List[str], dtype: int = 1):
  return function_(name=name, op='SelectV2', inputs=inputs, dtype=dtype)
