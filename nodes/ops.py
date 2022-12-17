import numpy as np
import tensorflow as tf

from typing import List, Optional, Tuple, Union
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util


def set_attr_str(node: node_def_pb2.NodeDef, key: str, value: str):
  s = value.encode('utf-8')
  node.attr[key].CopyFrom(attr_value_pb2.AttrValue(s=s))


def set_attr_int(node: node_def_pb2.NodeDef, key: str, value: int):
  node.attr[key].CopyFrom(attr_value_pb2.AttrValue(i=value))


def set_attr_float(node: node_def_pb2.NodeDef, key: str, value: float):
  node.attr[key].CopyFrom(attr_value_pb2.AttrValue(f=value))


def set_attr_boolean(node: node_def_pb2.NodeDef, key: str, value: bool = True):
  node.attr[key].CopyFrom(attr_value_pb2.AttrValue(b=value))


def set_attr_type(node: node_def_pb2.NodeDef, key: str,
                  dtype: int = types_pb2.DT_FLOAT):
  node.attr[key].CopyFrom(attr_value_pb2.AttrValue(type=dtype))


def set_attr_list(node: node_def_pb2.NodeDef, key: str,
                  value: Union[List[int], Tuple[int]]):
  value = attr_value_pb2.AttrValue.ListValue(i=value)
  node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=value))


def set_shape(node: node_def_pb2.NodeDef, shape: Union[List[int], Tuple[int]]):
  shape = tf.TensorShape(dims=shape)
  node.attr['shape'].CopyFrom(attr_value_pb2.AttrValue(shape=shape.as_proto()))


def set_constant_value(node: node_def_pb2.NodeDef, value: np.ndarray,
                       shape: Optional[Union[List[int], Tuple[int]]] = None,
                       dtype: int = types_pb2.DT_FLOAT):
  set_attr_type(node, key='dtype', dtype=dtype)
  node.attr['value'].tensor.dtype = dtype
  if not isinstance(value, np.ndarray):
    value = np.array(value)
  shape = value.shape
  value = value.tobytes()
  shape = tf.TensorShape(dims=shape)
  node.attr['value'].tensor.tensor_shape.CopyFrom(shape.as_proto())
  node.attr['value'].tensor.tensor_content = value


def set_constant_scalar(node: node_def_pb2.NodeDef, value: Union[int, float],
                        dtype: int = types_pb2.DT_FLOAT):
  set_attr_type(node, key='dtype', dtype=dtype)
  node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
      tensor=tensor_util.make_tensor_proto(values=value, dtype=dtype)))
