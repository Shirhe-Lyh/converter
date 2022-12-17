import inspect

from nodes import activation
from nodes import conv
from nodes import functional
from nodes import normalization
from nodes import pooling


def get_fn_names():
  """Return all function names in package nodes."""
  fn_names = []
  for module in [activation, conv, functional, normalization, pooling]:
    functions = inspect.getmembers(module, inspect.isfunction)
    for fn_name, _ in functions:
      fn_names.append(fn_name)
  return fn_names


def get_fn(fn_name: str):
  for module in [activation, conv, functional, normalization, pooling]:
    fn = getattr(module, fn_name, None)
    if fn is not None:
      return fn
  return None


def create(op: str, name: str, **kwargs):
  fn = get_fn(fn_name=op)
  if fn is None:
    fn_names = get_fn_names()
    raise ValueError('Unkown `op`: {}, expected: {}'.format(op, fn_names))
  return fn(name=name, **kwargs)
