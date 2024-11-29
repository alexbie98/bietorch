"""Tensor class."""

from abc import ABC, abstractmethod
import numbers
from typing import Any, Tuple

import numpy as np


class Tensor:
  """Tensor class."""

  def __init__(self, data: np.ndarray):
    assert np.issubdtype(data.dtype, np.number)
    self.data = data
    self.grad: np.ndarray = np.zeros_like(data)
    self._op: Op | None = None

  def backward(self):
    assert np.isscalar(self.data)
    self._backward(np.ones_like(self.data))

  def _backward(self, grad: np.ndarray):
    self.grad += _sum_to_shape(grad, self.data.shape)
    if self._op:
      self._op._backward(self.grad)

  def __str__(self) -> str:
    return (f'Tensor(\n{_indent("data=" + str(self.data))},'
            f'\n{_indent("grad=" + str(self.grad))},'
            f'\n{_indent("parent=" + str(self._op))},\n)')

  def __add__(self, other: Any):
    return add(self, to_tensor(other))

  def __mul__(self, other):
    return mult(self, to_tensor(other))

  __rmul__ = __mul__

  def __matmul__(self, other):
    return matmul(self, to_tensor(other))

  def __neg__(self):
    return self * -1


def _indent(text, indent='  '):
  return ''.join([indent + l for l in text.splitlines(True)])


class Op(ABC):

  @abstractmethod
  def _backward(self, grad: np.ndarray):
    pass

  def __str__(self) -> str:
    return f'{self.__class__.__name__}'


class MultOp(Op):
  def __init__(self, a: Tensor, b: Tensor):
    self._a = a
    self._b = b

  def _backward(self, grad: np.ndarray):
    agrad = grad * self._b.data
    bgrad = grad * self._a.data
    self._a._backward(agrad)
    self._b._backward(bgrad)


class AddOp(Op):

  def __init__(self, a: Tensor, b: Tensor):
    self._a = a
    self._b = b

  def _backward(self, grad: np.ndarray):
    self._a._backward(grad)
    self._b._backward(grad)


class MatmulOp(Op):

  def __init__(self, a: Tensor, b: Tensor):
    self._a = a  # [..., p, m]
    self._b = b  # [..., m, n]

  def _backward(self, grad: np.ndarray):
    # grad is [..., p, n]
    # [..., m, n] = [..., m, p] * [..., p, n]
    bgrad = np.matmul(self._a.swapaxes(-1, -2), grad)
    # [..., p, m] = [..., p, n] * [..., n, m]
    agrad = np.matmul(grad, self._b.swapaxes(-1, -2))
    self._a._backward(agrad)
    self._b._backward(bgrad)


def _sum_to_shape(a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
  assert a.shape[:-len(shape)] == shape
  x = np.sum(a, axis=tuple(range(-len(shape), 0)))
  return x


def add(a: Tensor, b: Tensor) -> Tensor:
  result = Tensor(a.data + b.data)
  result._op = AddOp(a, b)
  return result


def matmul(a: Tensor, b: Tensor) -> Tensor:
  result = Tensor(np.matmul(a.data, b.data))
  result._op = MatmulOp(a, b)
  return result


def mult(a: Tensor, b: Tensor) -> Tensor:
  result = Tensor(a.data * b.data)
  result._op = MultOp(a, b)
  return result


def to_tensor(a: Any) -> Tensor:
  if isinstance(a, Tensor):
    return a
  if isinstance(a, np.ndarray):
    return Tensor(a)
  if isinstance(a, numbers.Number):
    return Tensor(np.array(a))
  raise ValueError(f"Cannot convert {a} to Tensor.")
