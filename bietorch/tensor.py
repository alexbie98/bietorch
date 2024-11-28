"""Tensor class."""

import numpy as np


class Tensor:
  """Tensor class."""

  def __init__(self, data: np.ndarray, requires_grad: bool = False):
    self._data = data
    self._requires_grad = requires_grad
    self._grad = None
