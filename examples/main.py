"""Example of using bietorch library."""

import bietorch as bt
import numpy as np


def main() -> None:
  print('bietorch version: ', bt.__version__)

  np.random.seed(0)
  a = bt.Tensor(np.array(np.random.rand()))
  print('a:')
  print(a)

  z = (3.0 * a) + 2.0
  print('z = 3a+2:')
  print(z)

  print('a.grad:')
  print(a.grad)

  print('z graph:')
  bt.utils.print_graph(z)

  print('running z.backward()')
  z.backward()

  print('z graph:')
  bt.utils.print_graph(z)

  print('a.grad:')
  print(a.grad)


if __name__ == '__main__':
  main()
