"""Example of using bietorch library."""

import bietorch as bt
import numpy as np


def main() -> None:

  print(bt.__version__)
  print(bt.Tensor(np.array([1, 2, 3])))


if __name__ == '__main__':
  main()
