
from bietorch.tensor import Tensor, Op


def indent(text, indent='  '):
  return ''.join([indent + l for l in text.splitlines(True)])


def print_graph(x: Tensor | Op | None, indent='| ') -> None:
  """Prints the computation graph for the given tensor."""
  if x is None:
    return
  if isinstance(x, Tensor):
    print(indent(str(x), indent))
    print_graph(x._op, indent + '  ')
  else:
    print(indent(str(x), indent))
    print_graph(x._a, indent + '  ')
    print_graph(x._b, indent + '  ')
