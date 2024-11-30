"""Utility functions."""


def left_indent(text: str, indent: str = '  '):
  return ''.join([indent + l for l in text.splitlines(True)])
