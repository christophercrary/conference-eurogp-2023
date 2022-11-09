"""Custom math functions."""
from math import ceil, log

def clog(a, b=2):
    """Return ceiling of logarithm, base `b`, of `a`."""
    return ceil(log(a, b))