"""GP functions.

It is expected that all function inputs will be numeric.
"""
import numpy as np

def add(x1, x2):
    """Return result of addition."""
    res = np.add(x1, x2)
    if np.isinf(res) or np.isnan(res):
        res = np.inf
    return res

def aq(x1, x2):
    """Return result of analytical quotient.
    
    The analytical quotient is as defined by Ni et al. in their paper 
    'The use of an analytic quotient operator in genetic programming':  
    `aq(x1, x2) = (x1)/(sqrt(1+x2^(2)))`.
    """
    res = np.divide(x1, np.sqrt(np.add(1, np.square(x2))))
    if np.isinf(res) or np.isnan(res):
        res = np.inf
    return res

def exp(x): 
    """Return result of exponentiation, base `e`."""
    return np.exp(x)

def log(x):
    """Return result of protected logarithm, base `e`."""
    return np.log(np.abs(x)) if x != 0 else 0

def mul(x1, x2):
    """Return result of multiplication."""
    res = np.multiply(x1, x2)
    if np.isinf(res) or np.isnan(res):
        res = np.inf
    return res

def sin(x):
    """Return result of sine."""
    res = np.sin(x)
    if np.isnan(res):
        res = np.inf
    return res

def sqrt(x):
    """Return result of protected square root."""
    return np.sqrt(np.abs(x))

def sub(x1, x2):
    """Return result of subtraction."""
    res = np.subtract(x1, x2)
    if np.isinf(res) or np.isnan(res):
        res = np.inf
    return res

def tanh(x):
    """Return result of hyperbolic tangent."""
    return np.tanh(x)