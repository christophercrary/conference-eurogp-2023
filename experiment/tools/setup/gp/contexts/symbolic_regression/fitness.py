"""Fitness measures."""
import numpy as np

def mse(y_true, y_pred):
    """Mean-squared error."""
    return np.mean(np.square(np.subtract(y_true, y_pred)))

def rmse(y_true, y_pred):
    """Root-mean-squared error."""
    return np.sqrt(mse(y_true, y_pred))