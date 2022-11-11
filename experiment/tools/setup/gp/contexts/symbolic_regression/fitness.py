"""Fitness measures."""
import math

from sklearn.metrics import mean_squared_error, r2_score

def mse(y_true, y_pred):
    """Mean-squared error."""
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    """Root-mean-squared error."""
    return math.sqrt(mse(y_true, y_pred))

def r2(y_true, y_pred):
    """Coefficient of determination."""
    return r2_score(y_true, y_pred)