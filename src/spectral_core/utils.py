"""Small helpers shared across the project.

Defined once so PLSR, the CNN and the baseline comparison all report the same
quantities computed the same way.
"""

from typing import Dict

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute the metrics reported in the manuscript.

    Args:
        y_true: Measured values
        y_pred: Predicted values, of any shape broadcastable to 1-D

    Returns:
        Dictionary with r2, r, mae, mse and rmse
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_pred)

    return {
        'r2': float(r2_score(y_true, y_pred)),
        'r': float(pearsonr(y_true, y_pred)[0]),
        'mae': float(np.abs(y_true - y_pred).mean()),
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
    }
