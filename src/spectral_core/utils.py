"""Small helpers shared across the project.

Defined once so PLSR, the CNN and the baseline comparison all report the same
quantities computed the same way.
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

DEFAULT_COLOR = '#1f77b4'


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


def plot_predictions(
    y_true,
    y_pred,
    target_name: str = 'HbA1c%',
    figsize: Tuple[int, int] = (6, 4),
    text_pos: Optional[Tuple[float, float]] = None,
) -> Tuple:
    """
    Plot measured against predicted values with metrics.

    Shared by every model so the validation plots are directly comparable.

    Args:
        y_true: Measured values
        y_pred: Predicted values
        target_name: Axis label stem
        figsize: Figure size
        text_pos: (x, y) position for the metrics text. If None, auto-positioned

    Returns:
        Tuple of (figure, axes)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        x=y_true, y=y_pred, ax=ax, facecolor=DEFAULT_COLOR, edgecolor='k', s=25
    )
    ax.plot(y_true, y_true, color='k', linewidth=0.9)
    ax.set_xlabel(f'{target_name} measured')
    ax.set_ylabel(f'{target_name} predicted')

    if text_pos is None:
        top = max(y_true.max(), y_pred.max())
        span = top - min(y_true.min(), y_pred.min())
        text_pos = (min(y_true), top - 0.03 * span)

    ax.text(
        text_pos[0],
        text_pos[1],
        f"R² = {metrics['r2']:.3f}\nR = {metrics['r']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\nRMSE = {metrics['rmse']:.3f}\n"
        f"MSE = {metrics['mse']:.3f}",
        verticalalignment='top',
    )
    return fig, ax
