from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

from .utils import plot_predictions

DEFAULT_COLOR = "#1f77b4"


class PLSRComponents:
    """
    PLSR model with automatic component selection via cross-validation.
    Determines the optimal number of components by minimum cross-validated RMSE.
    """

    def __init__(self, target_name: str = "Target", **kwargs) -> None:
        """
        Initialize PLSR model.

        Args:
            target_name: Name of target variable for plot labels
            **kwargs: Additional parameters passed to PLSRegression
        """
        self.target_name = target_name
        self.plsr_params = kwargs
        self._fitted_model: Optional[PLSRegression] = None
        self.num_comp: Optional[int] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ncomp: int,
        cv: int = 10,
    ) -> int:
        """
        Fit PLSR and select the component count minimising cross-validated error.

        Every count from 1 to ncomp is scored by cross-validation and the one
        with the lowest RMSECV is retained.

        Args:
            X: Training spectra (n_samples, n_features)
            y: Training target values (n_samples,)
            ncomp: Maximum number of components to test
            cv: Number of cross-validation folds

        Returns:
            Selected number of components
        """
        self._ncomp = ncomp
        self._x_train = X
        self._y_train = y

        self.r2s = []
        self.rmses = []
        self.y_cvs = []

        for num_components in range(1, ncomp + 1):
            pls = PLSRegression(n_components=num_components, **self.plsr_params)
            y_cv = cross_val_predict(pls, X, y, cv=cv)

            self.r2s.append(r2_score(y, y_cv))
            self.rmses.append(np.sqrt(mean_squared_error(y, y_cv)))
            self.y_cvs.append(y_cv)

        self.num_comp = int(np.argmin(self.rmses)) + 1

        pls = PLSRegression(n_components=self.num_comp, **self.plsr_params)
        pls.fit(X, y)

        self._fitted_model = pls

        return self.num_comp

    def plot_cv_results(self, figsize: Tuple[int, int] = (10, 8)) -> Tuple:
        """
        Plot cross-validation results: R², RMSE, predictions, and residuals.

        Args:
            figsize: Figure size

        Returns:
            Tuple of (figure, axes)
        """
        if self.num_comp is None:
            raise RuntimeError("Call fit() before plotting cross-validation results.")

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        y_cv = self.y_cvs[self.num_comp - 1]

        sns.lineplot(
            x=np.arange(1, self._ncomp + 1),
            y=self.r2s,
            ax=axs[0, 0],
            marker="o",
            color="k",
            markeredgecolor="k",
            markerfacecolor=DEFAULT_COLOR,
            markersize=6,
        )
        axs[0, 0].set_xlabel("Number of components")
        axs[0, 0].set_ylabel("R²")
        axs[0, 0].axvline(
            self.num_comp, color="k", linestyle="--", linewidth=0.9
        )
        axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

        sns.lineplot(
            x=np.arange(1, self._ncomp + 1),
            y=self.rmses,
            ax=axs[0, 1],
            marker="o",
            color="k",
            markeredgecolor="k",
            markerfacecolor=DEFAULT_COLOR,
            markersize=6,
        )
        axs[0, 1].set_xlabel("Number of components")
        axs[0, 1].set_ylabel("RMSECV")
        axs[0, 1].axvline(
            self.num_comp, color="k", linestyle="--", linewidth=0.9
        )
        axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

        sns.scatterplot(x=self._y_train, y=y_cv, ax=axs[1, 0], edgecolor="k", s=25)
        axs[1, 0].plot(
            self._y_train, self._y_train, color="k", linestyle="--", linewidth=0.9
        )
        axs[1, 0].set_xlabel(f"{self.target_name} measured")
        axs[1, 0].set_ylabel(f"{self.target_name} predicted")

        residuals = y_cv - self._y_train
        sns.scatterplot(x=self._y_train, y=residuals, ax=axs[1, 1], edgecolor="k", s=25)
        axs[1, 1].axhline(y=0, color="k", linestyle="--", linewidth=0.9)
        axs[1, 1].set_xlabel(self.target_name)
        axs[1, 1].set_ylabel("Residuals")

        return fig, axs

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        figsize: Tuple[int, int] = (6, 4),
        text_pos: Optional[Tuple[float, float]] = None,
    ) -> Tuple:
        """
        Evaluate model on test set and plot results.

        Args:
            X_test: Test spectra
            y_test: Test target values
            figsize: Figure size
            text_pos: (x, y) position for metrics text. If None, auto-positioned

        Returns:
            Tuple of (figure, axes)
        """
        y_pred = self._require_fitted().predict(X_test)
        return self._plot_predictions(y_test, y_pred, figsize, text_pos)

    def get_fitted_model(self) -> PLSRegression:
        """Get the fitted PLSRegression model."""
        return self._require_fitted()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions for test data."""
        return self._require_fitted().predict(X_test)

    def _require_fitted(self) -> PLSRegression:
        """Return the fitted model, or explain that fit() has not run yet."""
        if self._fitted_model is None:
            raise RuntimeError("Call fit() before using the model.")
        return self._fitted_model

    def _plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (6, 4),
        text_pos: Optional[Tuple[float, float]] = None,
    ) -> Tuple:
        """
        Plot true vs predicted values with metrics.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            figsize: Figure size
            text_pos: (x, y) position for metrics text

        Returns:
            Tuple of (figure, axes)
        """
        return plot_predictions(
            y_true, y_pred, target_name=self.target_name,
            figsize=figsize, text_pos=text_pos,
        )
