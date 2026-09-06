from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

DEFAULT_COLOR = "#1f77b4"


class PLSRComponents:
    """
    PLSR model with automatic component selection via cross-validation.
    Determines optimal number of components using Q² threshold criterion.
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
        threshold: float = 0.0,
    ) -> int:
        """
        Fit PLSR model and determine optimal number of components.

        Components are added while Q^2 = 1 - PRESS_k / RSS_(k-1) stays above
        threshold. The default of 0 is Wold's R criterion (PRESS_k < RSS_(k-1)):
        keep adding while each component still improves prediction. Tenenhaus
        (1998), as used by SIMCA, applies the stricter 0.0975 instead.

        Note that the returned count includes the component that first falls
        below threshold; the stricter reading of Wold stops one component
        earlier. Q^2 is not necessarily monotonic, so a threshold above 0 can
        stop at a local dip.

        Args:
            X: Training spectra (n_samples, n_features)
            y: Training target values (n_samples,)
            ncomp: Maximum number of components to test
            cv: Number of cross-validation folds
            threshold: Q² threshold for component selection

        Returns:
            Optimal number of components
        """
        self._ncomp = ncomp
        self._x_train = X
        self._y_train = y

        self.r2s = []
        self.rmses = []
        self.y_cvs = []
        self.q2s = []
        self.press_l = []
        self.num_comp = None

        ress_l_minus_1 = np.sum((y - np.mean(y)) ** 2)

        for num_components in range(1, ncomp + 1):
            pls = PLSRegression(n_components=num_components, **self.plsr_params)

            y_cv = cross_val_predict(pls, X, y, cv=cv)
            r2 = r2_score(y, y_cv)
            rmse = np.sqrt(mean_squared_error(y, y_cv))

            press = np.sum((y - y_cv) ** 2)
            self.press_l.append(press)
            q2 = 1 - (press / ress_l_minus_1)
            self.q2s.append(q2)

            ress_l_minus_1 = press

            self.r2s.append(r2)
            self.rmses.append(rmse)
            self.y_cvs.append(y_cv)

            if q2 < threshold and self.num_comp is None:
                self.num_comp = num_components

        if self.num_comp is None:
            raise ValueError(
                f"Q² stayed above {threshold} for all {ncomp} components, so no "
                "component count was selected. Increase ncomp to search further, "
                "or raise threshold."
            )

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
        fig, ax = plt.subplots(figsize=figsize)

        r2 = r2_score(y_true, y_pred)
        r = pearsonr(y_true, y_pred)[0]
        mse = mean_squared_error(y_true, y_pred)
        mae = np.abs(y_true - y_pred).mean()
        rmse = np.sqrt(mse)

        sns.scatterplot(
            x=y_true, y=y_pred, ax=ax, facecolor=DEFAULT_COLOR, edgecolor="k", s=25
        )
        ax.plot(y_true, y_true, color="k", linewidth=0.9)

        ax.set_xlabel(f"{self.target_name} measured")
        ax.set_ylabel(f"{self.target_name} predicted")

        if text_pos is None:
            text_pos = (min(y_true), max(y_pred))

        ax.text(
            text_pos[0],
            text_pos[1],
            f"R² = {r2:.3f}\nR = {r:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}\nMSE = {mse:.3f}",
            verticalalignment="top",
        )

        return fig, ax
