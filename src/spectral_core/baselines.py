from typing import Dict, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


DEFAULT_RANDOM_STATE = 34
DEFAULT_CV_FOLDS = 10
DEFAULT_HEAVY_CV_FOLDS = 5

Candidate = Tuple[object, dict, int]


def build_candidates(
    n_features: int,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    heavy_cv_folds: int = DEFAULT_HEAVY_CV_FOLDS,
) -> Dict[str, Candidate]:
    """
    Build the benchmark models with their search grids.

    SVR is wrapped in a scaler because an RBF kernel is sensitive to feature
    scale; the tree models are not. The scaler sits inside the pipeline so it
    is refitted on each training fold rather than on all the data.

    The tree models use fewer folds than the others purely for runtime.

    Args:
        n_features: Width of the design matrix, which caps the PLSR components
        random_state: Seed for the tree models
        cv_folds: Folds used for PLSR and SVR
        heavy_cv_folds: Folds used for the tree models

    Returns:
        Dictionary mapping display name to (estimator, grid, folds)
    """
    return {
        "PLSR": (
            PLSRegression(),
            {"n_components": list(range(1, min(20, n_features) + 1))},
            cv_folds,
        ),
        "SVR (RBF)": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [1, 10, 100, 1000],
                "model__gamma": ["scale", 1e-4, 1e-3, 1e-2],
                "model__epsilon": [0.1, 0.5],
            },
            cv_folds,
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=300, random_state=random_state, n_jobs=1
            ),
            {
                "max_features": ["sqrt", 0.1, 0.3],
                "min_samples_leaf": [1, 3, 5],
            },
            heavy_cv_folds,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=300,
                subsample=0.8,
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
            ),
            {
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
                "colsample_bytree": [0.3, 0.7],
            },
            heavy_cv_folds,
        ),
    }
