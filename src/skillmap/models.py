"""Per-site regressors and small, shared validation-search grids."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from skillmap.config import SEED

LINEAR_GRID = [{"alpha": alpha} for alpha in (0.1, 1.0, 10.0, 100.0)]
HGB_GRID = [
    {"max_iter": iterations, "max_leaf_nodes": leaves}
    for iterations in (100, 200, 300) for leaves in (7, 15)
]


def make_model(model: str, params: dict[str, Any]) -> Pipeline | HistGradientBoostingRegressor:
    """Construct a fresh estimator, with transforms fitted inside the pipeline."""
    if model == "linear":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(**params)
        )
    if model == "hgb":
        return HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, early_stopping=False,
            l2_regularization=1.0, random_state=SEED, min_samples_leaf=30, **params,
        )
    raise ValueError(f"unknown learned model: {model}")


def fit_predict(
    train: pd.DataFrame, predict: pd.DataFrame, columns: Sequence[str],
    model: str, params: dict[str, Any],
) -> NDArray[np.float64]:
    """Fit one site's fold and predict; targets must already be filtered."""
    estimator = make_model(model, params)
    estimator.fit(train[list(columns)], train["y_true"])
    return estimator.predict(predict[list(columns)])
