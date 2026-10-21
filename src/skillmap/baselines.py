"""Persistence and fold-fitted seasonal climatology baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def annual_phase(dates: ArrayLike) -> NDArray[np.float64]:
    """Continuous day-of-year phase, including leap day without remapping."""
    return 2 * np.pi * pd.DatetimeIndex(dates).dayofyear.to_numpy() / 365.25


def harmonic_design(dates: ArrayLike) -> NDArray[np.float64]:
    """Intercept and three annual sine/cosine harmonics."""
    phase = annual_phase(dates)
    return np.column_stack(
        [np.ones(len(phase))]
        + [fn(k * phase) for k in range(1, 4) for fn in (np.sin, np.cos)]
    )


class Climatology:
    """Per-site harmonic regression; callers supply only fold training rows."""

    def fit(self, dates: ArrayLike, values: ArrayLike) -> Climatology:
        design = harmonic_design(dates)
        y = np.asarray(values, dtype=float)
        if y.ndim != 1 or len(y) != len(design):
            raise ValueError("dates and values must have the same length")
        valid = np.isfinite(y)
        if not valid.any():
            raise ValueError("climatology needs at least one finite training target")
        self.coef_ = np.linalg.lstsq(design[valid], y[valid], rcond=None)[0]
        return self

    def predict(self, dates: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "coef_"):
            raise ValueError("fit climatology before prediction")
        return harmonic_design(dates) @ self.coef_


def persistence(features: pd.DataFrame) -> NDArray[np.float64]:
    """Yesterday's Tmax, retaining NaN when yesterday was unobserved."""
    return features["tmax_lag1"].to_numpy(dtype=float)
