"""Paired moving-block inference on calendar-aligned absolute errors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skillmap.config import BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPS, SEED


def moving_block_weights(
    n_dates: int, *, block_days: int = BOOTSTRAP_BLOCK_DAYS,
    reps: int = BOOTSTRAP_REPS, seed: int = SEED,
) -> NDArray[np.int64]:
    """Date multiplicities for ordinary overlapping blocks, last block truncated.

    One row is one replicate of exactly n_dates draws. Reuse the entire matrix
    across sites to preserve contemporaneous dependence, including missing days.
    """
    if n_dates < 1 or block_days < 1 or reps < 1:
        raise ValueError("n_dates, block_days and reps must be positive")
    length = min(block_days, n_dates)
    n_blocks = (n_dates + length - 1) // length
    starts = np.random.default_rng(seed).integers(0, n_dates - length + 1, (reps, n_blocks))
    lengths = np.full(n_blocks, length)
    lengths[-1] = n_dates - length * (n_blocks - 1)
    differences = np.zeros((reps, n_dates + 1), dtype=np.int64)
    rows = np.arange(reps)[:, None]
    np.add.at(differences, (rows, starts), 1)
    np.add.at(differences, (rows, starts + lengths), -1)
    return differences[:, :-1].cumsum(axis=1)


def paired_bootstrap(
    model_errors: ArrayLike, reference_errors: ArrayLike, *,
    block_days: int = BOOTSTRAP_BLOCK_DAYS, reps: int = BOOTSTRAP_REPS,
    seed: int = SEED, weights: NDArray[np.int64] | None = None,
) -> dict[str, float]:
    """Return paired MAE difference/skill and percentile 95% intervals.

    Inputs are absolute errors, dates by sites (or a single site's vector).
    NaNs mark unavailable calendar dates; both losses use their paired support.
    Group MAEs average site MAEs equally, including in every replicate. A
    replicate lacking any site's observations is undefined and excluded from CIs.
    Skill is undefined (NaN) whenever the reference MAE is zero.
    """
    model = np.asarray(model_errors, dtype=float)
    reference = np.asarray(reference_errors, dtype=float)
    if model.shape != reference.shape or model.ndim not in (1, 2):
        raise ValueError("paired errors must have identical 1D or 2D shapes")
    if model.ndim == 1:
        model, reference = model[:, None], reference[:, None]
    valid = np.isfinite(model) & np.isfinite(reference)
    if not valid.any(axis=0).all():
        raise ValueError("every site needs paired observations")
    if ((model[valid] < 0).any() or (reference[valid] < 0).any()):
        raise ValueError("absolute errors must be nonnegative")
    weights = moving_block_weights(
        len(model), block_days=block_days, reps=reps, seed=seed
    ) if weights is None else np.asarray(weights)
    if weights.ndim != 2 or weights.shape[1] != len(model):
        raise ValueError("weights must have one column per calendar date")
    model = np.where(valid, model, 0.0)
    reference = np.where(valid, reference, 0.0)
    counts = valid.sum(axis=0)
    mae_model = float(np.mean(model.sum(axis=0) / counts))
    mae_ref = float(np.mean(reference.sum(axis=0) / counts))
    sampled_counts = weights @ valid.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sampled_model = np.mean((weights @ model) / sampled_counts, axis=1)
        sampled_ref = np.mean((weights @ reference) / sampled_counts, axis=1)
        sampled_skill = np.where(sampled_ref > 0, 1 - sampled_model / sampled_ref, np.nan)
    delta_low, delta_high = _interval(sampled_ref - sampled_model)
    skill_low, skill_high = _interval(sampled_skill)
    return {
        "mae_model": mae_model, "mae_ref": mae_ref, "delta_mae": mae_ref - mae_model,
        "delta_ci_low": delta_low, "delta_ci_high": delta_high,
        "skill": 1 - mae_model / mae_ref if mae_ref > 0 else float("nan"),
        "skill_ci_low": skill_low, "skill_ci_high": skill_high,
    }


def _interval(values: NDArray[np.float64]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)
