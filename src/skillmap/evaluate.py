"""Chronological per-site experiments, shared tuning and paired inference."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from threadpoolctl import threadpool_limits

from skillmap.baselines import Climatology, persistence
from skillmap.bootstrap import moving_block_weights, paired_bootstrap
from skillmap.config import (
    ABLATION_COLUMNS,
    APP_DATA,
    BASELINES,
    BOOTSTRAP_BLOCK_DAYS,
    BOOTSTRAP_REPS,
    GROUP_QUOTAS,
    METRIC_COLUMNS,
    MODELS,
    PREDICTION_COLUMNS,
    PROCESSED,
    SEED,
    SKILL_GROUP_COLUMNS,
    SKILL_SITE_COLUMNS,
    SKILL_VALUE_COLUMNS,
    SPLITS,
)
from skillmap.features import FEATURE_GROUPS, add_climatology_anomaly, build_features
from skillmap.models import HGB_GRID, LINEAR_GRID, POOLED_GRID, fit_predict, pooled_predict

LEARNED_MODELS = [model for model in MODELS if model not in BASELINES]


def metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float | int]:
    """Finite paired MAE, RMSE and signed forecast-minus-observation bias."""
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("truth and predictions must have the same shape")
    errors = (pred - true)[np.isfinite(true) & np.isfinite(pred)]
    if not len(errors):
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "n": 0}
    return {
        "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)), "n": len(errors),
    }


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error on finite paired observations."""
    return float(metrics(y_true, y_pred)["mae"])


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean square error on finite paired observations."""
    return float(metrics(y_true, y_pred)["rmse"])


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean forecast-minus-observation error on finite paired observations."""
    return float(metrics(y_true, y_pred)["bias"])


def split_mask(dates: pd.Series, split: str) -> pd.Series:
    """Inclusive contract split boundaries, indexed by target date."""
    return dates.between(*SPLITS[split])


@dataclass
class SiteFold:
    """One site's feature tables with fold-specific climatology anomalies."""

    site_id: str
    train: pd.DataFrame
    val: pd.DataFrame
    refit: pd.DataFrame
    test: pd.DataFrame
    climatology_val: NDArray[np.float64]
    climatology_test: NDArray[np.float64]


def _prepare_site(table: pd.DataFrame) -> SiteFold:
    table = table[np.isfinite(table["y_true"])].copy()
    train = table.loc[split_mask(table["date"], "train")]
    val = table.loc[split_mask(table["date"], "val")]
    test = table.loc[split_mask(table["date"], "test")]
    site_id = str(table["site_id"].iloc[0]) if len(table) else "<empty>"
    if any(frame.empty for frame in (train, val, test)):
        raise ValueError(f"{site_id}: each split needs finite targets")
    refit = pd.concat([train, val])
    climate_train = Climatology().fit(train["date"], train["y_true"])
    climate_refit = Climatology().fit(refit["date"], refit["y_true"])
    frames = [add_climatology_anomaly(frame, climate) for frame, climate in (
        (train, climate_train), (val, climate_train), (refit, climate_refit), (test, climate_refit)
    )]
    climates = (climate_train, climate_train, climate_refit, climate_refit)
    for frame, climate in zip(frames, climates, strict=True):
        frame["climatology_target"] = climate.predict(frame["date"])
    return SiteFold(
        site_id, *frames,
        climate_train.predict(val["date"]), climate_refit.predict(test["date"]),
    )


def _validation_candidates(
    fold: SiteFold, columns: Sequence[str], model: str, grid: Sequence[dict[str, Any]],
) -> list[NDArray[np.float64]]:
    return [fit_predict(fold.train, fold.val, columns, model, params) for params in grid]


def _choose_baselines(predictions: pd.DataFrame) -> dict[str, str]:
    choices = {}
    for site_id, rows in predictions.loc[predictions["split"] == "val"].groupby("site_id"):
        # Compare on identical observed dates so missing persistence cannot give it an advantage.
        paired = rows.dropna(subset=["y_true", *BASELINES])
        if paired.empty:
            raise ValueError(f"{site_id}: no paired validation baseline observations")
        choices[site_id] = min(BASELINES, key=lambda model: metrics(paired["y_true"], paired[model])["mae"])
    return choices


def _reference_name(choices: dict[str, str], site_ids: Sequence[str], reference: str) -> str:
    if reference != "best_baseline":
        return reference
    names = {choices[site_id] for site_id in site_ids}
    return next(iter(names)) if len(names) == 1 else "per_site"


def _skill_tables(
    predictions: pd.DataFrame, sites: pd.DataFrame, reps: int, block_days: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = _choose_baselines(predictions)
    models = [*MODELS, *(["hgb_pooled"] if "hgb_pooled" in predictions else [])]
    site_rows, group_rows = [], []
    for split, rows in predictions.groupby("split", sort=False):
        dates = pd.date_range(*SPLITS[split], freq="D")
        site_ids = sorted(rows["site_id"].unique())
        truth = rows.pivot(index="date", columns="site_id", values="y_true").reindex(
            index=dates, columns=site_ids
        )
        errors = {
            model: (rows.pivot(index="date", columns="site_id", values=model).reindex(
                index=dates, columns=site_ids
            ) - truth).abs().to_numpy()
            for model in models
        }
        errors["best_baseline"] = np.column_stack([
            errors[choices[site_id]][:, i] for i, site_id in enumerate(site_ids)
        ])
        weights = moving_block_weights(len(dates), reps=reps, block_days=block_days, seed=seed)
        group_map = sites.set_index("site_id")["koppen_group"]
        for model in [name for name in models if name not in BASELINES]:
            for reference in [*BASELINES, "best_baseline"]:
                for i, site_id in enumerate(site_ids):
                    summary = paired_bootstrap(errors[model][:, i], errors[reference][:, i], weights=weights)
                    site_rows.append({
                        "site_id": site_id, "split": split, "model": model, "reference": reference,
                        "reference_model": _reference_name(choices, [site_id], reference), **summary,
                    })
                for group in [*GROUP_QUOTAS, "ALL"]:
                    indices = [i for i, site_id in enumerate(site_ids)
                               if group == "ALL" or group_map.loc[site_id] == group]
                    summary = paired_bootstrap(
                        errors[model][:, indices], errors[reference][:, indices], weights=weights
                    ) if indices else {key: np.nan for key in SKILL_VALUE_COLUMNS if key != "reference_model"}
                    group_rows.append({
                        "koppen_group": group, "split": split, "model": model, "reference": reference,
                        "reference_model": _reference_name(
                            choices, [site_ids[i] for i in indices], reference
                        ),
                        **summary, "n_sites": len(indices),
                    })
    return pd.DataFrame(site_rows, columns=SKILL_SITE_COLUMNS), pd.DataFrame(
        group_rows, columns=SKILL_GROUP_COLUMNS
    )


def run_experiment(
    daily: pd.DataFrame, sites: pd.DataFrame, *, out: str | Path | None = None,
    site_ids: Sequence[str] | None = None, n_jobs: int = 1,
    linear_grid: Sequence[dict[str, Any]] | None = None,
    hgb_grid: Sequence[dict[str, Any]] | None = None,
    bootstrap_reps: int = BOOTSTRAP_REPS, block_days: int = BOOTSTRAP_BLOCK_DAYS, seed: int = SEED,
    app_out: str | Path | None = None,
    pooled: bool = True, pooled_grid: Sequence[dict[str, Any]] | None = None,
    copy_to_app: bool = True,
) -> dict[str, pd.DataFrame]:
    """Tune on validation, refit through validation end, then evaluate test once.

    Each model/feature group gets one shared configuration minimizing the
    equal-weight mean site validation MAE. No fitted transform sees test targets.
    Returned frames are keyed by artifact stem. Writing to PROCESSED also copies
    predictions to APP_DATA; custom output directories are self-contained unless
    app_out is given. Grids and bootstrap reps can be reduced for fast tests.
    """
    started = perf_counter()
    if sites["site_id"].duplicated().any():
        raise ValueError("sites must have unique site_id")
    selected_ids = list(sites["site_id"] if site_ids is None else site_ids)
    if not selected_ids or not set(selected_ids).issubset(set(sites["site_id"])):
        raise ValueError("select at least one known site")
    selected_sites = sites.loc[sites["site_id"].isin(selected_ids)]
    selected_daily = daily.loc[daily["site_id"].isin(selected_ids)]
    if set(selected_daily["site_id"]) != set(selected_ids):
        raise ValueError("selected site missing daily data")
    table = build_features(selected_daily)
    folds = [_prepare_site(rows) for _, rows in table.groupby("site_id", sort=True)]
    if pooled:
        descriptors = selected_sites.set_index("site_id")
        for fold in folds:
            site = descriptors.loc[fold.site_id]
            for frame in (fold.train, fold.val, fold.refit, fold.test):
                frame["lat"] = site["lat"]
                frame["abs_lat"] = abs(site["lat"])
                frame["elevation_m"] = site["elevation_m"]
                # Fixed domain encoding, independent of targets and fold membership.
                frame["koppen_category"] = list(GROUP_QUOTAS).index(site["koppen_group"])
    grids = {"linear": LINEAR_GRID if linear_grid is None else list(linear_grid),
             "hgb": HGB_GRID if hgb_grid is None else list(hgb_grid)}
    if any(not grid for grid in grids.values()):
        raise ValueError("model grids must not be empty")
    predictions = []
    for fold in folds:
        for split in ("val", "test"):
            frame = getattr(fold, split)[["site_id", "date", "y_true"]].copy()
            frame["split"] = split
            frame["persistence"] = persistence(getattr(fold, split))
            frame["climatology"] = getattr(fold, f"climatology_{split}")
            predictions.append(frame)
    prediction_frames = {(str(frame["site_id"].iloc[0]), str(frame["split"].iloc[0])): frame
                         for frame in predictions}
    ablation_rows, selected_params, validation_scores = [], {}, {}
    # Bound BLAS/OpenMP to one thread per fit; site-level joblib owns concurrency.
    with threadpool_limits(limits=1), Parallel(n_jobs=n_jobs, prefer="threads") as parallel:
        for feature_group, columns in FEATURE_GROUPS.items():
            selected_params[feature_group] = {}
            validation_scores[feature_group] = {}
            for model in LEARNED_MODELS:
                grid = grids[model]
                candidates = parallel(delayed(_validation_candidates)(fold, columns, model, grid)
                                      for fold in folds)
                scores = np.array([
                    [metrics(fold.val["y_true"], pred)["mae"] for pred in site_candidates]
                    for fold, site_candidates in zip(folds, candidates, strict=True)
                ])
                best = int(np.argmin(scores.mean(axis=0)))
                params = grid[best]
                selected_params[feature_group][model] = dict(params)
                validation_scores[feature_group][model] = [
                    {"params": dict(params), "mae": float(score)}
                    for params, score in zip(grid, scores.mean(axis=0), strict=True)
                ]
                test_predictions = parallel(
                    delayed(fit_predict)(fold.refit, fold.test, columns, model, params) for fold in folds
                )
                for fold, site_candidates, test_pred in zip(folds, candidates, test_predictions, strict=True):
                    for split, pred in (("val", site_candidates[best]), ("test", test_pred)):
                        ablation_rows.append({
                            "site_id": fold.site_id, "split": split, "model": model,
                            "feature_group": feature_group, "mae": metrics(getattr(fold, split)["y_true"],
                                                                           pred)["mae"],
                        })
                        if feature_group == "full":
                            prediction_frames[fold.site_id, split][model] = pred
            if pooled:
                grid = list(POOLED_GRID if pooled_grid is None else pooled_grid)
                if not grid:
                    raise ValueError("pooled grid must not be empty")
                train, val, refit, test = [pd.concat([getattr(fold, key) for fold in folds])
                                         for key in ("train", "val", "refit", "test")]
                candidates = [pooled_predict(train, val, columns, params) for params in grid]
                scores = [
                    val.assign(pred=pred).groupby("site_id").apply(
                        lambda rows: mae(rows["y_true"], rows["pred"]), include_groups=False
                    ).mean() for pred in candidates
                ]
                best = int(np.argmin(scores))
                selected_params[feature_group]["hgb_pooled"] = dict(grid[best])
                validation_scores[feature_group]["hgb_pooled"] = [
                    {"params": dict(params), "mae": float(score)}
                    for params, score in zip(grid, scores, strict=True)
                ]
                test_pred = pooled_predict(refit, test, columns, grid[best])
                for split, frame, pred in (("val", val, candidates[best]), ("test", test, test_pred)):
                    for site_id, rows in frame.assign(pred=pred).groupby("site_id"):
                        ablation_rows.append({
                            "site_id": site_id, "split": split, "model": "hgb_pooled",
                            "feature_group": feature_group, "mae": mae(rows["y_true"], rows["pred"]),
                        })
                        if feature_group == "full":
                            prediction_frames[site_id, split]["hgb_pooled"] = rows["pred"].to_numpy()
        models = [*MODELS, *(["hgb_pooled"] if pooled else [])]
        prediction_table = pd.concat(predictions, ignore_index=True)[
            [*PREDICTION_COLUMNS, *(["hgb_pooled"] if pooled else [])]
        ]
        metric_rows = [
            {"site_id": site_id, "split": split, "model": model, **metrics(rows["y_true"], rows[model])}
            for (site_id, split), rows in prediction_table.groupby(["site_id", "split"])
            for model in models
        ]
        site_skill, group_skill = _skill_tables(
            prediction_table, selected_sites, bootstrap_reps, block_days, seed
        )
    result = {
        "predictions": prediction_table, "metrics_site": pd.DataFrame(metric_rows, columns=METRIC_COLUMNS),
        "skill_site": site_skill, "skill_group": group_skill,
        "ablation_site": pd.DataFrame(ablation_rows, columns=ABLATION_COLUMNS),
    }
    prediction_table.attrs["selected_params"] = selected_params
    prediction_table.attrs["fit_windows"] = {
        "val": [str(SPLITS["train"][0].date()), str(SPLITS["train"][1].date())],
        "test": [str(SPLITS["train"][0].date()), str(SPLITS["val"][1].date())],
    }
    if out is not None:
        destination = Path(out)
        destination.mkdir(parents=True, exist_ok=True)
        prediction_table.to_parquet(destination / "predictions.parquet", index=False)
        for name, frame in result.items():
            if name != "predictions":
                frame.to_csv(destination / f"{name}.csv", index=False)
        metadata = {
            "selected_params": selected_params, "validation_scores": validation_scores,
            "fit_windows": prediction_table.attrs["fit_windows"], "site_ids": sorted(selected_ids),
            "n_sites": len(folds), "n_jobs": n_jobs, "pooled": pooled,
            "runtime_seconds": perf_counter() - started,
            "bootstrap_reps": bootstrap_reps, "block_days": block_days, "seed": seed,
        }
        (destination / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if copy_to_app and app_out is None and destination.resolve() == PROCESSED.resolve():
            app_out = APP_DATA
    if app_out is not None:
        Path(app_out).mkdir(parents=True, exist_ok=True)
        prediction_table.to_parquet(Path(app_out) / "predictions.parquet", index=False)
    return result
