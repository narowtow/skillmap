"""Training-only site descriptors and held-out geographic skill summaries.

Correlations are exploratory, with percentile bootstrap intervals over sites.
They describe this stratified sample, not a population-weighted global estimate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from skillmap.baselines import Climatology
from skillmap.config import BASELINES, BOOTSTRAP_REPS, GROUP_QUOTAS, N_SITES, PROCESSED, SEED, SPLITS
from skillmap.evaluate import _choose_baselines, split_mask
from skillmap.models import HGB_GRID, LINEAR_GRID, POOLED_GRID

DRIVERS = ["daily_change", "anomaly_acf1", "seasonal_amplitude", "anomaly_sd", "abs_lat", "elevation_m"]


def training_drivers(daily: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Describe training rows only; amplitude is harmonic peak-to-trough / 2."""
    rows = []
    for site_id, frame in daily.groupby("site_id", sort=True):
        frame = frame.loc[split_mask(pd.to_datetime(frame["date"]), "train")].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if not np.isfinite(frame["tmax"]).any():
            continue
        series = frame.set_index("date")["tmax"].sort_index().reindex(pd.date_range(*SPLITS["train"]))
        climate = Climatology().fit(series.index, series)
        anomaly = series - climate.predict(series.index)
        annual = climate.predict(pd.date_range(SPLITS["train"][0], periods=366))
        rows.append({
            "site_id": site_id, "daily_change": series.diff().abs().mean(),
            "anomaly_acf1": anomaly.corr(anomaly.shift(1)),
            "seasonal_amplitude": float(np.ptp(annual) / 2), "anomaly_sd": anomaly.std(ddof=1),
            "n_train": int(series.notna().sum()),
        })
    result = pd.DataFrame(rows).merge(sites, on="site_id", validate="one_to_one")
    result["abs_lat"] = result["lat"].abs()
    return result


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    left, right = rankdata(x, axis=-1), rankdata(y, axis=-1)
    left -= left.mean(axis=-1, keepdims=True)
    right -= right.mean(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (left * right).sum(axis=-1) / np.sqrt((left**2).sum(axis=-1) * (right**2).sum(axis=-1))


def spearman_ci(
    x: Any, y: Any, *, reps: int = BOOTSTRAP_REPS, seed: int = SEED,
) -> dict[str, float | int]:
    """Paired site resampling, with ties ranked anew inside each replicate."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or reps < 1:
        raise ValueError("x/y must be equal-length vectors and reps positive")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    result = {"rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_sites": len(x), "n_valid_reps": 0}
    if len(x) < 3:
        return result
    indices = np.random.default_rng(seed).integers(0, len(x), (reps, len(x)))
    samples = _rank_correlation(x[indices], y[indices])
    samples = samples[np.isfinite(samples)]
    result["rho"] = float(_rank_correlation(x, y))
    result["n_valid_reps"] = len(samples)
    if len(samples):
        result["ci_low"], result["ci_high"] = map(float, np.quantile(samples, [0.025, 0.975]))
    return result


def local_season(month: int, lat: float) -> str:
    """Meteorological local season; equator uses the northern convention."""
    if month not in range(1, 13):
        raise ValueError("month must be 1..12")
    north = ("winter", "spring", "summer", "autumn")
    index = (month % 12) // 3
    return north[(index + (2 if lat < 0 else 0)) % 4]


def seasonal_groups(predictions: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Test month and local-season errors; equal site weights on paired support."""
    choices = _choose_baselines(predictions)
    test = predictions.query("split == 'test'").merge(
        sites[["site_id", "lat", "koppen_group"]], on="site_id", validate="many_to_one"
    )
    test["month"] = pd.to_datetime(test["date"]).dt.month
    test["local_season"] = [local_season(m, lat) for m, lat in zip(test["month"], test["lat"], strict=True)]
    test["best_baseline"] = [getattr(row, choices[row.site_id]) for row in test.itertuples()]
    models = [model for model in ("linear", "hgb", "hgb_pooled") if model in test]
    rows = []
    for period_type in ("month", "local_season"):
        for period, frame in test.groupby(period_type):
            for group in [*GROUP_QUOTAS, "ALL"]:
                subset = frame if group == "ALL" else frame.loc[frame["koppen_group"] == group]
                for model in [*BASELINES, *models]:
                    for reference in [*BASELINES, "best_baseline"]:
                        paired = subset.loc[np.isfinite(subset[model]) & np.isfinite(subset[reference])]
                        losses = paired.assign(
                            model_error=(paired[model] - paired["y_true"]).abs(),
                            ref_error=(paired[reference] - paired["y_true"]).abs(),
                        ).groupby("site_id")[["model_error", "ref_error"]].mean()
                        a, b = losses.mean()
                        rows.append({
                            "period_type": period_type, "period": period, "koppen_group": group,
                            "model": model, "reference": reference, "mae_model": a, "mae_ref": b,
                            "skill": 1 - a / b if b > 0 else np.nan, "n_sites": len(losses),
                            "n_pairs": len(paired),
                        })
    return pd.DataFrame(rows)


def _native(value: Any) -> Any:
    """Make strict JSON, representing undefined numeric estimates as null."""
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_analysis(out: str | Path = PROCESSED) -> dict[str, Any]:
    """Read experiment artifacts, write drivers, seasons, correlations and findings."""
    out = Path(out)
    predictions = pd.read_parquet(out / "predictions.parquet")
    sites = pd.read_csv(out / "sites.csv", keep_default_na=False)
    ids = predictions["site_id"].unique()
    sites = sites.loc[sites["site_id"].isin(ids)]
    daily = pd.read_parquet(out / "daily.parquet")
    drivers = training_drivers(daily.loc[daily["site_id"].isin(ids)], sites)
    skill = pd.read_csv(out / "skill_site.csv").query("split == 'test'")
    metrics = pd.read_csv(out / "metrics_site.csv").query("split == 'test'")
    joined = skill.merge(drivers, on="site_id", validate="many_to_one").merge(
        metrics[["site_id", "model", "mae", "rmse", "bias", "n"]], on=["site_id", "model"],
        validate="many_to_one",
    )
    joined.to_csv(out / "drivers_site.csv", index=False)
    seasonal_groups(predictions, sites).to_csv(out / "seasonal_group.csv", index=False)
    correlations = []
    for (model, reference), frame in joined.groupby(["model", "reference"]):
        for outcome in ("skill", "mae"):
            for driver in DRIVERS:
                correlations.append({"model": model, "reference": reference, "outcome": outcome,
                                     "driver": driver, **spearman_ci(frame[driver], frame[outcome])})
    correlations = pd.DataFrame(correlations)
    correlations.to_csv(out / "driver_correlations.csv", index=False)
    summaries = []
    for (model, reference), frame in skill.groupby(["model", "reference"]):
        quantiles = frame["skill"].quantile([0.25, 0.5, 0.75])
        summaries.append({
            "model": model, "reference": reference, "median_skill": quantiles.loc[0.5],
            "q25_skill": quantiles.loc[0.25], "q75_skill": quantiles.loc[0.75],
            "n_sites": len(frame), "n_significant": int(frame["skill_ci_low"].gt(0).sum()),
            "fraction_significant": frame["skill_ci_low"].gt(0).mean(),
        })
    best = joined.query("reference == 'best_baseline'")
    losses = []
    for row in best.loc[best["skill"] < 0].itertuples():
        reasons = []
        if row.anomaly_acf1 >= drivers["anomaly_acf1"].median():
            reasons.append("above-median anomaly persistence makes yesterday a strong competitor")
        if row.daily_change >= drivers["daily_change"].median():
            reasons.append("above-median day-to-day variability may limit lag-based forecasts")
        if row.anomaly_sd <= drivers["anomaly_sd"].median():
            reasons.append("below-median anomaly variability leaves less improvement over climatology")
        if not reasons:
            reasons.append(
                "drivers do not give a clear explanation; sampling variation or model bias is plausible"
            )
        losses.append({"site_id": row.site_id, "name": row.name, "model": row.model,
                       "skill": row.skill, "ci_low": row.skill_ci_low, "ci_high": row.skill_ci_high,
                       "significant_loss": row.skill_ci_high < 0, "plausible_reasons": reasons})
    groups = pd.read_csv(out / "skill_group.csv").query(
        "split == 'test' and reference == 'best_baseline' and koppen_group != 'ALL' and n_sites > 0"
    )
    extremes = {}
    for model, frame in groups.groupby("model"):
        extremes[model] = {"best": frame.loc[frame["skill"].idxmax()].to_dict(),
                           "worst": frame.loc[frame["skill"].idxmin()].to_dict()}
    ablation = pd.read_csv(out / "ablation_site.csv").query("split == 'test'").pivot(
        index=["site_id", "model"], columns="feature_group", values="mae"
    ).reset_index()
    ablation["delta_mae_full_minus_temp"] = ablation["full"] - ablation["temp_calendar"]
    ablation_summary = ablation.groupby("model")["delta_mae_full_minus_temp"].agg(
        ["mean", "median"]
    ).reset_index().to_dict("records")
    pooled_comparison = None
    if "hgb_pooled" in set(metrics["model"]):
        paired = metrics.pivot(index="site_id", columns="model", values="mae")
        delta = paired["hgb_pooled"] - paired["hgb"]
        pooled_comparison = {"mean_mae_local": paired["hgb"].mean(),
                             "mean_mae_pooled": paired["hgb_pooled"].mean(),
                             "mean_delta_pooled_minus_local": delta.mean(),
                             "median_delta_pooled_minus_local": delta.median(),
                             "fraction_pooled_better": delta.lt(0).mean()}
    experiment_path = out / "experiment.json"
    experiment = json.loads(experiment_path.read_text()) if experiment_path.exists() else {}
    edges = []
    for feature_group, configurations in experiment.get("selected_params", {}).items():
        for model, params in configurations.items():
            grid = {"linear": LINEAR_GRID, "hgb": HGB_GRID, "hgb_pooled": POOLED_GRID}[model]
            for parameter, value in params.items():
                values = sorted({candidate[parameter] for candidate in grid})
                if len(values) > 1 and value in (values[0], values[-1]):
                    edges.append({"feature_group": feature_group, "model": model, "parameter": parameter,
                                  "value": value, "edge": "low" if value == values[0] else "high"})
    top = correlations.query("reference == 'best_baseline' and outcome == 'skill'").copy()
    top["absolute_rho"] = top["rho"].abs()
    top = top.sort_values("absolute_rho", ascending=False).groupby("model").head(4)
    significant_any = best.groupby("site_id")["skill_ci_low"].max().gt(0)
    runtime = experiment.get("runtime_seconds")
    findings = _native({
        "n_sites": len(sites), "site_ids": sorted(ids), "skill_summary": summaries,
        "mae_summary": metrics.groupby("model")["mae"].agg(
            ["mean", "median"]
        ).reset_index().to_dict("records"),
        "group_extremes": extremes, "top_driver_correlations": top.to_dict("records"),
        "losing_sites": losses, "ablation_deltas": ablation_summary, "pooled_vs_local": pooled_comparison,
        "grid_edges": edges, "experiment": experiment,
        "fraction_any_ml_significant": significant_any.mean(),
        "runtime_projection": {
            "target_sites": N_SITES,
            "linear_site_scaling_minutes": runtime * N_SITES / len(sites) / 60 if runtime else None,
            "assumption": "Same hardware and n_jobs; linear scaling in sites is an estimate. "
                          "Pooled fits may scale differently; verify when the full dataset arrives.",
        },
        "caveats": ["Preliminary subset; climate-group coverage may be unbalanced.",
                    "Intervals are conditional on fitted models; no model-selection uncertainty.",
                    "Exploratory driver correlations have no multiple-comparison adjustment.",
                    "Loss explanations are hypotheses, not causal conclusions.",
                    "Local seasons are meteorological; tropical wet/dry seasons are not represented."],
    })
    (out / "findings.json").write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")
    return findings
