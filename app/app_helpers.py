"""Pure presentation helpers; no research package or model dependencies."""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_LABELS = {
    "persistence": "Persistence", "climatology": "Climatology", "linear": "Linear (ridge)", "hgb": "HGB",
    "hgb_pooled": "HGB pooled (all sites)",
}
REFERENCE_LABELS = {
    "best_baseline": "Better simple rule (chosen on validation)",
    "persistence": "Persistence", "climatology": "Climatology",
}
COLORS = {"y_true": "#808080", "linear": "#0072B2", "hgb": "#D55E00", "hgb_pooled": "#56B4E9",
          "persistence": "#009E73", "climatology": "#CC79A7"}
GROUP_LABELS = {"A": "A · tropical", "B": "B · arid", "C": "C · temperate",
                "D": "D · continental", "E": "E · polar", "ALL": "All result sites"}
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def model_label(model: str) -> str:
    """Use familiar names, with a readable fallback for future exported models."""
    return MODEL_LABELS.get(model, model.replace("_", " ").capitalize())


def model_order(values) -> list[str]:
    """Keep known model order while discovering optional models from artifacts."""
    present = set(values)
    return [model for model in MODEL_LABELS if model in present] + sorted(present - MODEL_LABELS.keys())


def skill_color_limit(values: pd.Series) -> float:
    """Round the 95th absolute percentile up to 5 percentage points, with a 10% floor."""
    finite = np.abs(values.to_numpy(dtype=float))
    finite = finite[np.isfinite(finite)]
    return max(0.1, float(np.ceil(np.quantile(finite, 0.95) / 0.05) * 0.05)) if len(finite) else 0.1


def map_palette(theme: str | None) -> dict[str, str]:
    """Separate muted land from ocean in either theme."""
    if theme == "dark":
        return {"land": "#344552", "ocean": "#111E2A", "coast": "#738390"}
    return {"land": "#CFD9DE", "ocean": "#EAF1F5", "coast": "#94A4AD"}


def ablation_differences(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare feature groups only on sites that have both finite MAEs."""
    paired = rows.pivot(index=["site_id", "model"], columns="feature_group", values="mae")
    paired = paired.reindex(columns=["temp_calendar", "full"])
    paired["difference"] = paired["temp_calendar"] - paired["full"]
    paired = paired.loc[np.isfinite(paired["difference"])].reset_index()
    return paired[["site_id", "model", "difference"]]


def headline(rows: pd.DataFrame, model: str, reference: str) -> str:
    """Summarize finite point estimates without implying statistical significance."""
    finite = rows.loc[np.isfinite(rows["skill"]) & np.isfinite(rows["delta_mae"])]
    if finite.empty:
        return "Results are pending for this selection."
    n, wins = len(finite), int(finite["delta_mae"].gt(0).sum())
    ref = "the better simple rule" if reference == "best_baseline" else reference
    return (f"{model_label(model)} beats {ref} at {wins} of {n} sites by point estimate; "
            f"median skill {finite['skill'].median():.1%}.")


def interval_text(value: float, low: float, high: float, percent: bool = False) -> str:
    """Format estimates and handle undefined or absent intervals explicitly."""
    fmt = ".1%" if percent else ".2f"
    if not np.isfinite(value):
        return "undefined"
    estimate = format(value, fmt)
    if not np.isfinite(low) or not np.isfinite(high):
        return f"{estimate} (95% CI unavailable)"
    return f"{estimate} (95% CI {format(low, fmt)} to {format(high, fmt)})"


def map_rows(sites: pd.DataFrame, skill: pd.DataFrame) -> pd.DataFrame:
    """Retain pending sites and distinguish uncertain and unavailable intervals."""
    rows = sites.merge(skill, on="site_id", how="left", validate="one_to_one")
    finite = np.isfinite(rows["skill"])
    ci_available = np.isfinite(rows["delta_ci_low"]) & np.isfinite(rows["delta_ci_high"])
    rows["evidence"] = "Pending / skill unavailable"
    rows.loc[finite, "evidence"] = "95% CI unavailable"
    rows.loc[finite & ci_available, "evidence"] = "95% CI excludes zero"
    rows.loc[finite & ci_available & rows["delta_ci_low"].le(0)
             & rows["delta_ci_high"].ge(0), "evidence"] = "95% CI includes zero"
    rows["skill_interval"] = [interval_text(v, lo, hi, True) for v, lo, hi in zip(
        rows["skill"], rows["skill_ci_low"], rows["skill_ci_high"], strict=True)]
    return rows
