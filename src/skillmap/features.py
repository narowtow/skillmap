"""Calendar-aligned, strictly lagged per-site weather predictors."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from skillmap.baselines import Climatology, annual_phase
from skillmap.config import TARGET, WEATHER_COLUMNS

TEMP_CALENDAR = (
    [f"{variable}_lag{lag}" for variable in ("tmax", "tmin") for lag in range(1, 8)]
    + ["dtr_lag1", "doy_sin", "doy_cos", "tmax_anomaly_lag1"]
)
FULL = TEMP_CALENDAR + [
    f"{variable}_lag{lag}"
    for variable in WEATHER_COLUMNS
    if variable not in ("tmax", "tmin")
    for lag in range(1, 8)
] + ["pressure_tendency"]
FEATURE_GROUPS = {"temp_calendar": TEMP_CALENDAR, "full": FULL}


def add_climatology_anomaly(table: pd.DataFrame, climatology: Climatology) -> pd.DataFrame:
    """Apply an already fitted site's climatology to yesterday's temperature."""
    result = table.copy()
    yesterday = pd.DatetimeIndex(result["date"]) - pd.Timedelta(days=1)
    result["tmax_anomaly_lag1"] = result["tmax_lag1"] - climatology.predict(yesterday)
    return result


def build_features(
    daily: pd.DataFrame, climatologies: Mapping[str, Climatology] | None = None,
) -> pd.DataFrame:
    """Build target rows on complete site calendars; never fit a climatology here.

    Without fitted climatologies the anomaly is NaN. The experiment adds it
    separately for each fold, avoiding any implicit fit on future targets.
    """
    required = ["site_id", "date", *WEATHER_COLUMNS]
    missing = set(required) - set(daily.columns)
    if missing:
        raise ValueError(f"missing daily columns: {sorted(missing)}")
    daily = daily[required].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    if daily.duplicated(["site_id", "date"]).any():
        raise ValueError("duplicate site/date rows")
    frames = []
    for site_id, site in daily.groupby("site_id", sort=True):
        site = site.set_index("date").sort_index()
        site = site.reindex(pd.date_range(site.index.min(), site.index.max(), freq="D"))
        result = pd.DataFrame({"site_id": site_id, "date": site.index, "y_true": site[TARGET].values})
        for variable in WEATHER_COLUMNS:
            for lag in range(1, 8):
                result[f"{variable}_lag{lag}"] = site[variable].shift(lag).to_numpy()
        result["pressure_tendency"] = result["msl_mean_lag1"] - result["msl_mean_lag2"]
        result["dtr_lag1"] = result["tmax_lag1"] - result["tmin_lag1"]
        phase = annual_phase(result["date"])
        result["doy_sin"], result["doy_cos"] = np.sin(phase), np.cos(phase)
        result["tmax_anomaly_lag1"] = np.nan
        if climatologies is not None:
            result = add_climatology_anomaly(result, climatologies[site_id])
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["site_id", "date", "y_true", *FULL]
    )
