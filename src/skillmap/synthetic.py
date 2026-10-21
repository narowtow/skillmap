"""Synthetic sites and daily weather that follow the data contract.

Used by tests and by code that is built before the real download finishes. Each site has a
seasonal cycle plus AR(1) anomalies, so persistence, climatology and lagged models all have
something to find. Not a weather model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skillmap.config import END, START

_SITES = [
    # site_id, name, country, lat, lon, elev, pop, type, code, koppen, group
    ("gn-0001", "Wetville", "XA", 1.3, 103.8, 15, 5_000_000, "city", 1, "Af", "A"),
    ("gn-0002", "Dryburg", "XB", 24.5, 54.4, 10, 1_500_000, "city", 4, "BWh", "B"),
    ("gn-0003", "Mildton", "XC", 51.5, -0.1, 20, 8_000_000, "city", 15, "Cfb", "C"),
    ("gn-0004", "Frostsk", "XD", 62.0, 129.7, 120, 300_000, "city", 28, "Dfd", "D"),
    ("st-summit", "Summit Station", "GL", 72.58, -38.46, 3216, 0, "station", 30, "EF", "E"),
]

# amplitude of seasonal cycle, AR(1) phi, anomaly sd, mean Tmax
_PARAMS = {
    "A": (0.8, 0.5, 0.8, 31.0),
    "B": (8.0, 0.7, 1.5, 33.0),
    "C": (7.0, 0.75, 2.5, 15.0),
    "D": (22.0, 0.8, 4.0, -2.0),
    "E": (14.0, 0.85, 5.0, -28.0),
}


def make_sites(n: int | None = None) -> pd.DataFrame:
    cols = [
        "site_id", "name", "country_code", "lat", "lon", "elevation_m", "population",
        "site_type", "koppen_code", "koppen", "koppen_group",
    ]
    df = pd.DataFrame(_SITES[:n] if n else _SITES, columns=cols)
    df["koppen_desc"] = df["koppen"]
    df["koppen_fallback"] = False
    df["geonameid"] = pd.array([1, 2, 3, 4, None][: len(df)], dtype="Int64")
    return df


def make_daily(
    sites: pd.DataFrame | None = None,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
    seed: int = 0,
    missing_frac: float = 0.0,
) -> pd.DataFrame:
    """Long-format daily table: site_id, date, tmax, tmin, rh_mean, msl_mean, wind_mean."""
    sites = make_sites() if sites is None else sites
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()
    frames = []
    for row in sites.itertuples():
        amp, phi, sd, mean = _PARAMS[row.koppen_group]
        hemi = 1.0 if row.lat >= 0 else -1.0
        season = mean + hemi * amp * np.sin(2 * np.pi * (doy - 105) / 365.25)
        pressure = np.zeros(len(dates))
        anom = np.zeros(len(dates))
        eps = rng.normal(0, sd, len(dates))
        p_eps = rng.normal(0, 6, len(dates))
        for i in range(1, len(dates)):
            pressure[i] = 0.8 * pressure[i - 1] + p_eps[i]
            # falling pressure yesterday nudges today's anomaly down: a signal only lagged models see
            anom[i] = phi * anom[i - 1] + 0.08 * (pressure[i - 1] - pressure[max(i - 2, 0)]) + eps[i]
        tmax = season + anom
        frame = pd.DataFrame(
            {
                "site_id": row.site_id,
                "date": dates,
                "tmax": tmax,
                "tmin": tmax - rng.uniform(5, 12, len(dates)),
                "rh_mean": np.clip(70 - 0.5 * anom + rng.normal(0, 8, len(dates)), 5, 100),
                "msl_mean": 1013 + pressure,
                "wind_mean": np.abs(3 + rng.normal(0, 1.5, len(dates))),
            }
        )
        frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    if missing_frac > 0:
        cols = ["tmax", "tmin", "rh_mean", "msl_mean", "wind_mean"]
        mask = rng.random((len(daily), len(cols))) < missing_frac
        daily[cols] = daily[cols].mask(mask)
    return daily
