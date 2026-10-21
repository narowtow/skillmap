"""Shared constants: paths, study period, splits, variables, sampling quotas.

Everything that two modules must agree on lives here (see contracts/README.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CONFIG = ROOT / "config"
APP_DATA = ROOT / "app" / "data"
FIGURES = ROOT / "reports" / "figures"

GEONAMES_DIR = RAW / "geonames"
KOPPEN_TIF = RAW / "koppen" / "1991_2020" / "koppen_geiger_0p00833333.tif"
KOPPEN_LEGEND = RAW / "koppen" / "legend.txt"
STATIONS_CSV = CONFIG / "research_stations.csv"
OPENMETEO_CACHE = RAW / "openmeteo"

SITES_CSV = PROCESSED / "sites.csv"
COVERAGE_CSV = PROCESSED / "koppen_coverage.csv"
DAILY_PARQUET = PROCESSED / "daily.parquet"
API_META_CSV = PROCESSED / "api_meta.csv"

# --- Study period and chronological splits (by target date, shared by all sites) ---
START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2025-12-31")
SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (pd.Timestamp("2015-01-01"), pd.Timestamp("2021-12-31")),
    "val": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
    "test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")),
}

# --- Open-Meteo archive request ---
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_MODEL = "era5"
# API daily variable -> column name in daily.parquet
VARIABLES: dict[str, str] = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "relative_humidity_2m_mean": "rh_mean",
    "pressure_msl_mean": "msl_mean",
    "wind_speed_10m_mean": "wind_mean",
}
TARGET = "tmax"
WEATHER_COLUMNS = list(VARIABLES.values())

# Free tier: < 600/min, < 5,000/hour, < 10,000/day weighted calls. We stay well under.
RATE_LIMITS = {"minute": 300.0, "hour": 4000.0, "day": 8000.0}

# --- Site sampling ---
N_SITES = 200
GROUP_QUOTAS = {"A": 35, "B": 40, "C": 60, "D": 50, "E": 15}
MIN_PER_CLASS = 2
MIN_SEPARATION_KM = 250.0
ERA5_GRID_DEG = 0.25
MAX_PER_COUNTRY = 12
SEED = 42

# --- Evaluation ---
BOOTSTRAP_BLOCK_DAYS = 14
BOOTSTRAP_REPS = 1000
MODELS = ["persistence", "climatology", "linear", "hgb"]
BASELINES = ["persistence", "climatology"]
