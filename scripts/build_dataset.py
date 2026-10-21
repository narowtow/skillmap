"""Convert local archive cache files into daily.parquet and api_meta.csv."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmap import config
from skillmap.dataset import write_dataset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    daily, meta = write_dataset(pd.read_csv(config.SITES_CSV, keep_default_na=False))
    print(f"Wrote {len(daily):,} daily rows for {len(meta)} sites")
    print(daily[config.WEATHER_COLUMNS].isna().sum().to_string())


if __name__ == "__main__":
    main()
