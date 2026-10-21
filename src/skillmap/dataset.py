"""Build the contract daily table from validated local site-year cache files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from skillmap import config
from skillmap.fetch import cache_path, plan_chunks, read_cache

LOG = logging.getLogger(__name__)
META_COLUMNS = ["site_id", "grid_lat", "grid_lon", "grid_elevation", "timezone", "utc_offset_seconds"]


def build_dataset(
    sites: pd.DataFrame,
    cache_dir: Path = config.OPENMETEO_CACHE,
    *,
    start: pd.Timestamp = config.START,
    end: pd.Timestamp = config.END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep partial sites on a full date index; skip sites with no valid chunks."""
    frames, metadata = [], []
    dates = pd.date_range(start, end, name="date")
    for row in sites.itertuples():
        parts, meta = [], None
        chunks = plan_chunks(pd.DataFrame([row._asdict()]), start=start, end=end)
        for chunk in chunks:
            payload = read_cache(cache_path(cache_dir, chunk), chunk)
            if payload is None:
                continue
            daily = payload["daily"]
            part = pd.DataFrame({column: daily[variable] for variable, column in config.VARIABLES.items()},
                                index=pd.to_datetime(daily["time"]))
            parts.append(part.astype("float64"))
            current = {"site_id": row.site_id, "grid_lat": payload.get("latitude"),
                       "grid_lon": payload.get("longitude"), "grid_elevation": payload.get("elevation"),
                       "timezone": payload.get("timezone"),
                       "utc_offset_seconds": payload.get("utc_offset_seconds")}
            if meta is None:
                meta = current
            elif current != meta:
                LOG.warning("%s: API metadata changes across years; using earliest valid chunk", row.site_id)
        if not parts:
            LOG.warning("%s: no complete chunks; skipping site", row.site_id)
            continue
        if len(parts) < len(chunks):
            LOG.warning("%s: partially downloaded (%d/%d complete chunks)",
                        row.site_id, len(parts), len(chunks))
        frame = pd.concat(parts).reindex(dates)
        frame.insert(0, "site_id", row.site_id)
        frames.append(frame.reset_index())
        metadata.append(meta)
    columns = ["site_id", "date", *config.WEATHER_COLUMNS]
    daily_table = pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)
    daily_table["date"] = pd.to_datetime(daily_table.date).astype("datetime64[ns]")
    return daily_table, pd.DataFrame(metadata, columns=META_COLUMNS)


def write_dataset(
    sites: pd.DataFrame,
    cache_dir: Path = config.OPENMETEO_CACHE,
    daily_path: Path = config.DAILY_PARQUET,
    meta_path: Path = config.API_META_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and persist the two processed data-contract artifacts."""
    daily, meta = build_dataset(sites, cache_dir)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(daily_path, index=False)
    meta.to_csv(meta_path, index=False)
    return daily, meta
