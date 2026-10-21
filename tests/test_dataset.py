"""Cache fixtures preserve calendar dates, missing values, and metadata."""

import json

import pandas as pd
from conftest import payload

from skillmap import config
from skillmap.dataset import build_dataset, write_dataset
from skillmap.fetch import cache_path, plan_chunks
from skillmap.synthetic import make_sites


def test_partial_and_absent_sites(tmp_path, caplog):
    sites = make_sites(2)
    chunk = plan_chunks(sites)[0]
    body = payload(chunk.start, chunk.end)
    body["daily"][next(iter(config.VARIABLES))][0] = None
    path = cache_path(tmp_path, chunk)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(body))
    # A malformed second year is treated as missing, not a partial valid chunk.
    (path.parent / "2016.json").write_text('{"daily":{"time":[]}}')
    daily, meta = build_dataset(sites, tmp_path)
    dates = pd.date_range(config.START, config.END)
    assert len(daily) == len(dates) and daily.site_id.nunique() == 1
    assert str(daily.date.dtype) == "datetime64[ns]"
    assert daily.date.tolist() == dates.tolist()
    assert daily.tmax.iloc[0:1].isna().all()
    assert daily.loc[daily.date > chunk.end, config.WEATHER_COLUMNS].isna().all().all()
    assert meta.iloc[0].to_dict() == {
        "site_id": chunk.site_id, "grid_lat": 1.0, "grid_lon": 2.0, "grid_elevation": 3.0,
        "timezone": "Test/Local", "utc_offset_seconds": 3600,
    }
    assert "partially downloaded" in caplog.text and "no complete chunks" in caplog.text
    output = tmp_path / "processed"
    expected, _ = write_dataset(sites, tmp_path, output / "daily.parquet", output / "api_meta.csv")
    pd.testing.assert_frame_equal(pd.read_parquet(output / "daily.parquet"), expected)


def test_leap_year_and_empty_dataset(tmp_path):
    sites = make_sites(1)
    empty, meta = build_dataset(sites, tmp_path)
    assert empty.empty and meta.empty
    chunk = [c for c in plan_chunks(sites) if c.year == 2020][0]
    path = cache_path(tmp_path, chunk)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload(chunk.start, chunk.end)))
    daily, _ = build_dataset(sites, tmp_path)
    assert len(daily.loc[daily.date.dt.year == 2020]) == 366
    assert daily.loc[daily.date == pd.Timestamp("2020-02-29"), "tmax"].notna().all()
