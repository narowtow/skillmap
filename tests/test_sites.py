"""Pure selection constraints and optional tiny-raster lookup fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skillmap import config, sites
from skillmap.synthetic import make_sites


def candidates(n=400):
    rng = np.random.default_rng(config.SEED)
    codes = np.tile(np.arange(1, 31), n // 30 + 1)[:n]
    groups = np.array(list("AAABBBBCCCCCCCCCDDDDDDDDDDDDEE"))[codes - 1]
    frame = pd.DataFrame({
        "site_id": [f"gn-{i}" for i in range(n)], "name": [f"Site {i}" for i in range(n)],
        "country_code": [f"X{i % 100}" for i in range(n)],
        "lat": rng.uniform(-70, 80, n), "lon": rng.uniform(-180, 180, n),
        "elevation_m": 0.0, "population": rng.integers(100_000, 1_000_000, n), "site_type": "city",
        "koppen_code": codes, "koppen": [f"{group}{code}" for group, code in zip(groups, codes, strict=True)],
        "koppen_group": groups, "koppen_desc": "fixture", "koppen_fallback": False,
        "geonameid": pd.array(range(n), dtype="Int64"),
    })
    return frame[sites.SITE_COLUMNS]


def test_deterministic_coverage_and_quotas():
    pool = candidates()
    selected = sites.select_sites(pool)
    shuffled = sites.select_sites(pool.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(selected, shuffled)
    assert len(selected) == config.N_SITES
    assert selected.groupby("koppen_group").size().to_dict() == config.GROUP_QUOTAS
    assert selected.groupby("koppen_code").size().min() >= config.MIN_PER_CLASS
    assert selected.country_code.value_counts().max() <= config.MAX_PER_COUNTRY
    assert not selected.site_id.duplicated().any()
    cells = np.rint(selected[["lat", "lon"]].to_numpy() / config.ERA5_GRID_DEG)
    assert len(np.unique(cells, axis=0)) == len(selected)


def test_hard_cell_constraint_and_distance_relaxation():
    pool = pd.concat([make_sites(1)] * 3, ignore_index=True)
    pool["site_id"] = ["gn-a", "gn-b", "gn-c"]
    pool["lat"] = [0.0, 0.01, 0.5]  # first two share one ERA5 cell
    pool["lon"] = 0.0
    pool["population"] = [300, 200, 100]
    selected = sites.select_sites(pool, n_sites=3, quotas={"A": 3})
    assert selected.site_id.tolist() == ["gn-a", "gn-c"]
    assert selected.attrs["relaxations"]["distance"] == 1


def test_dateline_cells_are_unique():
    pool = pd.concat([make_sites(1)] * 2, ignore_index=True)
    pool["site_id"] = ["gn-a", "gn-b"]
    pool["lon"] = [-179.99, 179.99]
    assert len(sites.select_sites(pool, n_sites=2, quotas={"A": 2})) == 1
    assert sites.haversine(0, -179.99, 0, 179.99) < 3


def test_country_cap_relaxes_only_for_class_coverage(monkeypatch):
    monkeypatch.setattr(config, "MAX_PER_COUNTRY", 1)
    pool = make_sites(2).assign(country_code="XX")
    selected = sites.select_sites(pool, n_sites=2, quotas={"A": 1, "B": 1})
    assert len(selected) == 2
    assert selected.attrs["relaxations"]["country"] == 1


def test_redistribute_unfilled_quota():
    pool = candidates(120)
    pool = pool.loc[pool.koppen_group != "E"]
    selected = sites.select_sites(pool, n_sites=60, quotas={"A": 6, "B": 8, "C": 18, "D": 24, "E": 4})
    assert len(selected) == 60
    assert selected.attrs["relaxations"]["redistributed"] == 4


def test_coverage_includes_absent_classes():
    pool = make_sites()
    legend = candidates(30)[["koppen_code", "koppen", "koppen_group", "koppen_desc"]]
    audit = sites.coverage_table(pool, pool, legend)
    assert len(audit) == 30
    assert audit.loc[audit.koppen_code == 2, "n_selected"].item() == 0
    assert audit.loc[audit.koppen_code == 30, "countries"].item() == "GL"


def test_geonames_schema_and_literal_na(tmp_path):
    path = tmp_path / "cities.txt"
    rows = [
        [1, "Wíndhoek", "Windhoek", "", -22.5, 17.1, "P", "PPLC", "NA", "", "", "", "", "",
         400000, 1, 1700, "Africa/Windhoek", "2026-01-01"],
        [2, "Mountain", "Mountain", "", 1, 1, "T", "MT", "PE", "", "", "", "", "",
         0, 1, 5000, "", "2026-01-01"],
    ]
    path.write_text("\n".join("\t".join(map(str, row)) for row in rows), encoding="utf-8")
    result = sites.read_cities(path)
    assert result.site_id.tolist() == ["gn-1"]
    assert result.country_code.item() == "NA" and result.elevation_m.item() == 1700
    assert result.name.item() == "Windhoek"


def test_legend(tmp_path):
    path = tmp_path / "legend.txt"
    path.write_text("header\n" + "\n".join(f" {i}: C{i} Description {i} [1 2 3]" for i in range(1, 31)))
    legend = sites.read_legend(path)
    assert len(legend) == 30 and legend.koppen_desc.iloc[0] == "Description 1"
    path.write_text("1: Af Tropical [1 2 3]")
    with pytest.raises(ValueError):
        sites.read_legend(path)


def test_small_city_supplement_only_for_rare_main_classes(monkeypatch, tmp_path):
    main = candidates(4)
    main["koppen_code"] = [1, 1, 2, 3]
    small = candidates(3)
    small["site_id"] = ["gn-small1", "gn-small2", "gn-small3"]
    small["koppen_code"] = [1, 2, 4]
    station = make_sites().tail(1)
    legend = candidates(30)[["koppen_code", "koppen", "koppen_group", "koppen_desc"]]
    monkeypatch.setattr(sites, "read_legend", lambda path: legend)
    monkeypatch.setattr(sites, "read_cities", lambda path: main if path.name == "cities15000.txt" else small)
    monkeypatch.setattr(sites, "read_stations", lambda path: station)
    monkeypatch.setattr(sites, "sample_koppen", lambda frame, tif: frame)
    # Lookup normally returns only the code; ensure merging attaches the legend once.
    for frame in (main, small, station):
        frame.drop(columns=["koppen", "koppen_group", "koppen_desc"], inplace=True)
    result, _ = sites.load_candidates(geonames_dir=tmp_path)
    assert "gn-small1" not in result.site_id.tolist()  # class 1 already has two large-dump candidates
    assert {"gn-small2", "gn-small3", "st-summit"} <= set(result.site_id)


def test_lookup_and_coastal_fallback(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "classes.tif"
    values = np.zeros((20, 20), dtype="uint8")
    values[5, 5:7] = [1, 2]
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=1,
                       dtype="uint8", crs="EPSG:4326", nodata=0,
                       transform=from_origin(0, 0.2, 0.01, 0.01)) as raster:
        raster.write(values, 1)
    pool = pd.DataFrame({"lat": [0.145, 0.145, 0.005], "lon": [0.055, 0.075, 0.195]})
    result = sites.sample_koppen(pool, path)
    assert result.koppen_code.tolist() == [1, 2]
    assert result.koppen_fallback.tolist() == [False, True]


def test_fallback_across_dateline(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "dateline.tif"
    values = np.zeros((2, 36000), dtype="uint8")
    values[1, 0] = 3
    with rasterio.open(path, "w", driver="GTiff", width=36000, height=2, count=1,
                       dtype="uint8", crs="EPSG:4326", nodata=0,
                       transform=from_origin(-180, 0.02, 0.01, 0.01)) as raster:
        raster.write(values, 1)
    result = sites.sample_koppen(pd.DataFrame({"lat": [0.005], "lon": [179.995]}), path)
    assert result.koppen_code.item() == 3 and result.koppen_fallback.item()
