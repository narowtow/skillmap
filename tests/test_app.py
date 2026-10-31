"""Offline export checks and Streamlit control coverage on real and tiny data."""

from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.app_helpers import (
    ablation_differences,
    headline,
    map_palette,
    map_rows,
    model_label,
    skill_color_limit,
)
from scripts.export_app_data import export_app_data
from skillmap import config
from skillmap.synthetic import make_sites

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


@pytest.fixture
def tiny_source(tmp_path):
    """Two result sites, one pending, empty groups, missing days and undefined skill."""
    source = tmp_path / "source"
    source.mkdir()
    sites = make_sites(3)
    sites.loc[0, "country_code"] = "NA"  # Namibia must not become a missing country.
    sites.to_csv(source / "sites.csv", index=False)
    predictions = []
    for site_index, site_id in enumerate(sites["site_id"].iloc[:2]):
        for split in ("val", "test"):
            for i, offset in enumerate([0, 1, 3, 31]):
                truth = 10.0 + i
                predictions.append({
                    "site_id": site_id, "date": config.SPLITS[split][0] + pd.Timedelta(days=offset),
                    "split": split, "y_true": truth,
                    "persistence": truth + (2.0 if site_index == 0 else 0.0),
                    "climatology": truth + 3.0,
                    "linear": truth + 1.0,
                    "hgb": np.nan if i == 0 else truth + 0.5,
                    "unused_column": 999.0,
                })
    predictions = pd.DataFrame(predictions)
    predictions.to_parquet(source / "predictions.parquet", index=False)
    metric_rows, skill_rows, ablation_rows = [], [], []
    for (site_id, split), daily in predictions.groupby(["site_id", "split"]):
        for model in config.MODELS:
            error = daily[model] - daily["y_true"]
            metric_rows.append(dict(
                site_id=site_id, split=split, model=model, mae=error.abs().mean(),
                rmse=np.sqrt((error**2).mean()), bias=error.mean(), n=error.notna().sum(),
            ))
        for model in config.MODELS[2:]:
            for reference in [*config.BASELINES, "best_baseline"]:
                ref = "persistence" if reference == "best_baseline" else reference
                mask = np.isfinite(daily[model]) & np.isfinite(daily[ref])
                mae_model = (daily.loc[mask, model] - daily.loc[mask, "y_true"]).abs().mean()
                mae_ref = (daily.loc[mask, ref] - daily.loc[mask, "y_true"]).abs().mean()
                delta = mae_ref - mae_model
                skill = 1 - mae_model / mae_ref if mae_ref else np.nan
                skill_rows.append(dict(
                    site_id=site_id, split=split, model=model, reference=reference, reference_model=ref,
                    mae_model=mae_model, mae_ref=mae_ref, delta_mae=delta,
                    delta_ci_low=delta - 2, delta_ci_high=delta + 2,
                    skill=skill, skill_ci_low=skill - 0.2, skill_ci_high=skill + 0.2,
                ))
            for feature_group in ("temp_calendar", "full"):
                ablation_rows.append(dict(site_id=site_id, split=split, model=model,
                                          feature_group=feature_group, mae=mae_model))
    skills = pd.DataFrame(skill_rows)
    skills.loc[0, ["delta_ci_low", "delta_ci_high"]] = np.nan
    skills.to_csv(source / "skill_site.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(source / "metrics_site.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(source / "ablation_site.csv", index=False)
    groups = skills.drop(columns="site_id").copy()
    groups["koppen_group"] = "ALL"
    groups = groups.groupby(["koppen_group", "split", "model", "reference"], as_index=False).agg({
        **dict.fromkeys(config.SKILL_VALUE_COLUMNS[1:], "mean"), "reference_model": "first",
    })
    groups["n_sites"] = 2
    absent = groups.copy()
    absent["koppen_group"], absent["n_sites"] = "E", 0
    absent[config.SKILL_VALUE_COLUMNS[1:]] = np.nan
    pd.concat([groups, absent]).to_csv(source / "skill_group.csv", index=False)
    benchmark = tmp_path / "benchmark.csv"
    pd.DataFrame({"threads": [1, 2], "requests": [4, 4], "seconds": [2.0, 1.0],
                  "speedup": [1.0, 2.0]}).to_csv(benchmark, index=False)
    return source, benchmark


@pytest.fixture(params=["real", "tiny", "pooled"])
def bundle(request, tmp_path, tiny_source):
    if request.param == "real":
        return ROOT / "app" / "data"
    source, benchmark = tiny_source
    if request.param == "pooled":
        predictions = pd.read_parquet(source / "predictions.parquet")
        predictions["hgb_pooled"] = predictions["hgb"] + 0.1
        predictions.to_parquet(source / "predictions.parquet", index=False)
        # A pooled experiment need not emit a pooled feature ablation.
        for name in ["metrics_site", "skill_site", "skill_group"]:
            path = source / f"{name}.csv"
            rows = pd.read_csv(path)
            pooled = rows.loc[rows["model"].eq("hgb")].copy()
            pooled["model"] = "hgb_pooled"
            pd.concat([rows, pooled]).to_csv(path, index=False)
    destination = tmp_path / "app_data"
    export_app_data(source, destination, benchmark)
    return destination


def run_ok(app):
    """Surface Streamlit exceptions with their messages instead of a bare count."""
    app.run(timeout=30)
    assert not app.exception, [error.message for error in app.exception]
    assert any("Weather data by Open-Meteo.com" in item.value for item in app.markdown)
    assert not any("bundle is incomplete" in item.value for item in app.info)
    return app


def test_export_contract_and_size(tiny_source, tmp_path):
    source, benchmark = tiny_source
    destination = tmp_path / "export"
    before = (source / "predictions.parquet").read_bytes()
    manifest = export_app_data(source, destination, benchmark)
    assert before == (source / "predictions.parquet").read_bytes()
    assert manifest["n_sites_with_results"] == 2
    assert manifest["n_sites"] == 3
    assert manifest["models"] == config.MODELS
    assert manifest["splits"]["test"][0] == str(config.SPLITS["test"][0].date())
    assert manifest["data_period"] == [str(config.START.date()), str(config.END.date())]
    assert manifest["bundle_bytes"] == sum(p.stat().st_size for p in destination.iterdir())
    assert json.loads((destination / "manifest.json").read_text()) == manifest
    sites = pd.read_parquet(destination / "sites.parquet")
    assert sites["results_available"].tolist() == [True, True, False]
    assert sites.iloc[0]["country_code"] == "NA"
    predictions = pd.read_parquet(destination / "predictions.parquet")
    assert list(predictions.columns) == config.PREDICTION_COLUMNS
    assert all(predictions[column].dtype == np.float32 for column in ["y_true", *config.MODELS])
    skills = pd.read_parquet(destination / "skill_site.parquet")
    assert skills.loc[skills["model"].eq("linear"), "n"].eq(4).all()
    assert skills.loc[skills["model"].eq("hgb"), "n"].eq(3).all()
    assert len(pd.read_parquet(destination / "benchmark_concurrency.parquet")) == 2


def test_preliminary_bundle_is_small():
    manifest = json.loads((ROOT / "app/data/manifest.json").read_text())
    measured = sum(path.stat().st_size for path in (ROOT / "app/data").iterdir() if path.is_file())
    assert measured == manifest["bundle_bytes"]
    assert measured < 25_000_000
    if manifest["n_sites_with_results"]:
        assert measured * config.N_SITES / manifest["n_sites_with_results"] < 25_000_000


def test_headline_and_uncertainty_categories():
    rows = pd.DataFrame({"skill": [0.2, -0.1, 0, np.nan], "delta_mae": [1.0, -0.5, 0, 1]})
    assert headline(rows, "linear", "best_baseline") == (
        "Linear (ridge) beats the better simple rule at 1 of 3 sites by point estimate; median skill 0.0%."
    )
    assert "pending" in headline(rows.iloc[3:], "hgb", "persistence")
    skill = pd.DataFrame({
        "site_id": ["a", "b", "c"], "skill": [0.8, -0.8, 0.1],
        "skill_ci_low": [0.1, -0.9, np.nan], "skill_ci_high": [0.9, 0.1, np.nan],
        "delta_ci_low": [0.1, -0.9, np.nan], "delta_ci_high": [0.9, 0.1, np.nan],
    })
    mapped = map_rows(pd.DataFrame({"site_id": ["a", "b", "c", "pending"]}), skill)
    assert mapped["evidence"].tolist() == [
        "95% CI excludes zero", "95% CI includes zero", "95% CI unavailable", "Pending / skill unavailable",
    ]
    assert mapped.iloc[0]["skill"] == 0.8  # No clipping outside color encoding.
    assert "80.0%" in mapped.iloc[0]["skill_interval"]


def test_map_control_combinations(bundle, monkeypatch):
    monkeypatch.setenv("SKILLMAP_APP_DATA", str(bundle))
    app = run_ok(AppTest.from_file(str(APP)))
    manifest = json.loads((bundle / "manifest.json").read_text())
    models = [model for model in manifest["models"] if model not in config.BASELINES]
    skills = pd.read_parquet(bundle / "skill_site.parquet")
    for model, reference, split, raw in itertools.product(
        models, ["best_baseline", *config.BASELINES], ["test", "val"], [False, True],
    ):
        app.selectbox(key="model").set_value(model)
        app.selectbox(key="reference").set_value(reference)
        app.selectbox(key="split").set_value(split)
        app.toggle(key="raw_mae").set_value(raw)
        run_ok(app)
        figure = json.loads(app.get("plotly_chart")[0].proto.spec)
        assert figure["layout"]["geo"]["projection"]["type"] == "natural earth"
        geo = figure["layout"]["geo"]
        assert all(geo[layer] for layer in ["showland", "showocean", "showcountries", "showcoastlines"])
        assert not geo["showframe"]
        assert geo["landcolor"] != geo["oceancolor"]
        assert geo["bgcolor"] == figure["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"
        assert geo["lonaxis"]["range"] == [-180, 180]
        assert geo["lataxis"]["range"] == [-90, 90]
        assert geo["projection"]["scale"] == 1
        assert figure["layout"]["margin"] == {"l": 0, "r": 0, "t": 0, "b": 45}
        config_json = json.loads(app.get("plotly_chart")[0].proto.config)
        assert config_json["scrollZoom"] is False
        assert config_json["displayModeBar"] is True
        assert figure["layout"]["height"] == 600
        assert "width" not in figure["layout"]
        assert not figure["layout"]["geo"]["lataxis"]["showgrid"]
        assert not figure["layout"]["geo"]["lonaxis"]["showgrid"]
        assert not any(trace["name"] == "Land outline" for trace in figure["data"])
        for trace in figure["data"]:
            if trace["name"] != "Pending / unavailable":
                assert trace["marker"]["line"] == {"width": 0.7, "color": "#263442"}
        pending = [trace for trace in figure["data"] if trace["name"] == "Pending / unavailable"]
        if pending:
            assert pending[0]["marker"]["symbol"] == "circle"
            assert pending[0]["marker"]["size"] <= 3
        if not raw:
            selected = skills.loc[skills["split"].eq(split) & skills["reference"].eq(reference)]
            limit = skill_color_limit(selected["skill"])
            assert figure["layout"]["coloraxis"]["cmin"] == -limit
            assert figure["layout"]["coloraxis"]["cmax"] == limit
        assert bool(app.warning) == (split == "val")


@pytest.mark.parametrize(("base", "browser", "theme"), [
    (None, "light", "light"), (None, "dark", "dark"), ("light", "dark", "light"),
])
def test_map_theme_backgrounds(base, browser, theme, monkeypatch):
    import streamlit
    from streamlit.runtime.context import ContextProxy, StreamlitTheme

    get_option = streamlit.get_option
    monkeypatch.setattr(streamlit, "get_option",
                        lambda key: base if key == "theme.base" else get_option(key))
    monkeypatch.setenv("SKILLMAP_APP_DATA", str(ROOT / "app/data"))
    monkeypatch.setattr(ContextProxy, "theme", property(lambda _: StreamlitTheme({"type": browser})))
    app = run_ok(AppTest.from_file(str(APP)))
    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    geo, palette = figure["layout"]["geo"], map_palette(theme)
    assert geo["landcolor"] == palette["land"]
    assert geo["oceancolor"] == palette["ocean"]
    assert geo["countrycolor"] == geo["coastlinecolor"] == palette["coast"]
    assert geo["bgcolor"] == figure["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"


def test_other_pages_and_city_controls(bundle, monkeypatch):
    monkeypatch.setenv("SKILLMAP_APP_DATA", str(bundle))
    app = run_ok(AppTest.from_file(str(APP)))
    app.radio(key="page").set_value("City explorer")
    run_ok(app)
    choices = pd.read_parquet(bundle / "sites.parquet")
    models = pd.read_parquet(bundle / "metrics_site.parquet")["model"].unique()
    for site_id in choices.loc[choices["results_available"], "site_id"]:
        app.selectbox(key="city").set_value(site_id)
        run_ok(app)
        assert len(app.dataframe[0].value) == len(models)
        assert set(app.dataframe[0].value["Model"]) == {model_label(model) for model in models}
    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert pd.Timestamp(figure["layout"]["xaxis"]["range"][1]) - pd.Timestamp(
        figure["layout"]["xaxis"]["range"][0]) == pd.Timedelta(days=119)
    assert figure["layout"]["xaxis"]["rangeslider"]["visible"]
    assert [button["label"] for button in figure["layout"]["xaxis"]["rangeselector"]["buttons"]] == [
        "3M", "1Y", "All",
    ]
    monthly = json.loads(app.get("plotly_chart")[1].proto.spec)
    assert monthly["layout"]["xaxis"]["ticktext"][0] == "Jan"
    assert monthly["layout"]["xaxis"]["ticktext"][-1] == "Dec"
    assert "background-color" in app.dataframe[0].proto.arrow_data.styler.styles
    learned = [model for model in models if model not in config.BASELINES]
    for lines, persistence, climatology in itertools.product(
        [list(subset) for length in range(len(learned) + 1)
         for subset in itertools.combinations(learned, length)], [False, True], [False, True],
    ):
        app.multiselect(key="city_models").set_value(lines)
        app.checkbox[0].set_value(persistence)
        app.checkbox[1].set_value(climatology)
        run_ok(app)
    app.radio(key="page").set_value("Climate zones")
    run_ok(app)
    for reference, split in itertools.product(["best_baseline", *config.BASELINES], ["test", "val"]):
        app.selectbox(key="reference").set_value(reference)
        app.selectbox(key="split").set_value(split)
        run_ok(app)
        plots = [json.loads(item.proto.spec) for item in app.get("plotly_chart")]
        group = plots[0]
        assert group["layout"]["xaxis"]["title"]["text"] == "Climate group"
        assert group["data"][0]["x"][0] in {
            "A · tropical", "B · arid", "C · temperate", "D · continental", "E · polar", "All result sites",
        }
        assert plots[-2]["data"][0]["type"] == "bar"
        assert plots[-1]["data"][0]["type"] == "box"
        assert plots[-1]["layout"]["yaxis"]["title"]["text"] == "MAE reduction from extra variables (°C)"
        assert "model" not in plots[-1]["layout"]["xaxis"]["title"]["text"]
    app.radio(key="page").set_value("Method & limits")
    run_ok(app)
    assert len(app.dataframe) == 2


def test_color_scale_and_paired_ablation():
    assert skill_color_limit(pd.Series([0.04, 0.12, 0.24, np.nan])) == 0.25
    assert skill_color_limit(pd.Series([0.01, -0.02])) == 0.1
    assert skill_color_limit(pd.Series([np.nan])) == 0.1
    assert skill_color_limit(pd.Series([-0.35])) == pytest.approx(0.35)
    for theme in ["light", "dark"]:
        palette = map_palette(theme)
        assert palette["land"] != palette["ocean"]
    ablation = pd.DataFrame({
        "site_id": ["a", "a", "b", "b", "unpaired"], "model": ["hgb"] * 5,
        "feature_group": ["full", "temp_calendar", "full", "temp_calendar", "full"],
        "mae": [1.0, 1.04, 2.0, 1.98, 100.0],
    })
    difference = ablation_differences(ablation)
    assert difference["difference"].tolist() == pytest.approx([0.04, -0.02])
    assert difference["difference"].mean() == pytest.approx(0.01)


def test_optional_pooled_export(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    predictions = pd.read_parquet(bundle / "predictions.parquet")
    metrics = pd.read_parquet(bundle / "metrics_site.parquet")
    present = "hgb_pooled" in metrics["model"].values
    assert ("hgb_pooled" in predictions) == present
    assert ("hgb_pooled" in manifest["models"]) == present
    if present:
        assert predictions["hgb_pooled"].dtype == np.float32
        skills = pd.read_parquet(bundle / "skill_site.parquet")
        assert skills.loc[skills["model"].eq("hgb_pooled"), "n"].gt(0).all()


def test_refresh_picks_up_new_result_site(tiny_source, tmp_path, monkeypatch):
    source, benchmark = tiny_source
    destination = tmp_path / "refresh"
    export_app_data(source, destination, benchmark)
    monkeypatch.setenv("SKILLMAP_APP_DATA", str(destination))
    app = run_ok(AppTest.from_file(str(APP)))
    assert any("2 of 3 sites" in item.value for item in app.caption)
    sites = pd.read_csv(source / "sites.csv")
    first_id, added_id = sites["site_id"].iloc[[0, 2]]
    for name in ["metrics_site", "skill_site", "ablation_site", "predictions"]:
        path = source / f"{name}.{'parquet' if name == 'predictions' else 'csv'}"
        original = pd.read_parquet(path) if name == "predictions" else pd.read_csv(path)
        added = original.loc[original["site_id"].eq(first_id)].copy()
        added["site_id"] = added_id
        updated = pd.concat([original, added], ignore_index=True)
        if name == "predictions":
            updated.to_parquet(path, index=False)
        else:
            updated.to_csv(path, index=False)
    manifest = export_app_data(source, destination, benchmark)
    assert manifest["n_sites_with_results"] == 3
    run_ok(app)
    assert any("3 of 3 sites" in item.value for item in app.caption)
    assert not any("Preliminary" in item.value for item in app.caption)


def test_all_pending_and_incomplete_bundle(tiny_source, tmp_path, monkeypatch):
    source, benchmark = tiny_source
    for name in ["metrics_site", "skill_site", "skill_group", "ablation_site"]:
        data = pd.read_csv(source / f"{name}.csv").iloc[:0]
        data.to_csv(source / f"{name}.csv", index=False)
    pd.read_parquet(source / "predictions.parquet").iloc[:0].to_parquet(source / "predictions.parquet")
    destination = tmp_path / "empty"
    export_app_data(source, destination, benchmark)
    monkeypatch.setenv("SKILLMAP_APP_DATA", str(destination))
    app = run_ok(AppTest.from_file(str(APP)))
    for page in ["City explorer", "Climate zones", "Method & limits"]:
        app.radio(key="page").set_value(page)
        run_ok(app)
    monkeypatch.setenv("SKILLMAP_APP_DATA", str(tmp_path / "missing"))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert "bundle is incomplete" in app.info[0].value


def test_app_imports_only_runtime_dependencies():
    allowed = {"__future__", "json", "os", "sys", "pathlib", "numpy", "pandas", "plotly",
               "streamlit", "app_helpers"}
    for path in (ROOT / "app").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in allowed for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module.split(".")[0] in allowed
