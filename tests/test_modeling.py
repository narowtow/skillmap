"""Leakage, chronological fitting, paired inference and artifact contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import skillmap.evaluate as evaluation
from skillmap.baselines import Climatology, harmonic_design
from skillmap.bootstrap import moving_block_weights, paired_bootstrap
from skillmap.config import (
    ABLATION_COLUMNS,
    END,
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    SKILL_GROUP_COLUMNS,
    SKILL_SITE_COLUMNS,
    SPLITS,
    START,
    WEATHER_COLUMNS,
)
from skillmap.evaluate import _prepare_site, bias, mae, metrics, rmse, run_experiment, split_mask
from skillmap.features import FEATURE_GROUPS, build_features
from skillmap.models import make_model
from skillmap.synthetic import make_daily, make_sites


@pytest.mark.parametrize("variable", WEATHER_COLUMNS)
def test_predictors_use_only_previous_calendar_days(variable):
    sites = make_sites(2)
    daily = make_daily(sites, start=START, end=START + pd.Timedelta(days=30))
    climates = {site_id: Climatology().fit([START], [20]) for site_id in sites["site_id"]}
    original = build_features(daily, climates)
    changed = daily.copy()
    date = START + pd.Timedelta(days=12)
    changed.loc[(changed["site_id"] == sites.iloc[0].site_id) & (changed["date"] == date), variable] += 100
    updated = build_features(changed, climates)
    columns = FEATURE_GROUPS["full"]
    protected = (original["date"] <= date) | (original["site_id"] != sites.iloc[0].site_id)
    pd.testing.assert_frame_equal(original.loc[protected, columns], updated.loc[protected, columns])
    later = (original["date"] > date) & (original["site_id"] == sites.iloc[0].site_id)
    assert not original.loc[later, columns].equals(updated.loc[later, columns])


def test_calendar_gaps_and_site_boundaries():
    daily = make_daily(make_sites(2), start=START, end=START + pd.Timedelta(days=10))
    missing_date = START + pd.Timedelta(days=5)
    daily = daily.loc[~((daily["site_id"] == "gn-0001") & (daily["date"] == missing_date))]
    features = build_features(daily)
    site = features.loc[features["site_id"] == "gn-0001"].set_index("date")
    assert pd.isna(site.loc[missing_date + pd.Timedelta(days=1), "tmax_lag1"])
    expected = daily.loc[(daily["site_id"] == "gn-0001") &
                         (daily["date"] == missing_date - pd.Timedelta(days=1)), "tmax"].iloc[0]
    assert site.loc[missing_date + pd.Timedelta(days=1), "tmax_lag2"] == expected
    assert pd.isna(site.loc[missing_date, "y_true"])
    first = features.groupby("site_id").head(1)
    assert first[[f"tmax_lag{k}" for k in range(1, 8)]].isna().all().all()
    assert set(FEATURE_GROUPS["temp_calendar"]) < set(FEATURE_GROUPS["full"])


def test_climatology_harmonics_leap_day_and_fold_isolation():
    daily = make_daily(make_sites(1))
    table = build_features(daily)
    before = _prepare_site(table)
    changed = table.copy()
    changed.loc[changed["date"] > SPLITS["train"][1], "y_true"] += 1000
    after = _prepare_site(changed)
    np.testing.assert_array_equal(before.climatology_val, after.climatology_val)
    pd.testing.assert_frame_equal(before.train, after.train)
    pd.testing.assert_series_equal(before.val["tmax_anomaly_lag1"], after.val["tmax_anomaly_lag1"])
    test_only = table.copy()
    test_only.loc[split_mask(test_only["date"], "test"), "y_true"] -= 1000
    np.testing.assert_array_equal(before.climatology_test, _prepare_site(test_only).climatology_test)
    dates = pd.date_range(START, END)
    values = harmonic_design(dates) @ np.arange(7)
    climate = Climatology().fit(dates, values)
    np.testing.assert_allclose(climate.predict(dates), values, atol=1e-12)
    assert np.isfinite(climate.predict([pd.Timestamp("2024-02-29")])).all()


def test_split_ranges_are_exhaustive_and_disjoint():
    dates = pd.Series(pd.date_range(START, END))
    masks = [split_mask(dates, split) for split in SPLITS]
    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(len(dates)))
    for split in SPLITS:
        selected = dates[split_mask(dates, split)]
        assert (selected.min(), selected.max()) == SPLITS[split]


def test_metrics_hand_calculation_and_missing_pairs():
    actual = metrics([1, 2, 3, np.nan, 5], [2, 0, 6, 5, np.nan])
    assert actual == {"mae": 2.0, "rmse": pytest.approx(np.sqrt(14 / 3)), "bias": 2 / 3, "n": 3}
    assert metrics([np.nan], [1])["n"] == 0
    assert mae([1, 2], [3, 2]) == 1
    assert rmse([1, 2], [3, 2]) == pytest.approx(np.sqrt(2))
    assert bias([1, 2], [3, 2]) == 1
    with pytest.raises(ValueError, match="same shape"):
        metrics([1], [1, 2])


def test_linear_transforms_fit_on_training_data():
    estimator = make_model("linear", {"alpha": 1.0})
    train = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, np.nan]])
    estimator.fit(train, [1, 2, 3])
    np.testing.assert_array_equal(estimator.named_steps["simpleimputer"].statistics_, [3, 0])
    estimator.predict([[10000, 5000]])
    np.testing.assert_array_equal(estimator.named_steps["standardscaler"].mean_, [3, 0])
    hgb = make_model("hgb", {"max_iter": 100, "max_leaf_nodes": 7})
    assert hgb.loss == "absolute_error" and not hgb.early_stopping and hgb.min_samples_leaf == 30


def test_bootstrap_known_delta_seed_and_zero_reference():
    result = paired_bootstrap(np.ones(100), np.full(100, 3.0), reps=100)
    assert result["delta_mae"] == 2
    assert result["delta_ci_low"] <= 2 <= result["delta_ci_high"]
    assert result["skill_ci_low"] == pytest.approx(2 / 3)
    weights = moving_block_weights(101, block_days=14, reps=50)
    np.testing.assert_array_equal(weights.sum(axis=1), np.full(50, 101))
    np.testing.assert_array_equal(weights, moving_block_weights(101, block_days=14, reps=50))
    assert np.isnan(paired_bootstrap(np.ones(10), np.zeros(10), reps=10)["skill"])


def test_group_bootstrap_shared_blocks_and_equal_site_weights():
    dates = np.arange(100)
    weather = np.where(dates < 50, 0.0, 4.0)
    model = np.column_stack([weather, 4 - weather])
    reference = np.full_like(model, 5.0)
    result = paired_bootstrap(model, reference, reps=200)
    # Opposite weather variations cancel in EVERY jointly resampled replicate.
    assert result["delta_ci_low"] == pytest.approx(3)
    assert result["delta_ci_high"] == pytest.approx(3)
    assert paired_bootstrap(model[:, 0], reference[:, 0], reps=200)["delta_ci_low"] < 3
    sparse_model = np.column_stack([np.ones(100), np.r_[np.ones(10) * 3, np.full(90, np.nan)]])
    sparse_reference = np.where(np.isfinite(sparse_model), 5.0, np.nan)
    assert paired_bootstrap(sparse_model, sparse_reference, reps=30)["mae_model"] == 2
    # Direct enumeration verifies vectorized weights and paired missing support.
    weights = moving_block_weights(100, reps=30)
    direct = []
    for w in weights:
        sampled_dates = np.repeat(np.arange(100), w)
        sampled = sparse_model[sampled_dates]
        with np.errstate(invalid="ignore", divide="ignore"):
            site_means = np.nansum(sampled, axis=0) / np.isfinite(sampled).sum(axis=0)
        direct.append(5 - np.mean(site_means))
    low, high = np.nanquantile(direct, [0.025, 0.975])
    summary = paired_bootstrap(sparse_model, sparse_reference, weights=weights)
    assert summary["delta_ci_low"] == pytest.approx(low)
    assert summary["delta_ci_high"] == pytest.approx(high)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    sites = make_sites(3)
    daily = make_daily(sites)
    daily.loc[(daily["site_id"] == sites.iloc[0].site_id) &
              (daily["date"] == SPLITS["val"][0]), "tmax"] = np.nan
    calls = []
    original = evaluation.fit_predict

    def recording_fit(train, predict, columns, model, params):
        calls.append((train["date"].min(), train["date"].max(), predict["date"].min(),
                      predict["date"].max(), model, tuple(columns), dict(params)))
        return original(train, predict, columns, model, params)

    out = tmp_path_factory.mktemp("experiment")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluation, "fit_predict", recording_fit)
        result = run_experiment(
            daily, sites, out=out, app_out=out / "app", bootstrap_reps=30,
            linear_grid=[{"alpha": 1.0}, {"alpha": 100.0}],
            hgb_grid=[{"max_iter": 60, "max_leaf_nodes": 7}, {"max_iter": 100, "max_leaf_nodes": 7}],
        )
    return result, out, calls, sites


def test_end_to_end_contract_and_sanity(experiment):
    result, out, _, _ = experiment
    schemas = {"predictions": PREDICTION_COLUMNS, "metrics_site": METRIC_COLUMNS,
               "skill_site": SKILL_SITE_COLUMNS, "skill_group": SKILL_GROUP_COLUMNS,
               "ablation_site": ABLATION_COLUMNS}
    for name, columns in schemas.items():
        frame = pd.read_parquet(out / f"{name}.parquet") if name == "predictions" else pd.read_csv(
            out / f"{name}.csv"
        )
        assert list(frame.columns) == columns
        assert list(result[name].columns) == columns
        assert len(frame) == len(result[name]) > 0
    predictions = result["predictions"]
    pd.testing.assert_frame_equal(pd.read_parquet(out / "app" / "predictions.parquet"), predictions)
    assert not predictions["y_true"].isna().any()
    assert not predictions.duplicated(["site_id", "date"]).any()
    assert set(predictions["split"]) == {"val", "test"}
    skill = result["skill_group"]
    assert set(skill["koppen_group"]) == set("ABCDE") | {"ALL"}
    assert skill.loc[skill["koppen_group"] == "D", "n_sites"].eq(0).all()
    means = result["metrics_site"].query("split == 'test'").groupby("model")["mae"].mean()
    assert means["linear"] < means["persistence"]
    assert means["hgb"] < means["persistence"]


def test_actual_fit_windows_shared_tuning_and_ablation(experiment):
    result, _, calls, _ = experiment
    for train_start, train_end, prediction_start, prediction_end, model, columns, params in calls:
        assert train_start == SPLITS["train"][0]
        if prediction_start >= SPLITS["test"][0]:
            assert train_end == SPLITS["val"][1]  # actual training ends 2023-12-31
            group = next(group for group, names in FEATURE_GROUPS.items() if tuple(names) == columns)
            assert params == result["predictions"].attrs["selected_params"][group][model]
            assert prediction_end == SPLITS["test"][1]
        else:
            assert train_end == SPLITS["train"][1]
            assert prediction_end == SPLITS["val"][1]
    ablation = result["ablation_site"]
    assert len(ablation) == 3 * 2 * 2 * 2
    full = ablation.query("feature_group == 'full'").merge(
        result["metrics_site"], on=["site_id", "split", "model"], suffixes=("_ablation", "_main")
    )
    np.testing.assert_array_equal(full["mae_ablation"], full["mae_main"])
    skill = result["skill_site"].query("reference == 'best_baseline'")
    assert skill.groupby("site_id")["reference_model"].nunique().eq(1).all()


def test_baseline_selection_uses_validation_and_common_support():
    rows = pd.DataFrame({
        "site_id": ["x"] * 4, "split": ["val", "val", "test", "test"],
        "y_true": [0, 100, 0, 0], "persistence": [1, np.nan, 50, 50],
        "climatology": [2, 100, 0, 0],
    })
    assert evaluation._choose_baselines(rows) == {"x": "persistence"}


def test_shared_grid_selection_averages_sites_equally(monkeypatch):
    sites = make_sites(2)
    daily = make_daily(sites)
    # Site 2 has only one validation target: pooling days would select config A,
    # while equal-weight site MAE correctly selects B (1.5 vs 2.0 degrees).
    sparse = (daily["site_id"] == sites.iloc[1].site_id) & split_mask(daily["date"], "val")
    daily.loc[sparse & (daily["date"] != SPLITS["val"][0]), "tmax"] = np.nan

    def fake_fit(train, predict, columns, model, params):
        is_b = params.get("alpha", params.get("max_iter")) == 2
        is_site_one = train["site_id"].iloc[0] == sites.iloc[0].site_id
        error = (3 if is_b else 0) if is_site_one else (0 if is_b else 4)
        return predict["y_true"].to_numpy() + error

    monkeypatch.setattr(evaluation, "fit_predict", fake_fit)
    result = run_experiment(
        daily, sites, bootstrap_reps=5, linear_grid=[{"alpha": 1}, {"alpha": 2}],
        hgb_grid=[{"max_iter": 1}, {"max_iter": 2}], n_jobs=2,
    )
    for configurations in result["predictions"].attrs["selected_params"].values():
        assert configurations == {"linear": {"alpha": 2}, "hgb": {"max_iter": 2}}


def test_rejects_duplicate_days_and_missing_splits():
    daily = make_daily(make_sites(1), end=START + pd.Timedelta(days=30))
    with pytest.raises(ValueError, match="duplicate"):
        build_features(pd.concat([daily, daily.iloc[:1]]))
    with pytest.raises(ValueError, match="each split"):
        run_experiment(daily, make_sites(1))
