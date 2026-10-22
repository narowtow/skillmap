"""Pooled folds use past-only climate features and equal-weight site tuning."""

import numpy as np
import pandas as pd

import skillmap.evaluate as evaluation
from skillmap.config import SPLITS
from skillmap.evaluate import _prepare_site, run_experiment
from skillmap.features import build_features
from skillmap.models import make_model
from skillmap.synthetic import make_daily, make_sites


def test_pooled_regresses_anomaly_and_restores_climatology(monkeypatch):
    import skillmap.models as models

    captured = {}

    class Estimator:
        def fit(self, x, y):
            captured["x"] = x.copy()
            captured["y"] = y.copy()

        def predict(self, x):
            return np.full(len(x), 2.0)

    monkeypatch.setattr(models, "make_model", lambda name, params: Estimator())
    train = pd.DataFrame({
        "lag": [10, 11], "y_true": [20, 22], "climatology_target": [18, 19],
        "lat": [1, 1], "abs_lat": [1, 1], "elevation_m": [20, 20], "koppen_category": [0, 0],
    })
    pred = models.pooled_predict(train, train, ["lag"], {})
    np.testing.assert_array_equal(captured["y"], [2, 3])
    np.testing.assert_array_equal(pred, [20, 21])
    assert list(captured["x"]) == ["lag", *models.POOLED_DESCRIPTORS]


def test_pooled_climate_features_are_fold_fitted():
    table = build_features(make_daily(make_sites(1)))
    before = _prepare_site(table)
    table.loc[table["date"] >= SPLITS["test"][0], "y_true"] += 10000
    after = _prepare_site(table)
    for name in ("train", "val", "refit"):
        pd.testing.assert_frame_equal(getattr(before, name), getattr(after, name))
    np.testing.assert_array_equal(before.test["climatology_target"], after.test["climatology_target"])
    table.loc[table["date"] >= SPLITS["val"][0], "y_true"] += 10000
    after = _prepare_site(table)
    np.testing.assert_array_equal(before.val["climatology_target"], after.val["climatology_target"])
    pd.testing.assert_frame_equal(before.train, after.train)
    assert make_model("hgb_pooled", {"max_iter": 2}).categorical_features == ["koppen_category"]


def test_pooled_actual_fold_windows_anomaly_target_and_site_tuning(monkeypatch):
    sites = make_sites(2)
    daily = make_daily(sites)
    sparse = (daily["site_id"] == sites.iloc[1].site_id) & daily["date"].between(*SPLITS["val"])
    daily.loc[sparse & (daily["date"] != SPLITS["val"][0]), "tmax"] = np.nan
    calls = []
    real_pooled = evaluation.pooled_predict

    def record(train, predict, columns, params):
        calls.append((train.copy(), predict.copy(), params.copy()))
        if predict["date"].min() == SPLITS["test"][0]:
            # Exercise an actual fit and anomaly reconstruction, including categorical input.
            return real_pooled(train, predict, columns, params)
        is_b = params["max_iter"] == 2
        error = np.where(predict["site_id"] == sites.iloc[0].site_id, 3 if is_b else 0, 0 if is_b else 4)
        return predict["y_true"].to_numpy() + error

    monkeypatch.setattr(evaluation, "pooled_predict", record)
    result = run_experiment(
        daily, sites, bootstrap_reps=5, linear_grid=[{"alpha": 1}],
        hgb_grid=[{"max_iter": 2}], pooled_grid=[{"max_iter": 1}, {"max_iter": 2}],
    )
    for group in result["predictions"].attrs["selected_params"].values():
        assert group["hgb_pooled"] == {"max_iter": 2}
    for train, predict, _ in calls:
        expected_end = SPLITS["val"][1] if predict["date"].min() >= SPLITS["test"][0] else SPLITS["train"][1]
        assert train["date"].max() == expected_end
        assert train["date"].max() < predict["date"].min()
        assert set(train["site_id"]) == set(sites["site_id"])
        assert train["climatology_target"].notna().all()
        assert train["koppen_category"].nunique() == 2
    assert "hgb_pooled" in result["predictions"]
    assert "hgb_pooled" in set(result["skill_site"]["model"])
    assert result["predictions"]["hgb_pooled"].notna().all()
