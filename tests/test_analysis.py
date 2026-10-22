"""Training isolation, seasonal conventions and site-resampled inference."""

import numpy as np
import pandas as pd
import pytest

from skillmap.analysis import local_season, spearman_ci, training_drivers
from skillmap.config import SPLITS
from skillmap.synthetic import make_daily, make_sites


def test_training_drivers_ignore_all_future_weather():
    sites = make_sites(2)
    daily = make_daily(sites)
    before = training_drivers(daily, sites)
    daily.loc[daily["date"] > SPLITS["train"][1], "tmax"] = 1e6
    pd.testing.assert_frame_equal(before, training_drivers(daily, sites))
    # A missing day must not turn a two-day change into a day-to-day change.
    daily.loc[daily["date"] == SPLITS["train"][0] + pd.Timedelta(days=10), "tmax"] = np.nan
    after = training_drivers(daily, sites)
    assert after["n_train"].eq(before["n_train"] - 1).all()


def test_hemisphere_seasons():
    assert local_season(1, 40) == "winter"
    assert local_season(12, -40) == "summer"
    assert local_season(7, -40) == "winter"
    assert local_season(4, 0) == "spring"
    assert local_season(10, -2) == "spring"
    with pytest.raises(ValueError):
        local_season(0, 30)


def test_spearman_ci_shape_ties_missing_and_determinism():
    x = [1, 1, 2, 3, 4, 5, np.nan]
    y = [5, 5, 4, 3, 2, 1, 0]
    result = spearman_ci(x, y, reps=50)
    assert set(result) == {"rho", "ci_low", "ci_high", "n_sites", "n_valid_reps"}
    assert result["n_sites"] == 6
    assert result["ci_low"] <= result["rho"] <= result["ci_high"]
    assert result["rho"] == pytest.approx(-1)
    assert result == spearman_ci(x, y, reps=50)
    assert np.isnan(spearman_ci([1, 1, 1], [1, 2, 3], reps=5)["rho"])

