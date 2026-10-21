# Data contract

These are the files that separate modules hand to each other. Constants (paths, splits, variables, quotas) live in
`src/skillmap/config.py`. `src/skillmap/synthetic.py` produces data in these formats for tests.

All dates are local calendar days (Open-Meteo `timezone=auto`) stored as `datetime64[ns]` at midnight, without a timezone.
Temperatures are in °C, relative humidity in %, pressure in hPa, and wind in m/s. Missing values are NaN, never sentinels.

## `data/processed/sites.csv`: one row per selected site

| column | type | notes |
|---|---|---|
| `site_id` | str | unique and stable. Cities: `gn-<geonameid>`. Stations: `st-<station_id>` |
| `name` | str | ASCII-safe display name |
| `country_code` | str | ISO-3166 alpha-2 (`AQ` for Antarctica) |
| `lat`, `lon` | float | WGS84 degrees, as sampled (not the grid cell) |
| `elevation_m` | float | GeoNames `dem` (or station elevation) |
| `population` | int | 0 for stations |
| `site_type` | str | `city` or `station` |
| `koppen_code` | int | 1–30, Beck et al. 2023 legend, 1991–2020 map |
| `koppen` | str | e.g. `Cfb` |
| `koppen_group` | str | `A`–`E` (first letter) |
| `koppen_desc` | str | legend description |
| `koppen_fallback` | bool | true if the point was NoData (coast) and the nearest classified cell within 5 km was used |
| `geonameid` | Int64 | null for stations |

`data/processed/koppen_coverage.csv` has one row for each of the 30 classes: `koppen_code, koppen, koppen_group,
n_candidates, n_selected, countries` (`;`-joined, selected only).

## `data/processed/daily.parquet`: long format, one row per (site, date)

`site_id, date, tmax, tmin, rh_mean, msl_mean, wind_mean`. It covers `config.START`–`config.END` for every site, with one
row per date even when values are missing.

`data/processed/api_meta.csv` has one row per site, from the API response: `site_id, grid_lat, grid_lon,
grid_elevation, timezone, utc_offset_seconds`.

## Raw API cache (git-ignored)

`data/raw/openmeteo/<site_id>/<year>.json` holds the unmodified response body for one site-year.
`data/raw/openmeteo/ledger.jsonl` has one JSON object per line per completed chunk:
`{"site_id", "year", "weight", "status", "fetched_at", "attempts"}`. A chunk is done when its JSON exists and
parses with all requested variables.

## Forecast timing (leakage rule)

A row is (site, target date `d`). **Every predictor must be computable from data dated ≤ `d − 1`**, plus calendar
facts about `d` (day of year). Climatology and any fitted transform (scaler, imputer, harmonics) are fit only on the
training rows of the current fold.

Splits are by target date, from `config.SPLITS`. Validation predictions come from models fit on `train`. Test
predictions come from models refit on `train ∪ val` with hyperparameters chosen on validation. Test is evaluated once.

## `data/processed/predictions.parquet` (also copied to `app/data/`)

`site_id, date, split (val|test), y_true, persistence, climatology, linear, hgb` (+ optional `hgb_pooled`).
Rows with NaN `y_true` are dropped.

## `data/processed/metrics_site.csv`

One row per (site, split, model): `site_id, split, model, mae, rmse, bias, n`.

## `data/processed/skill_site.csv`

One row per (site, split, model, reference). `model ∈ {linear, hgb, ...}`,
`reference ∈ {persistence, climatology, best_baseline}`.
Columns: `site_id, split, model, reference, reference_model, mae_model, mae_ref, delta_mae, delta_ci_low,
delta_ci_high, skill, skill_ci_low, skill_ci_high`.
- `delta_mae = mae_ref − mae_model` (°C, positive = ML better); `skill = 1 − mae_model / mae_ref`.
- CIs: paired moving-block bootstrap over dates (`config.BOOTSTRAP_BLOCK_DAYS`, `config.BOOTSTRAP_REPS`), 95%.
- `best_baseline` = whichever of persistence/climatology had the lower **validation** MAE for that site;
  `reference_model` names it. The same choice is used for test.

## `data/processed/skill_group.csv`

The same columns as `skill_site.csv`, with `koppen_group` (A–E, or `ALL`) in place of `site_id`, plus `n_sites`.
Group MAE is the equal-weight mean over sites. Bootstrap resamples the same date blocks across all sites in the group.

For a group's `best_baseline`, `reference_model` names the baseline if every site chose the same one, or
`per_site` if choices differ. All six groups are emitted; absent groups have `n_sites=0` and NaN estimates.

## `data/processed/ablation_site.csv`

One row per (site, split, learned model, feature group): `site_id, split, model, feature_group, mae`.
`feature_group` is `temp_calendar` or `full`. Each feature group selects its own shared model configuration on
the equal-weight mean site validation MAE. `full` matches the main predictions and metrics.

## Evaluation details

Baseline selection compares persistence and climatology on their common finite validation dates. Metrics
drop nonfinite truth/prediction pairs; skill comparisons use common finite support for each model/reference
pair. Thus `n` can differ by model and skill-table MAEs can differ from unpaired metric-table MAEs when data
are missing. Missing calendar dates remain in the bootstrap sampling index. The ordinary overlapping
moving-block bootstrap samples blocks with replacement and truncates the last block to the split's length.
A replicate without any paired observations for one of its sites is undefined and omitted from its interval;
skill is NaN when reference MAE is zero. Intervals are conditional on the fitted, selected models and do not
include model-selection uncertainty. Validation results are selection diagnostics; test results are held out.

When the experiment writes to `config.PROCESSED`, predictions are also copied to `config.APP_DATA`.
Custom output directories are self-contained unless `app_out` is explicitly supplied to `run_experiment`.
