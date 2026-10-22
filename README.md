# Skillmap

Where in the world is weather predictable, and where does machine learning actually beat simple rules?

Skillmap forecasts next-day high temperature for a sample of cities across all Köppen climate zones and compares
linear and gradient-boosting models against persistence and climatology baselines, using time-based splits only.

Work in progress.

## Data pipeline

The data pipeline uses local GeoNames dumps, the Beck et al. (2023) 1991–2020 Köppen raster and legend,
and `config/research_stations.csv`. Shared settings and output schemas are in
`src/skillmap/config.py` and `contracts/README.md`.

```powershell
.venv/Scripts/python.exe -m scripts.build_sites
.venv/Scripts/python.exe -m scripts.download --dry-run
.venv/Scripts/python.exe -m scripts.download --sites 5 --budget 750 --max-workers 4
.venv/Scripts/python.exe -m scripts.build_dataset
.venv/Scripts/python.exe -m scripts.benchmark_concurrency --dry-run
```

`build_sites` samples local raster cells and permits a nearest classified cell within 5 km only for coastal
NoData. Selection covers rare classes first, then fills group quotas using geographic spread, preferring
cities with at least 100,000 residents. ERA5 cell uniqueness is mandatory. Separation relaxes from the
configured threshold through 100, 25, and 0 km only when necessary; country caps can relax for class coverage.
Unfilled group quotas are redistributed. The coverage CSV reports gaps without changing any classifications.

The generated sample contains 200 sites (A: 35, B: 40, C: 60, D: 50, E: 15). Csc has one candidate and one
selected site; Dsd has no candidates. The build required two spacing relaxations and no country-cap relaxations.

The downloader defaults to complete site histories first, interleaving climate groups in the site order.
Use `--order year-major` for earliest years across sites, or `--site-ids gn-123 st-summit` to choose sites.
Valid caches are skipped; missing or invalid chunks are fetched again. Every HTTP attempt, including retries,
counts toward `--budget` and the shared sliding-window limits. A 429 pauses all workers. Ledger timestamps and
weights seed subsequent runs, conservatively placing a chunk's retry usage at its final timestamp. Run only one
downloader process per cache; threads within that process share all accounting. An abrupt process termination
before journaling cannot be recovered from the ledger.

`build_dataset` keeps partially downloaded sites on the complete study-date index, filling absent days with NaN,
and warns about skipped sites with no valid chunks. API metadata comes from the earliest valid chunk, with a
warning if later years differ. The concurrency benchmark performs real requests only without `--dry-run`;
it uses no cache, shares rate limits across runs, and writes measured timings to `reports/benchmark_concurrency.csv`.

## Experiment

Run the chronological evaluation without downloaded data:

```powershell
.venv/Scripts/python.exe scripts/run_experiment.py --synthetic --out data/synthetic_run
```

Omit `--synthetic` to read `config.DAILY_PARQUET` and `config.SITES_CSV`. Use `--daily`, `--sites`,
`--out`, `--site-ids`, and `--n-jobs` to override inputs, destination, site selection, and site concurrency.
The runner writes predictions, site metrics, paired site/group skill intervals, and feature ablations.
See [the data contract](contracts/README.md) for schemas and missing-data semantics.

The optional pooled HGB is enabled by default (`--pooled` / `--no-pooled`). It learns anomalies across
sites using lagged weather, site descriptors, categorical Köppen group and fold-fitted climatology.
Each local and pooled feature variant tunes its own shared configuration on equal-weight site validation MAE.
`experiment.json` records all candidate scores, selected parameters, fit windows and measured runtime.

For an offline rerun on complete histories, including the current partially downloaded dataset:

```powershell
.venv/Scripts/python.exe -m scripts.run_experiment --complete-only --pooled --n-jobs 8 --no-app-copy
.venv/Scripts/python.exe -m scripts.analysis
.venv/Scripts/python.exe -m scripts.make_figures
.venv/Scripts/python.exe -m scripts.execute_notebooks
```

`scripts/run_all.sh` documents the complete pipeline, including the app worker's `export_app_data` step.
`SKIP_INPUTS=1` skips site building, network downloads and dataset building. Notebooks read processed outputs;
they do not rerun the full experiment. `scripts/build_notebooks.py` regenerates their unexecuted sources.

## Preliminary held-out results

This run uses 35 sites with at least 99% nonmissing Tmax, seven per climate group. Train is 2015–2021,
validation is 2022–2023, and test is 2024–2025. These numbers come from
[`findings.json`](data/processed/findings.json); they describe this subset, not the complete sampling frame.

| Model | Median skill vs best baseline (IQR) | Sites with 95% skill CI above zero |
|---|---:|---:|
| Ridge | 9.0% (6.8–11.7%) | 32/35 (91.4%) |
| Local HGB | 9.2% (6.7–11.7%) | 34/35 (97.1%) |
| Pooled HGB | 9.4% (7.2–11.0%) | 33/35 (94.3%) |

The best baseline is selected per site using validation only. All three models have positive point skill
at every evaluated site. Pooled HGB's mean test MAE is 1.5696 °C versus 1.5692 °C locally; pooling improves
18/35 sites. Adding humidity, pressure and wind lags lowers mean MAE by 0.0495 °C for local HGB,
0.0328 °C for pooled HGB and 0.0382 °C for ridge, relative to temperature/calendar variants.

Local HGB's group skill is largest in A (12.2%) and smallest in D (6.2%). Its strongest training-driver
association is anomaly lag-1 autocorrelation: Spearman ρ = −0.78 (site-bootstrap 95% CI −0.88 to −0.58).
Higher persistence leaves less room to improve on yesterday's temperature. This is an exploratory
association, not an independent causal effect; driver intervals have no multiple-testing adjustment.

![Local HGB skill across evaluated sites](reports/figures/skill_map.png)

The full-feature ridge selects alpha 0.01, inside the extended grid. Full-feature local HGB selects
700 iterations (upper edge) and seven leaves (lower edge); pooled HGB selects 400 iterations (upper edge).
The temperature/calendar variant selects ridge alpha 0.001 and local HGB 200 iterations, both lower edges.
The grids are kept fixed despite these boundary choices.

The measured experiment took 4.74 minutes with eight jobs. Linear scaling in site count projects 27.1 minutes
for 200 sites on the same hardware; the full run has not been benchmarked and pooled fitting may scale
differently. Forecast intervals use paired moving blocks of dates and condition on model selection.
All three [executed notebooks](notebooks/) explain coverage, timing, uncertainty and geographic comparisons.
