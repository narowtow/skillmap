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
