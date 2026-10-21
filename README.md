# Skillmap

Where in the world is weather predictable, and where does machine learning actually beat simple rules?

Skillmap forecasts next-day high temperature for a sample of cities across all Köppen climate zones and compares
linear and gradient-boosting models against persistence and climatology baselines, using time-based splits only.

Run the chronological evaluation without downloaded data:

```powershell
.venv/Scripts/python.exe -m scripts.run_experiment --synthetic --out reports/synthetic_benchmark
```

Omit `--synthetic` to read `config.DAILY_PARQUET` and `config.SITES_CSV`. Use `--daily`, `--sites`,
`--out`, `--site-ids`, and `--n-jobs` to override inputs, destination, site selection, and site concurrency.
The runner writes predictions, site metrics, paired site/group skill intervals, and feature ablations.
See [the data contract](contracts/README.md) for schemas and missing-data semantics, and
[the synthetic evaluation report](reports/model_evaluation.md) for measured results and runtime.
