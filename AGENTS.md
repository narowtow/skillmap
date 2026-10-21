# AGENTS.md

Skillmap asks where in the world next-day high temperature (Tmax) is predictable, and where ML beats simple
rules. ~200 sites stratified by Köppen climate class; Open-Meteo ERA5 daily data 2015–2025; persistence and
climatology baselines vs per-site ridge regression and gradient boosting; Streamlit app.

## Ground rules
- **Read `contracts/README.md` first.** File formats, the forecast-timing (leakage) rule and split semantics are
  fixed there. Shared constants live in `src/skillmap/config.py`; import them, never re-hardcode them.
- **No network.** All dependencies are preinstalled. Never call the real API in tests; mock HTTP.
- **No leakage.** Predictors for target date d use data ≤ d−1 only. Climatology/scalers are fit on training rows
  of the current fold. Splits are chronological; never shuffle across time.
- Python ≥ 3.12, pandas/numpy/scikit-learn. Type hints, short docstrings, small pure functions where possible.
  Match the existing style (ruff config in `pyproject.toml`).
- Don't commit raw data (`data/raw/`). Processed outputs in `data/processed/` and `app/data/` are fine.
- Keep the README and docs factual: report only numbers that a script actually produced.

## Running things
- Python: `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python`. The package is imported from `src/`
  (pytest sets `pythonpath`; scripts add `src` to `sys.path` themselves).
- The venv's base interpreter is the in-workspace copy at `.pyruntime/` (git-ignored). If launching Python is
  denied, check that `.venv/pyvenv.cfg` `home` points to `<this worktree>/.pyruntime` and fix it.
- Gate: `bash scripts/check.sh` (ruff + pytest). If bash is unavailable, run its two steps directly.
  Run it before finishing and report the result honestly.

## Layout
```
src/skillmap/   config, synthetic, sites, fetch, dataset, features, baselines, models, evaluate, bootstrap
scripts/        CLI entry points (build_sites, download, benchmark_concurrency, build_dataset, run_experiment, ...)
config/         research_stations.csv
contracts/      data contract
data/raw/       git-ignored downloads (GeoNames, Köppen GeoTIFF, API cache)
data/processed/ sites, coverage audit, daily table, predictions, metrics
app/            Streamlit app + small Parquet artifacts
notebooks/  reports/figures/  tests/
```
