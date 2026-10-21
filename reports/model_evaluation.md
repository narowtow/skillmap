# Synthetic modeling and evaluation benchmark

These are synthetic checks, not estimates of real-world forecasting performance. No real weather was
downloaded or evaluated. The generated artifacts are in `reports/synthetic_benchmark/`.

## Method

Predictors use calendar-day lags within each site, including explicit missing dates. History carries across
split boundaries. All rows with finite targets are usable even when some or all lagged predictors are missing.
The ridge pipeline fits its median imputer and scaler inside each fold; HGB handles missing predictors itself.
Climatology fits an intercept and three annual harmonics on training targets, separately for train and refit
windows. Yesterday's anomaly uses that fold's fitted climatology evaluated for yesterday.

Validation uses 2015–2021 fits and 2022–2023 targets. Test uses refits on 2015–2023 and 2024–2025 targets.
Each model and feature group selects one shared configuration by equal-weight mean site validation MAE.
Ridge searches alpha 0.1, 1, 10, 100. HGB searches 100, 200, 300 iterations and 7, 15 leaves with the
requested fixed parameters. Both feature groups selected ridge alpha 10 and HGB 200 iterations/7 leaves.
The temp/calendar ablation is tuned separately from the full weather model.

The best baseline is selected per site on common finite validation observations and reused on test. Paired
bootstrap comparisons use overlapping 14-day date blocks, 1,000 replicates, and seed 42. All sites within a
group share each replicate's sampled date blocks, and each group MAE weights its sites equally. Missing
support and zero-reference behavior are documented in the contract. These intervals condition on model and
hyperparameter selection; validation intervals are selection diagnostics.

## Measured results

Command, with all five synthetic sites, the full 2015–2025 period, full grids, both feature groups, and
1,000 bootstrap replicates:

```powershell
.venv/Scripts/python.exe -m scripts.run_experiment --synthetic --out reports/synthetic_benchmark --n-jobs 1
```

The script reported **19.73 seconds**, timing data generation through artifact writes (excluding interpreter
startup and module imports). A linear serial extrapolation to 200 sites is `19.73 * 200 / 5 = 789.2 seconds`,
or **about 13.2 minutes**. This is an estimate; a complete 200-site experiment has not been run. Actual runtime
depends on data, shared tuning choices, hardware and concurrency.

Test MAEs, equal weight across the five sites:

| Model | MAE (°C) |
|---|---:|
| Persistence | 2.435 |
| Climatology | 3.920 |
| Ridge | 2.265 |
| HGB | 2.349 |

Against the validation-selected best baseline, overall test skill was **0.071 [0.056, 0.085]** for ridge and
**0.036 [0.010, 0.059]** for HGB (paired 95% intervals). HGB did not improve every site: its group D test MAE
was 3.421°C versus persistence's 3.368°C. These small synthetic samples do not support real climate claims.

Separate performance checks used `python -m timeit`, one repetition, with numerical thread pools limited to
one thread. One group bootstrap on 731 dates × 200 sites × 1,000 replicates took **31.2 milliseconds** using
constructed finite absolute errors. Building both skill tables (all site/group/model/reference/split
comparisons) for 200 sites took **7.95 seconds**, using 40 copies of the five-site prediction artifact with
distinct IDs. This measures inference throughput, not performance on 200 independent real sites.

A full-grid CLI smoke run with two sites and `--n-jobs 2` completed in **8.88 seconds**. Joblib distributes
site fits across threads while the experiment limits numerical thread pools to prevent nested oversubscription.

## Verification

Tests cover raw-input lag causality, fold-fitted climatology isolation, missing dates, site boundaries,
chronological splits, actual fit-window end dates, train-only pipeline transforms, hand-computed metrics,
bootstrap shared-date dependence and equal site weights, aggregate shared tuning, output schemas, baseline
selection, ablations, and synthetic average improvement over persistence. The suite uses small grids and
30 bootstrap replicates for the three-site end-to-end run.

`bash scripts/check.sh` passed: Ruff reported no findings and pytest passed **18 tests in 7.58 seconds**.
The sandbox run used a workspace-local `TMP`/`TEMP` directory because the default Windows pytest temporary
directory was inaccessible. The existing venv's `home` was repaired to point to this worktree's `.pyruntime`.

The optional smoothed day-of-year climatology sensitivity baseline is not implemented. Real-data behavior
and an actual full 200-site model fit remain unverified. No files were committed.
