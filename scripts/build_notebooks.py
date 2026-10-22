"""Build the three lightweight, artifact-driven analysis notebook sources."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


SETUP = """
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
get_ipython().run_line_magic('matplotlib', 'inline')
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from skillmap.config import PROCESSED, FIGURES, SPLITS, WEATHER_COLUMNS
findings = json.loads((PROCESSED / 'findings.json').read_text())
sites = pd.read_csv(PROCESSED / 'sites.csv', keep_default_na=False)
predictions = pd.read_parquet(PROCESSED / 'predictions.parquet')
selected = sites[sites.site_id.isin(findings['site_ids'])]
display(Markdown(f"This run evaluates **{findings['n_sites']} complete sites**. "
                 f"The full sampling frame contains **{len(sites)} sites**."))
"""


def main() -> None:
    notebooks = {
        "01_sites_and_data": [
            md("""
                # Sites, coverage and data quality

                Skillmap stratifies sites by Köppen class, with climate-group quotas, rare-class coverage
                and geographic spread. Cities and research stations share the same data contract.
                The sample is designed for climate comparisons; it is not weighted by population or land area.
                See `contracts/README.md` and `src/skillmap/sites.py` for the exact selection rules.
                The coverage audit records absent classes without assigning invented classifications.
            """), code(SETUP),
            code("""
                coverage = pd.read_csv(PROCESSED / 'koppen_coverage.csv')
                display(coverage)
                display(selected.groupby('koppen_group').size().rename('evaluated_sites').to_frame())
                from scripts.make_figures import land, COLORS
                fig, ax = plt.subplots(figsize=(12, 6))
                land(ax)
                ax.scatter(sites.lon, sites.lat, s=12, color='#aaaaaa', label='Sampling frame')
                for group, frame in selected.groupby('koppen_group'):
                    ax.scatter(frame.lon, frame.lat, s=45, color=COLORS[group], label=group, zorder=3)
                ax.set_title('Evaluated sites within the stratified sampling frame')
                ax.legend(ncols=3)
                plt.show()
            """), md("""
                ## Data QA

                `daily.parquet` keeps complete calendars even for partially downloaded histories. The QA
                below reports the entire available table and the evaluated subset separately. Observed
                ranges are sanity checks, not assertions that all real weather falls within fixed limits.
            """), code("""
                daily = pd.read_parquet(PROCESSED / 'daily.parquet')
                daily['date'] = pd.to_datetime(daily.date)
                complete = daily.groupby('site_id').tmax.apply(lambda x: x.notna().mean())
                display(complete.rename('fraction_finite_tmax').sort_values().to_frame())
                evaluated = daily[daily.site_id.isin(findings['site_ids'])]
                qa = pd.DataFrame({
                    'missing_all': daily[WEATHER_COLUMNS].isna().mean(),
                    'missing_evaluated': evaluated[WEATHER_COLUMNS].isna().mean(),
                    'min_evaluated': evaluated[WEATHER_COLUMNS].min(),
                    'max_evaluated': evaluated[WEATHER_COLUMNS].max(),
                    'median_evaluated': evaluated[WEATHER_COLUMNS].median(),
                    'unit': ['°C', '°C', '%', 'hPa', 'm/s'],
                })
                display(qa)
                display(pd.Series({
                    'duplicate_site_dates': int(daily.duplicated(['site_id', 'date']).sum()),
                    'tmin_above_tmax': int((evaluated.tmin > evaluated.tmax).sum()),
                    'humidity_outside_0_100': int((~evaluated.rh_mean.between(0,100)
                                                   & evaluated.rh_mean.notna()).sum()),
                    'negative_wind': int(evaluated.wind_mean.lt(0).sum()),
                }).to_frame('count'))
                sample_id = findings['site_ids'][0]
                sample = evaluated[evaluated.site_id == sample_id].sort_values('date')
                ax = sample.plot(x='date', y=['tmax', 'tmin'], figsize=(12,4), linewidth=0.6)
                ax.set(title=f'Available ERA5 temperature history: {sample_id}', ylabel='Temperature (°C)')
                plt.show()
            """), md("""
                ## API and interpretation caveats

                These are ERA5 reanalysis daily aggregates at an API grid cell, not independent station
                observations or an operational forecast archive. Grid-cell elevation and coordinates can
                differ from the selected site. Lagged reanalysis is a retrospective predictability experiment;
                real-time reporting latency and forecast availability are not simulated.

                `timezone=auto` yields local calendar dates, but local-day boundaries are fixed at the
                request-time UTC offset; historical daylight-saving transitions are not reconstructed.
                Cached response metadata records the returned timezone and offset. Temperature is in °C,
                humidity in %, mean sea-level pressure in hPa, and wind speed in m/s.
            """), code("""
                meta = pd.read_csv(PROCESSED / 'api_meta.csv')
                display(meta[meta.site_id.isin(findings['site_ids'])])
            """),
        ],
        "02_baselines_and_models": [
            md("""
                # Baselines, models and honest chronology

                A target row dated **d** can use weather through **d−1** and calendar facts for **d**.
                Climatology, scalers and imputers are fitted on the current fold's training rows.
                Validation chooses shared hyperparameters using the equal-weight mean site MAE.
                Test models refit on train ∪ validation, using the selected configuration.
            """), code(SETUP), code("""
                display(pd.DataFrame(SPLITS, index=['start', 'end']).T)
                from skillmap.features import build_features, FULL
                from skillmap.evaluate import _prepare_site
                daily = pd.read_parquet(PROCESSED / 'daily.parquet')
                sample = daily[daily.site_id == findings['site_ids'][0]].copy()
                table = build_features(sample)
                fold = _prepare_site(table)
                demo = fold.val.head(5).copy()
                demo['latest_weather_date'] = demo.date - pd.Timedelta(days=1)
                display(demo[['date','latest_weather_date','y_true','tmax_lag1',
                              'tmax_lag7','tmax_anomaly_lag1','climatology_target']])
                target = demo.date.iloc[0]
                changed = sample.copy()
                changed.loc[changed.date >= target, WEATHER_COLUMNS] += 1000
                changed_features = build_features(changed)
                protected = table.date <= target
                pd.testing.assert_frame_equal(table.loc[protected, FULL],
                                              changed_features.loc[protected, FULL])
                print('Changing weather on d and all later dates leaves predictors through d unchanged.')
            """), md("""
                ## Models and selected configurations

                Persistence predicts yesterday's Tmax. Climatology fits three annual harmonics per site.
                Local ridge uses median imputation and scaling; local HGB handles missing predictors directly.
                Pooled HGB learns temperature anomalies across sites using the same lag predictors, latitude,
                absolute latitude, elevation, categorical Köppen group and fold-fitted climatology for d.
                Climatology is added back after predicting an anomaly.

                The best baseline is selected separately per site on paired validation dates, and that
                choice is retained on test. Validation numbers below are selection diagnostics.
            """), code("""
                experiment = findings['experiment']
                display(pd.DataFrame(experiment['selected_params']).T)
                scores = []
                for feature_group, models in experiment['validation_scores'].items():
                    for model, candidates in models.items():
                        for candidate in candidates:
                            scores.append(dict(feature_group=feature_group, model=model, **candidate))
                display(pd.DataFrame(scores))
                display(pd.DataFrame(findings['grid_edges']))
                metrics = pd.read_csv(PROCESSED / 'metrics_site.csv')
                display(metrics.groupby(['model','split']).mae.mean().unstack('split'))
                display(pd.DataFrame(experiment['fit_windows']))
                minutes = experiment['runtime_seconds'] / 60
                display(Markdown(f"Measured experiment runtime: **{minutes:.2f} min**, "
                                 f"with **{experiment['n_jobs']} jobs**."))
            """), md("""
                ## Uncertainty

                Skill is `1 − model_MAE / reference_MAE`; positive values favor ML. Paired moving-block
                bootstrap resamples contiguous calendar-date blocks. Missing dates stay in the index,
                and each comparison uses common finite support. Group estimates weight sites equally
                and resample the same date blocks across sites to preserve contemporaneous dependence.
                Percentile intervals condition on the fitted, selected models. They do not include model
                selection uncertainty. Exploratory driver intervals instead bootstrap entire sites.
            """), code("""
                display(pd.Series({k: experiment[k] for k in ['block_days','bootstrap_reps','seed']}))
                skill = pd.read_csv(PROCESSED / 'skill_group.csv')
                display(skill.query("koppen_group == 'ALL' and reference == 'best_baseline'"))
            """),
        ],
        "03_where_ml_wins": [
            md("""
                # Where ML wins, and where prediction remains difficult

                All results below come from saved held-out artifacts. Site descriptors use only the
                training period. The preliminary subset is not a globally representative climate sample.
            """), code(SETUP), code("""
                def figure(name):
                    display(Image(filename=str(FIGURES / name), width=900))
                figure('skill_map.png')
                figure('mae_map.png')
                figure('baseline_vs_model.png')
                display(pd.DataFrame(findings['skill_summary']))
            """), md("## Climate groups and training-period drivers"), code("""
                figure('skill_by_group.png')
                display(pd.DataFrame(findings['group_extremes']).T)
                figure('skill_vs_driver.png')
                display(pd.DataFrame(findings['top_driver_correlations']))
            """), md("""
                Correlations are descriptive associations across sampled sites. Bootstrap intervals do
                not adjust for testing several drivers. Weather variability, latitude and climate class
                are related, so these associations should not be read as independent causal effects.
            """), md("## Calendar months, local seasons and losing sites"), code("""
                figure('seasonal.png')
                seasonal = pd.read_csv(PROCESSED / 'seasonal_group.csv')
                display(seasonal.query("period_type == 'local_season' and model == 'hgb' "
                                       "and reference == 'best_baseline' and n_sites > 0"))
                losing = pd.DataFrame(findings['losing_sites'])
                display(losing if len(losing) else Markdown('No negative point-skill sites in this run.'))
                figure('example_city.png')
            """), md("""
                Local seasons reverse north/south meteorological seasons; the equator uses the northern
                convention. Tropical wet/dry seasons need a different treatment. Losing-site explanations
                are driver-based hypotheses, not established causes. Example cities are selected after
                evaluation as illustrations of extremes; their snippets are not a new evaluation set.
            """), md("## Local versus pooled, and weather-feature ablation"), code("""
                display(pd.Series(findings['pooled_vs_local']))
                display(pd.DataFrame(findings['ablation_deltas']))
                figure('ablation.png')
                figure('concurrency_speedup.png')
            """), md("## Computed conclusion"), code("""
                lines = [f"This held-out run covers {findings['n_sites']} complete sites."]
                for row in findings['skill_summary']:
                    if row['reference'] == 'best_baseline':
                        lines.append(
                            f"{row['model']}: median skill {row['median_skill']:.1%} "
                            f"(IQR {row['q25_skill']:.1%}–{row['q75_skill']:.1%}); "
                            f"{row['n_significant']}/{row['n_sites']} sites have a 95% interval above zero."
                        )
                pooled = findings['pooled_vs_local']
                if pooled:
                    lines.append(f"Pooled minus local HGB mean MAE is "
                                 f"{pooled['mean_delta_pooled_minus_local']:+.3f} °C; "
                                 f"pooling improves {pooled['fraction_pooled_better']:.1%} of sites.")
                lines.extend(findings['caveats'])
                display(Markdown(chr(10).join(lines)))
            """),
        ],
    }
    destination = ROOT / "notebooks"
    destination.mkdir(exist_ok=True)
    for name, cells in notebooks.items():
        notebook = nbf.v4.new_notebook(cells=cells)
        notebook.metadata["kernelspec"] = {
            "display_name": "Skillmap venv", "language": "python", "name": "skillmap-venv"
        }
        nbf.write(notebook, destination / f"{name}.ipynb")
    print(f"Built {len(notebooks)} notebook sources")


if __name__ == "__main__":
    main()
