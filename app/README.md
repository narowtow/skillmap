# Skillmap app

Run from the repository root, without downloading or training anything:

```powershell
.venv/Scripts/python.exe -m scripts.export_app_data
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.headless true
```

The same export command refreshes the bundle when the full experiment finishes. The app imports no
research modules and uses only the five packages pinned in `app/requirements.txt` plus the standard library.
For Streamlit Community Cloud, select `app/streamlit_app.py` as the entrypoint. The repository theme leaves
the light/dark base to the viewer's preference.

The exporter reads the contract tables from `data/processed/` and the measured concurrency benchmark from
`reports/benchmark_concurrency.csv`. It writes Zstandard-compressed Parquet tables and `manifest.json` to
`app/data/`. It keeps all sampled sites, flags those with metrics and predictions, trims prediction columns,
downcasts floating-point values to float32, and adds paired valid-day counts to the site skill table.
Neither source artifacts nor model fits are modified. The manifest copies study settings from
`src/skillmap/config.py` and records the export timestamp, coverage and measured bundle size.

The preliminary export measured **1,007,582 bytes (1.008 MB)** for **35 of 200 sites with results**.
Multiplying that whole bundle by 200/35 gives **5.758 MB** at 200 result sites. This is an estimate, not a
measured full-run size; compression and missing data may change it. The exporter prints updated measured
sizes after every run.

Set `SKILLMAP_APP_DATA` to read a different exported directory. Each table is cached independently and
its modification time invalidates the cache on replacement. Predictions are loaded only for the city view.
City charts and metrics always show held-out test results; map and climate views offer test and clearly
labeled tuning-validation results. Missing results, undefined skill and unavailable intervals are explicit.
Group MAE averages sites equally; skill comparisons use paired dates and may differ from unpaired MAE.
Optional models, including `hgb_pooled`, are discovered from the exported artifacts. The city view offers
all available learned prediction lines and starts on the final 120 test days, with 3M / 1Y / All controls
and a range slider. Monthly errors still summarize the full test period. Metrics use two decimals and
highlight every model tied for the lowest MAE. Feature ablation shows temp+calendar minus full MAE on
paired sites, with both an equal-site mean and the distribution of site differences.

The map uses Plotly's natural-earth projection with built-in filled land, ocean, coastlines and subtle
country borders. Plotly fetches its world topojson from its CDN in the viewer's browser. Python reads
only precomputed local artifacts, and tests do not request the CDN. Land and ocean colors follow the
active light/dark theme; paper and geo backgrounds are transparent. Scroll-wheel zoom is disabled,
while the zoom/pan toolbar remains available. The initial map extent covers the whole world with minimal
margins. There is no graticule; pending sites use faint dots and result points have a thin dark outline.
The symmetric color limit is the rounded 95th percentile
of absolute skill across models for the selected split/reference, with a 10% floor; hovering retains the
true values. The map is 600 px tall and follows container width, with the scale below it to leave more
room on narrow displays. Teal multiselect chips and phone-width padding are scoped CSS overrides.
Theme configuration is provided for launches from both the repository root and `app/`.

Tests use `streamlit.testing.v1.AppTest` on both the real bundle and tiny generated artifacts. They exercise
map comparisons, city controls, climate filters, all available city selections, pending-only data and an
incomplete bundle, with and without the optional pooled model. They also check export types, paired counts,
metadata, size, headline logic, runtime imports, chart ranges, dynamic color limits and paired ablation
differences, built-in map layers, light/dark map colors, transparent backgrounds and disabled wheel zoom.
No browser rendering or Streamlit Community Cloud deployment is covered by these headless tests.
