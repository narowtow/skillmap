#!/usr/bin/env bash
# Full pipeline, from site selection through executed analysis notebooks.
# Download is the only network step; run it only when network access is authorized.
# For an offline rerun with existing inputs: SKIP_INPUTS=1 bash scripts/run_all.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
py="$root/.venv/Scripts/python.exe"
[ -x "$py" ] || py="$root/.venv/bin/python"
if [ "${SKIP_INPUTS:-0}" != "1" ]; then
    "$py" -m scripts.build_sites
    "$py" -m scripts.download
    "$py" -m scripts.build_dataset
fi
# Explicit complete-site selection also supports partially downloaded datasets.
"$py" -m scripts.run_experiment --complete-only --pooled --n-jobs 8 --no-app-copy
"$py" -m scripts.analysis
"$py" -m scripts.make_figures
# Owned by the app worker; this script must exist before running the entire pipeline.
"$py" -m scripts.export_app_data
"$py" -m scripts.execute_notebooks
