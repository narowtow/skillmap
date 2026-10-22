"""Export the small, self-contained Streamlit bundle; never train or fetch data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skillmap import config  # noqa: E402

SITE_COLUMNS = [
    "site_id", "name", "country_code", "lat", "lon", "koppen", "koppen_group", "koppen_desc",
]


def export_app_data(
    source: Path = config.PROCESSED,
    destination: Path = config.APP_DATA,
    benchmark: Path = config.ROOT / "reports" / "benchmark_concurrency.csv",
) -> dict:
    """Trim artifacts and return manifest metadata, including measured bundle size."""
    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(source / "metrics_site.csv", usecols=config.METRIC_COLUMNS,
                          dtype={"mae": float, "rmse": float, "bias": float, "n": "int64"})
    prediction_path = source / "predictions.parquet"
    available_columns = pq.read_schema(prediction_path).names
    known_models = list(dict.fromkeys([*config.MODELS, *sorted(metrics["model"].unique())]))
    models = [model for model in known_models if model in available_columns]
    prediction_columns = [column for column in config.PREDICTION_COLUMNS if column not in config.MODELS]
    predictions = pd.read_parquet(prediction_path, columns=[*prediction_columns, *models])
    predictions = predictions.sort_values(["site_id", "split", "date"]).reset_index(drop=True)
    sites = pd.read_csv(source / "sites.csv", usecols=SITE_COLUMNS, keep_default_na=False)
    result_ids = set(metrics.loc[metrics["n"] > 0, "site_id"]) & set(predictions["site_id"])
    sites["results_available"] = sites["site_id"].isin(result_ids)
    numeric_skill = dict.fromkeys(config.SKILL_VALUE_COLUMNS[1:], float)
    skill = pd.read_csv(source / "skill_site.csv", usecols=config.SKILL_SITE_COLUMNS,
                        dtype=numeric_skill)
    # Counts use the same finite pairs as the skill comparison, not unpaired metric counts.
    counts = []
    for (model, reference), rows in skill.groupby(["model", "reference"]):
        chosen = rows[["site_id", "split", "reference_model"]]
        paired = predictions.merge(chosen, on=["site_id", "split"], validate="many_to_one")
        reference_values = np.where(
            paired["reference_model"].eq(config.BASELINES[0]),
            paired[config.BASELINES[0]], paired[config.BASELINES[1]],
        )
        model_values = paired[model] if model in paired else np.nan
        paired["n"] = np.isfinite(paired["y_true"]) & np.isfinite(model_values)
        paired["n"] &= np.isfinite(reference_values)
        count = paired.groupby(["site_id", "split"], as_index=False)["n"].sum()
        count["model"], count["reference"] = model, reference
        counts.append(count)
    if counts:
        skill = skill.merge(pd.concat(counts), on=["site_id", "split", "model", "reference"])
    else:
        skill["n"] = pd.Series(dtype="int64")
    tables = {
        "sites": sites,
        "metrics_site": metrics,
        "skill_site": skill,
        "skill_group": pd.read_csv(source / "skill_group.csv", usecols=config.SKILL_GROUP_COLUMNS,
                                   dtype={**numeric_skill, "n_sites": "int64"}),
        "ablation_site": pd.read_csv(source / "ablation_site.csv", usecols=config.ABLATION_COLUMNS,
                                     dtype={"mae": float}),
        "predictions": predictions,
        "benchmark_concurrency": pd.read_csv(benchmark),
    }
    for name, table in tables.items():
        floats = table.select_dtypes(include="floating").columns
        table = table.astype(dict.fromkeys(floats, "float32"))
        table.to_parquet(destination / f"{name}.parquet", index=False, compression="zstd")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_sites": len(sites),
        "n_sites_with_results": int(sites["results_available"].sum()),
        "splits": {key: [str(start.date()), str(end.date())] for key, (start, end) in config.SPLITS.items()},
        "models": models,
        "baselines": config.BASELINES,
        "data_period": [str(config.START.date()), str(config.END.date())],
        "prediction_period": [
            str(predictions["date"].min().date()), str(predictions["date"].max().date()),
        ] if len(predictions) else None,
        "weather_variables": config.WEATHER_COLUMNS,
        "bootstrap_block_days": config.BOOTSTRAP_BLOCK_DAYS,
        "bootstrap_reps": config.BOOTSTRAP_REPS,
    }
    # Include the manifest itself in the measured total (iterate to a stable digit count).
    bundle_bytes = sum((destination / f"{name}.parquet").stat().st_size for name in tables)
    manifest_path = destination / "manifest.json"
    manifest["bundle_bytes"] = bundle_bytes
    for _ in range(5):
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        measured = bundle_bytes + manifest_path.stat().st_size
        if measured == manifest["bundle_bytes"]:
            break
        manifest["bundle_bytes"] = measured
    return manifest


def main() -> None:
    """Export default research outputs or explicitly supplied artifact directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=config.PROCESSED)
    parser.add_argument("--destination", type=Path, default=config.APP_DATA)
    parser.add_argument("--benchmark", type=Path, default=config.ROOT / "reports/benchmark_concurrency.csv")
    args = parser.parse_args()
    manifest = export_app_data(args.source, args.destination, args.benchmark)
    n = manifest["n_sites_with_results"]
    size = manifest["bundle_bytes"] / 1_000_000
    print(f"Exported {n}/{manifest['n_sites']} sites with results: {size:.3f} MB")
    if n:
        projected = size * config.N_SITES / n
        print(f"Linear size extrapolation to {config.N_SITES} result sites: {projected:.3f} MB")


if __name__ == "__main__":
    main()
