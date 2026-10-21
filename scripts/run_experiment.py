"""Run the Skillmap chronological evaluation on daily weather or synthetic data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from skillmap.config import DAILY_PARQUET, PROCESSED, SITES_CSV  # noqa: E402
from skillmap.evaluate import run_experiment  # noqa: E402
from skillmap.synthetic import make_daily, make_sites  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=DAILY_PARQUET)
    parser.add_argument("--sites", type=Path, default=SITES_CSV)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", type=Path, default=PROCESSED)
    parser.add_argument("--site-ids", nargs="+", help="space- or comma-separated site IDs")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    started = perf_counter()
    sites = make_sites() if args.synthetic else pd.read_csv(args.sites, dtype={"site_id": str})
    daily = make_daily(sites) if args.synthetic else pd.read_parquet(args.daily)
    site_ids = [site_id for token in args.site_ids for site_id in token.split(",")] if args.site_ids else None
    result = run_experiment(daily, sites, out=args.out, site_ids=site_ids, n_jobs=args.n_jobs)
    grouped = result["metrics_site"].merge(sites[["site_id", "koppen_group"]], on="site_id")
    table = grouped.groupby(["split", "koppen_group", "model"])["mae"].mean().unstack("model")
    overall = grouped.groupby(["split", "model"])["mae"].mean().unstack("model")
    overall["koppen_group"] = "ALL"
    overall = overall.reset_index().set_index(["split", "koppen_group"])
    print("MAE (equal weight per site):")
    print(pd.concat([table, overall]).sort_index().to_string(float_format=lambda x: f"{x:.3f}"))
    skill = result["skill_group"]
    print("\nSkill vs validation-selected best baseline (paired 95% block-bootstrap CI):")
    print(skill.loc[skill["reference"] == "best_baseline", [
        "split", "koppen_group", "model", "n_sites", "skill", "skill_ci_low", "skill_ci_high",
        "delta_mae", "delta_ci_low", "delta_ci_high",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSelected shared configurations: {result['predictions'].attrs['selected_params']}")
    print(f"Runtime: {perf_counter() - started:.2f} s; sites: {result['predictions']['site_id'].nunique()}")


if __name__ == "__main__":
    main()
