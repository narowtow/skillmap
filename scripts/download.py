"""Download site-year archive chunks; real HTTP is only used without --dry-run."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmap import config
from skillmap.fetch import cache_path, download_all, order_sites, plan_chunks, read_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--sites", type=int, help="First N sites after group interleaving")
    parser.add_argument("--site-ids", nargs="+", help="Space- or comma-separated site IDs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--order", choices=["site-major", "year-major"], default="site-major")
    args = parser.parse_args()
    sites = order_sites(pd.read_csv(config.SITES_CSV, keep_default_na=False))
    if args.site_ids:
        ids = {item for value in args.site_ids for item in value.split(",")}
        missing = ids - set(sites.site_id)
        if missing:
            parser.error(f"Unknown site IDs: {sorted(missing)}")
        sites = sites.loc[sites.site_id.isin(ids)]
    if args.sites is not None:
        if args.sites < 1:
            parser.error("--sites must be positive")
        sites = sites.head(args.sites)
    if args.max_workers < 1 or (args.budget is not None and args.budget < 0):
        parser.error("Workers must be positive and budget nonnegative")
    chunks = plan_chunks(sites, args.order)
    pending = [c for c in chunks if read_cache(cache_path(config.OPENMETEO_CACHE, c), c) is None]
    total = sum(c.weight for c in pending)
    print(f"Sites: {len(sites)}; chunks: {len(chunks)}; pending: {len(pending)}; weight: {total:.2f}")
    print(f"Estimated daily-cap equivalents: {total / config.RATE_LIMITS['day']:.3f}")
    print(f"Order: {args.order}; budget: {args.budget}; workers: {args.max_workers}")
    if not args.dry_run:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        results = download_all(sites, args.max_workers, args.budget, order=args.order)
        print(pd.Series([r.status for r in results]).value_counts().to_string())


if __name__ == "__main__":
    main()
