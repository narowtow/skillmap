"""Time 48 uncached, 14-day requests at 1/2/4/8 threads (192 base weighted calls)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmap import config
from skillmap.fetch import Chunk, PauseGate, WeightedRateLimiter, fetch_chunk, ledger_usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sites = pd.read_csv(config.SITES_CSV, keep_default_na=False).head(48)
    if len(sites) != 48:
        parser.error("Benchmark requires at least 48 selected sites")
    # Each run gets its own 14-day window so no run can benefit from a server-side cache of an earlier one.
    def batch(run: int) -> list[Chunk]:
        start = config.START + pd.Timedelta(days=14 * run)
        return [Chunk(row.site_id, row.lat, row.lon, start, start + pd.Timedelta(days=13))
                for row in sites.itertuples()]

    chunks = batch(0)
    print(f"Threads: 1, 2, 4, 8; requests/run: {len(chunks)}; days/request: 14")
    print(f"Total base weight: {sum(c.weight for c in chunks) * 4:.0f} (retries charged separately)")
    if args.dry_run:
        return
    logging.basicConfig(level=logging.INFO)
    # One limiter across all runs automatically pauses at any sliding-window cap.
    limiter = WeightedRateLimiter(past_usage=ledger_usage(config.OPENMETEO_CACHE / "ledger.jsonl"))
    gate = PauseGate()
    records = []
    with requests.Session() as session:
        for run, threads in enumerate((1, 2, 4, 8)):
            chunks = batch(run)
            began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=threads) as executor:
                results = list(executor.map(lambda chunk: fetch_chunk(
                    chunk, session=session, limiter=limiter, gate=gate, use_cache=False,
                ), chunks))
            seconds = time.perf_counter() - began
            if any(result.status != "done" for result in results):
                raise RuntimeError("Benchmark failed; speedups require all 48 requests to succeed")
            records.append({"threads": threads, "requests": len(chunks), "seconds": seconds,
                            "speedup": records[0]["seconds"] / seconds if records else 1.0})
    output = pd.DataFrame(records)
    destination = config.ROOT / "reports" / "benchmark_concurrency.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
