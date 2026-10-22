"""Compute training-only descriptors and summarize held-out experiment artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmap.analysis import run_analysis  # noqa: E402
from skillmap.config import PROCESSED  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROCESSED)
    findings = run_analysis(parser.parse_args().out)
    print(f"Analyzed {findings['n_sites']} sites; see findings.json for computed summaries.")


if __name__ == "__main__":
    main()
