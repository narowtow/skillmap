"""Build the local, Köppen-stratified site sample and coverage audit."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmap import config
from skillmap.sites import coverage_table, load_candidates, select_sites


def main() -> None:
    candidates, legend = load_candidates()
    sites = select_sites(candidates)
    coverage = coverage_table(candidates, sites, legend)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    sites.to_csv(config.SITES_CSV, index=False)
    coverage.to_csv(config.COVERAGE_CSV, index=False)
    print(f"Selected {len(sites)} of {len(candidates)} candidates")
    print(sites.groupby("koppen_group").size().to_string())
    print("Classes with fewer than 2 selected:")
    print(coverage.loc[coverage.n_selected < 2].to_string(index=False))
    print(f"Constraint relaxations / redistributed slots: {sites.attrs['relaxations']}")


if __name__ == "__main__":
    main()
