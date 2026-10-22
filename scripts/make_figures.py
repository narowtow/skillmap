"""Render readable, offline matplotlib figures from processed experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillmap.analysis import DRIVERS  # noqa: E402
from skillmap.config import FIGURES, PROCESSED, ROOT  # noqa: E402

COLORS = {"A": "#009E73", "B": "#E69F00", "C": "#0072B2", "D": "#CC79A7", "E": "#D55E00"}
MODEL_COLORS = {"linear": "#0072B2", "hgb": "#D55E00", "hgb_pooled": "#009E73"}
LABELS = {"daily_change": "Mean daily change (°C)", "anomaly_acf1": "Anomaly lag-1 correlation",
          "seasonal_amplitude": "Seasonal half-range (°C)", "anomaly_sd": "Anomaly SD (°C)",
          "abs_lat": "Absolute latitude (°)", "elevation_m": "Elevation (m)"}


def land(ax: plt.Axes, path: Path = ROOT / "data/external/ne_110m_land.geojson") -> None:
    """Parse GeoJSON polygon rings without mapping dependencies; use lon/lat axes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for feature in data["features"]:
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            ax.add_patch(Polygon(polygon[0], facecolor="#e7e7e7", edgecolor="#bbbbbb", lw=0.3))
            for hole in polygon[1:]:
                ax.add_patch(Polygon(hole, facecolor="white", edgecolor="#bbbbbb", lw=0.2))
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude (°)", ylabel="Latitude (°)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.18)


def save(fig: plt.Figure, destination: Path, name: str) -> None:
    fig.savefig(destination / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                         "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False})
    sites = pd.read_csv(args.data / "sites.csv", keep_default_na=False)
    drivers = pd.read_csv(args.data / "drivers_site.csv")
    hgb = drivers.query("model == 'hgb' and reference == 'best_baseline'")
    findings = json.loads((args.data / "findings.json").read_text())
    for variable, name in (("skill", "skill_map.png"), ("mae", "mae_map.png")):
        fig, ax = plt.subplots(figsize=(9, 5))
        land(ax)
        if variable == "skill":
            bound = max(0.01, float(hgb[variable].abs().max()))
            scatter = ax.scatter(hgb.lon, hgb.lat, c=hgb[variable], cmap="PuOr", s=65,
                                 norm=TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound),
                                 edgecolor="#333333", linewidth=0.5, zorder=3)
            title = f"HGB median skill is {hgb.skill.median():.1%} across {len(hgb)} held-out sites"
            label = "Skill vs validation-selected best baseline"
        else:
            scatter = ax.scatter(hgb.lon, hgb.lat, c=hgb[variable], cmap="viridis", s=65,
                                 edgecolor="#333333", linewidth=0.5, zorder=3)
            title = f"HGB test MAE spans {hgb.mae.min():.2f}–{hgb.mae.max():.2f} °C"
            label = "Test MAE (°C; lower is more predictable)"
        fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.18, shrink=0.8, label=label)
        ax.set_title(title)
        save(fig, args.out, name)

    group = pd.read_csv(args.data / "skill_group.csv").query(
        "split == 'test' and reference == 'best_baseline' and n_sites > 0"
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = list(group.koppen_group.unique())
    models = list(group.model.unique())
    for j, model in enumerate(models):
        frame = group.query("model == @model").set_index("koppen_group").reindex(names)
        x = np.arange(len(names)) + (j - (len(models) - 1) / 2) * 0.18
        ax.vlines(x, frame.skill_ci_low, frame.skill_ci_high, color=MODEL_COLORS[model], lw=2)
        ax.scatter(x, frame.skill, color=MODEL_COLORS[model], label=model, s=35, zorder=3)
    ax.axhline(0, color="gray", lw=1)
    counts = group.drop_duplicates("koppen_group").set_index("koppen_group").n_sites
    hgb_groups = group.query("model == 'hgb' and koppen_group != 'ALL'")
    best_group = hgb_groups.loc[hgb_groups.skill.idxmax(), "koppen_group"]
    worst_group = hgb_groups.loc[hgb_groups.skill.idxmin(), "koppen_group"]
    ax.set(xticks=np.arange(len(names)), xticklabels=[f"{g}\n(n={counts[g]})" for g in names],
           ylabel="Test skill vs best baseline",
           title=f"Local HGB gains are largest in {best_group} and smallest in {worst_group}")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save(fig, args.out, "skill_by_group.png")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for group_name, frame in hgb.groupby("koppen_group"):
        ax.scatter(frame.mae_ref, frame.mae_model, color=COLORS[group_name], label=group_name, s=55)
    limit = max(hgb.mae_ref.max(), hgb.mae_model.max()) * 1.08
    ax.plot([0, limit], [0, limit], "--", color="gray", label="Equal MAE")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Best-baseline test MAE (°C)",
           ylabel="HGB test MAE (°C)",
           title=f"HGB beats the baseline at {hgb.skill.gt(0).sum()} of {len(hgb)} sites")
    ax.legend(ncols=2)
    ax.grid(alpha=0.2)
    save(fig, args.out, "baseline_vs_model.png")

    corr = pd.read_csv(args.data / "driver_correlations.csv").query(
        "model == 'hgb' and reference == 'best_baseline' and outcome == 'skill'"
    )
    strongest = corr.loc[corr.rho.abs().sort_values(ascending=False).index].head(min(4, len(DRIVERS)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, row in zip(axes.flat, strongest.itertuples(), strict=True):
        for group_name, frame in hgb.groupby("koppen_group"):
            ax.scatter(frame[row.driver], frame.skill, color=COLORS[group_name], s=35, label=group_name)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(xlabel=LABELS[row.driver], ylabel="HGB test skill",
               title=f"ρ={row.rho:.2f} [95% CI {row.ci_low:.2f}, {row.ci_high:.2f}]")
    axes.flat[0].legend(ncols=3, fontsize=9)
    driver_name = LABELS[strongest.iloc[0].driver].split(" (")[0].lower()
    fig.suptitle(f"HGB gains correlate most with {driver_name}", fontsize=15)
    save(fig, args.out, "skill_vs_driver.png")

    ablation = pd.read_csv(args.data / "ablation_site.csv").query("split == 'test'").pivot(
        index=["site_id", "model"], columns="feature_group", values="mae"
    ).reset_index()
    ablation["delta"] = ablation["full"] - ablation["temp_calendar"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rng = np.random.default_rng(0)
    for i, (model, frame) in enumerate(ablation.groupby("model", sort=False)):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(frame)), frame.delta,
                   color=MODEL_COLORS[model], alpha=0.65, s=30)
        ax.scatter(i, frame.delta.mean(), color="black", marker="D", s=60, zorder=4)
    ax.axhline(0, color="gray", lw=1)
    ax.set(xticks=range(ablation.model.nunique()), xticklabels=ablation.model.unique(),
           ylabel="Full − temperature/calendar MAE (°C)",
           title=f"Extra weather lags lower mean MAE for "
                 f"{ablation.groupby('model').delta.mean().lt(0).sum()}/{ablation.model.nunique()} models")
    ax.text(0.01, 0.03, "Dots: sites; black diamonds: equal-site means", transform=ax.transAxes, fontsize=10)
    save(fig, args.out, "ablation.png")

    seasonal = pd.read_csv(args.data / "seasonal_group.csv").query(
        "reference == 'best_baseline' and model == 'hgb' and n_sites > 0"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for group_name, frame in seasonal.groupby("koppen_group"):
        if group_name == "ALL":
            continue
        month = frame.query("period_type == 'month'").copy()
        month["period"] = month.period.astype(int)
        month = month.sort_values("period")
        season = frame.query("period_type == 'local_season'").set_index("period").reindex(
            ["summer", "autumn", "winter", "spring"]
        )
        axes[0].plot(month.period, month.skill, color=COLORS[group_name], marker="o", label=group_name)
        axes[1].plot(range(4), season.skill, color=COLORS[group_name], marker="o", label=group_name)
    axes[0].set(xlabel="Calendar month", xticks=range(1, 13), ylabel="HGB test skill")
    axes[1].set(xticks=range(4), xticklabels=["Summer", "Autumn", "Winter", "Spring"], xlabel="Local season")
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.8)
        ax.grid(alpha=0.2)
    axes[0].legend(ncols=3)
    overall_seasons = seasonal.query("koppen_group == 'ALL' and period_type == 'local_season'")
    peak = overall_seasons.loc[overall_seasons.skill.idxmax(), "period"]
    fig.suptitle(f"Overall HGB gain peaks in local {peak}; patterns differ by climate", fontsize=15)
    save(fig, args.out, "seasonal.png")

    pred = pd.read_parquet(args.data / "predictions.parquet").query("split == 'test'")
    example_ids = [hgb.loc[hgb.skill.idxmax(), "site_id"], hgb.loc[hgb.skill.idxmin(), "site_id"]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    for ax, site_id in zip(axes, example_ids, strict=True):
        frame = pred.loc[pred.site_id == site_id].sort_values("date").head(90)
        site = sites.set_index("site_id").loc[site_id]
        site_skill = hgb.set_index("site_id").loc[site_id]
        reference = site_skill.reference_model
        ax.plot(frame.date, frame.y_true, color="black", lw=1.6, label="ERA5 Tmax")
        ax.plot(frame.date, frame.hgb, color=MODEL_COLORS["hgb"], lw=1.3, label="HGB")
        ax.plot(frame.date, frame[reference], color="#0072B2", lw=1, alpha=0.7, label=reference)
        ax.set(ylabel="Tmax (°C)",
               title=f"{site['name']} ({site_id}): full-test skill {site_skill.skill:.1%}")
        ax.legend(ncols=3, loc="upper left")
        ax.grid(alpha=0.2)
    fig.suptitle("First 90 test days at the highest- and lowest-skill sites", fontsize=15)
    save(fig, args.out, "example_city.png")

    benchmark = pd.read_csv(ROOT / "reports/benchmark_concurrency.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(benchmark.threads, benchmark.speedup, "o-", color="#0072B2", label="Recorded request benchmark")
    ax.plot(benchmark.threads, benchmark.threads, "--", color="gray", label="Ideal linear")
    best = benchmark.loc[benchmark.threads.idxmax()]
    ax.set(xlabel="Download threads", ylabel="Speedup vs one thread", xticks=benchmark.threads,
           title=f"Recorded download speedup reaches {best.speedup:.1f}× with {int(best.threads)} threads")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, args.out, "concurrency_speedup.png")
    print(f"Wrote nine figures for {findings['n_sites']} sites to {args.out}")


if __name__ == "__main__":
    main()
