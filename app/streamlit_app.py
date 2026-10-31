"""Skillmap: a precomputed explorer for next-day Tmax skill."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_helpers import (  # noqa: E402
    COLORS,
    GROUP_LABELS,
    MONTH_LABELS,
    REFERENCE_LABELS,
    ablation_differences,
    headline,
    interval_text,
    map_palette,
    map_rows,
    model_label,
    model_order,
    skill_color_limit,
)

DATA = Path(os.environ.get("SKILLMAP_APP_DATA", Path(__file__).parent / "data"))
PAGES = ["Map", "City explorer", "Climate zones", "Method & limits"]


@st.cache_data(show_spinner=False)
def load_table(directory: str, name: str, modified: int) -> pd.DataFrame:
    """Cache each small table separately; file timestamps invalidate replaced exports."""
    return pd.read_parquet(Path(directory) / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def load_json(path: str, modified: int) -> dict:
    """Cache exported metadata."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(name: str) -> pd.DataFrame:
    """Read one exported table without loading unused predictions on other pages."""
    path = DATA / f"{name}.parquet"
    return load_table(str(DATA), name, path.stat().st_mtime_ns)


def theme_type() -> str | None:
    """Configured base theme if pinned, else the browser's reported theme."""
    return st.get_option("theme.base") or st.context.theme.type


def chart(figure: go.Figure, key: str) -> None:
    """Apply a theme-aware, restrained chart style."""
    is_map = key == "world_map"
    margin = dict(l=0, r=0, t=0, b=45) if is_map else dict(l=12, r=12, t=35, b=12)
    figure.update_layout(margin=margin, legend_title_text="",
                         legend=dict(orientation="h", y=-0.22, x=0, font_size=11), autosize=True)
    figure.for_each_trace(lambda trace: trace.update(
        name=model_label(trace.name) if trace.name in COLORS else trace.name,
    ))
    st.plotly_chart(figure, width="stretch", key=key, config={
        "displaylogo": False, "scrollZoom": False, "displayModeBar": True if is_map else "hover",
    })


def selection(rows: pd.DataFrame, split: str, reference: str, model: str | None = None) -> pd.DataFrame:
    """Select the same comparison across the site and group artifacts."""
    mask = rows["split"].eq(split) & rows["reference"].eq(reference)
    if model is not None:
        mask &= rows["model"].eq(model)
    return rows.loc[mask].copy()


def show_map(sites: pd.DataFrame, model: str, reference: str, split: str) -> None:
    """Map point estimates, interval evidence, and the full pending-site sample."""
    skill = selection(table("skill_site"), split, reference, model)
    st.subheader(headline(skill, model, reference))
    raw = st.toggle("Show raw model MAE (°C)", key="raw_mae")
    rows = map_rows(sites, skill)
    metrics = table("metrics_site")
    metrics = metrics.loc[metrics["split"].eq(split) & metrics["model"].eq(model)]
    rows = rows.merge(metrics[["site_id", "mae"]], on="site_id", how="left", validate="one_to_one")
    value = "mae" if raw else "skill"
    finite = rows.loc[np.isfinite(rows[value])].copy()
    # One scale per reference/split, shared by all learned models for comparable toggling.
    color_limit = skill_color_limit(selection(table("skill_site"), split, reference)["skill"])
    finite["color_value"] = finite[value] if raw else finite[value].clip(-color_limit, color_limit)
    figure = go.Figure()
    palette = map_palette(theme_type())
    if len(finite):
        points = px.scatter_geo(
            finite, lat="lat", lon="lon", color="color_value", symbol="evidence",
            symbol_map={"95% CI excludes zero": "circle", "95% CI includes zero": "diamond",
                        "95% CI unavailable": "square", "Pending / skill unavailable": "square"},
            color_continuous_scale="Cividis" if raw else "RdBu",
            range_color=None if raw else (-color_limit, color_limit),
            custom_data=["name", "country_code", "koppen", "koppen_desc", "mae_model", "mae_ref",
                         "reference_model", "skill_interval", "n", "mae"],
        )
        for trace in points.data:
            trace.marker.size = 9
            trace.marker.opacity = 0.45 if "includes" in trace.name else 0.9
            trace.marker.line = dict(width=0.7, color="#263442")
            trace.hovertemplate = (
                "<b>%{customdata[0]}, %{customdata[1]}</b><br>%{customdata[2]}: %{customdata[3]}"
                "<br>Paired MAE: model %{customdata[4]:.2f} °C / reference %{customdata[5]:.2f} °C"
                "<br>Reference: %{customdata[6]}<br>Skill: %{customdata[7]}"
                "<br>Paired days: %{customdata[8]}<br>Model MAE (all valid pairs): %{customdata[9]:.2f} °C"
                "<extra>%{fullData.name}</extra>"
            )
            figure.add_trace(trace)
        figure.update_layout(coloraxis=points.layout.coloraxis)
    pending = rows.loc[~np.isfinite(rows[value])]
    if len(pending):
        figure.add_trace(go.Scattergeo(
            lat=pending["lat"], lon=pending["lon"],
            text=(pending["name"] + ", " + pending["country_code"] + "<br>"
                  + pending["koppen"] + ": " + pending["koppen_desc"]),
            mode="markers", marker=dict(size=3, opacity=0.4, color="#999999", symbol="circle"),
            name="Pending / unavailable", hovertemplate="%{text}<br>Pending / unavailable<extra></extra>",
        ))
    # Plotly loads world topojson in the viewer's browser; Python reads only precomputed app data.
    figure.update_geos(
        visible=True, projection_type="natural earth", projection_scale=1, fitbounds=False,
        center=dict(lon=0, lat=0), lonaxis_range=[-180, 180], lataxis_range=[-90, 90],
        showland=True, landcolor=palette["land"], showocean=True, oceancolor=palette["ocean"],
        showcountries=True, countrycolor=palette["coast"], countrywidth=0.4,
        showcoastlines=True, coastlinecolor=palette["coast"], coastlinewidth=0.5, showframe=False,
        lataxis_showgrid=False, lonaxis_showgrid=False, bgcolor="rgba(0,0,0,0)",
    )
    figure.update_layout(
        height=600, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(title="MAE (°C)" if raw else "Skill", orientation="h", thickness=12,
                               len=0.75, x=0.5, xanchor="center", y=0, yanchor="top",
                               tickformat=".2f" if raw else ".0%"),
    )
    chart(figure, "world_map")
    st.caption(
        "Diamonds with lower opacity: the 95% interval for ΔMAE includes zero. Circles: it excludes zero; "
        "squares: interval unavailable. Small faint gray dots: pending or undefined for this selection. "
        f"Skill colors span ±{color_limit:.0%}, rounded from the 95th percentile of absolute skill "
        "across models for this reference and split (minimum ±10%). "
        "Only color is clipped; hover shows true values. "
        "Positive skill favors ML. Raw MAE uses all valid truth/model pairs; paired comparisons can differ."
    )


def show_city(sites: pd.DataFrame, manifest: dict) -> None:
    """Explore held-out predictions without calculating or fitting any models."""
    available = sites.loc[sites["results_available"]].sort_values(["name", "country_code"])
    if available.empty:
        st.info("City results are pending.")
        return
    labels = available.set_index("site_id").apply(
        lambda row: f"{row['name']}, {row['country_code']} ({row['koppen']})", axis=1,
    ).to_dict()
    site_id = st.selectbox("Site with results", list(labels), format_func=labels.get, key="city")
    test_start, test_end = manifest["splits"]["test"]
    st.caption(f"Held-out test period: {test_start} to {test_end}.")
    predictions = table("predictions")
    metrics = table("metrics_site")
    metrics = metrics.loc[metrics["site_id"].eq(site_id) & metrics["split"].eq("test")]
    learned = [name for name in model_order(metrics["model"]) if name not in manifest["baselines"]
               and name in predictions.columns]
    models = st.multiselect("ML lines", learned, default=learned, format_func=model_label, key="city_models")
    persistence = st.checkbox("Persistence line", value=False)
    climatology = st.checkbox("Climatology line", value=False)
    models = models + (["persistence"] if persistence else []) + (["climatology"] if climatology else [])
    daily = predictions.loc[predictions["site_id"].eq(site_id) & predictions["split"].eq("test")]
    daily = daily.sort_values("date")
    if daily.empty:
        st.info("Test predictions are pending for this site.")
    else:
        # Reindex local calendar gaps so a missing day is never connected by a line.
        daily = daily.set_index("date").reindex(pd.date_range(test_start, test_end)).rename_axis("date")
        series = daily.reset_index().melt(id_vars="date", value_vars=["y_true", *models],
                                         var_name="series", value_name="Tmax (°C)")
        figure = px.line(series, x="date", y="Tmax (°C)", color="series", color_discrete_map=COLORS)
        figure.for_each_trace(lambda trace: trace.update(
            name="Actual (ERA5)" if trace.name == "y_true" else model_label(trace.name), connectgaps=False,
        ))
        figure.update_xaxes(
            title="Target date", range=[max(pd.Timestamp(test_start), pd.Timestamp(test_end)
                                           - pd.Timedelta(days=119)), pd.Timestamp(test_end)],
            rangeslider=dict(visible=True, thickness=0.08),
            rangeselector=dict(buttons=[
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(label="All", step="all"),
            ], bgcolor=map_palette(theme_type())["land"]),
        )
        chart(figure, "city_forecasts")
    st.subheader("Test metrics · all models")
    display_metrics = metrics[["model", "mae", "rmse", "bias", "n"]].copy()
    display_metrics["model"] = display_metrics["model"].map(model_label)
    display_metrics = display_metrics.rename(columns={
        "model": "Model",
        "mae": "MAE (°C)", "rmse": "RMSE (°C)", "bias": "Bias (°C)", "n": "Valid days",
    }).reset_index(drop=True)
    st.dataframe(
        display_metrics.style.format(dict.fromkeys(["MAE (°C)", "RMSE (°C)", "Bias (°C)"], "{:.2f}"))
        .highlight_min(subset=["MAE (°C)"],
                       props="background-color: rgba(22,139,150,0.25); font-weight: bold;"),
        column_config={name: st.column_config.NumberColumn(format="%.2f")
                       for name in ["MAE (°C)", "RMSE (°C)", "Bias (°C)"]},
        hide_index=True, width="stretch",
    )
    st.caption("The lowest test MAE is highlighted; ties are highlighted together.")
    if len(daily) and models:
        errors = daily[models].sub(daily["y_true"], axis=0).abs()
        errors = errors.where(np.isfinite(errors))
        monthly = errors.groupby(errors.index.month).mean().rename_axis("month").reset_index()
        monthly = monthly.melt(id_vars="month", var_name="model", value_name="MAE (°C)")
        figure = px.line(monthly, x="month", y="MAE (°C)", color="model", markers=True,
                         color_discrete_map=COLORS, labels={"month": "Calendar month", "model": "Model"})
        figure.for_each_trace(lambda trace: trace.update(name=model_label(trace.name)))
        figure.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTH_LABELS)
        chart(figure, "monthly_mae")
        st.caption("MAE by calendar month across test years; available finite truth/prediction pairs only.")
    comparison = selection(table("skill_site"), "test", "best_baseline")
    comparison = comparison.loc[comparison["site_id"].eq(site_id)]
    if comparison.empty:
        st.info("Validation baseline choice and paired test comparisons are pending.")
    else:
        baseline = comparison.iloc[0]["reference_model"]
        sentences = [f"Validation selected {model_label(baseline)} as the better simple rule for this site."]
        for row in comparison.itertuples():
            ci = interval_text(row.delta_mae, row.delta_ci_low, row.delta_ci_high)
            sentences.append(f"{model_label(row.model)} test improvement: {ci} °C (positive favors ML).")
        st.write(" ".join(sentences))


def show_climates(sites: pd.DataFrame, reference: str, split: str) -> None:
    """Show equal-site group summaries separately from the distribution of sites."""
    groups = selection(table("skill_group"), split, reference)
    groups = groups.loc[groups["n_sites"].gt(0) & np.isfinite(groups["skill"])].copy()
    st.subheader("Group skill · equal weight per site")
    if groups.empty:
        st.info("Group results are pending for this selection.")
    else:
        groups["group_name"] = groups["koppen_group"].map(GROUP_LABELS)
        groups["ci_plus"] = (groups["skill_ci_high"] - groups["skill"]).clip(lower=0)
        groups["ci_minus"] = (groups["skill"] - groups["skill_ci_low"]).clip(lower=0)
        figure = px.scatter(groups, x="group_name", y="skill", color="model", symbol="model",
                            error_y="ci_plus", error_y_minus="ci_minus", color_discrete_map=COLORS,
                            hover_data=["n_sites", "skill_ci_low", "skill_ci_high"],
                            labels={"group_name": "Climate group", "model": "Model", "n_sites": "Sites",
                                    "skill_ci_low": "95% CI lower", "skill_ci_high": "95% CI upper"})
        figure.add_hline(y=0, line_dash="dot", line_color="#808080")
        figure.update_yaxes(tickformat=".0%", title="Skill (95% CI)")
        chart(figure, "group_skill")
    st.caption("A: tropical · B: arid · C: temperate · D: continental · E: polar · ALL: all result sites. "
               "Intervals are paired moving-block bootstrap intervals; absent groups are omitted.")
    skill = selection(table("skill_site"), split, reference).merge(
        sites[["site_id", "name", "koppen", "koppen_group"]], on="site_id", validate="many_to_one",
    )
    skill["group_name"] = skill["koppen_group"].map(GROUP_LABELS)
    for category, label in [("koppen_group", "Köppen group"), ("koppen", "Köppen class")]:
        if len(skill):
            x = "group_name" if category == "koppen_group" else category
            figure = px.strip(
                skill, x=x, y="skill", color="model", color_discrete_map=COLORS,
                hover_data=["name", "koppen"], category_orders={x: sorted(skill[x].unique())},
                labels={x: label, "model": "Model", "name": "Site", "koppen": "Köppen class",
                        "skill": "Site skill"},
            )
            figure.add_hline(y=0, line_dash="dot", line_color="#808080")
            figure.update_yaxes(tickformat=".0%", title="Site skill")
            figure.update_xaxes(title=label)
            chart(figure, f"strip_{category}")
    st.subheader("Group MAE (°C) · equal weight per site")
    metrics = table("metrics_site")
    metrics = metrics.loc[metrics["split"].eq(split)].merge(
        sites[["site_id", "koppen_group"]], on="site_id", validate="many_to_one",
    )
    group_mae = metrics.pivot_table(index="koppen_group", columns="model", values="mae", aggfunc="mean")
    group_mae = group_mae.reindex(columns=model_order(metrics["model"]))
    group_mae = group_mae.rename(index=GROUP_LABELS, columns=model_label)
    group_mae.index.name, group_mae.columns.name = "Climate group", "Model"
    st.dataframe(group_mae.style.format("{:.2f}"), width="stretch")
    st.caption("Each MAE uses its model's valid days, then averages sites. "
               "Paired skill comparisons may differ.")
    st.subheader("Feature ablation")
    ablation = table("ablation_site")
    ablation = ablation.loc[ablation["split"].eq(split)].merge(
        sites[["site_id", "koppen_group"]], on="site_id", validate="many_to_one",
    )
    differences = ablation_differences(ablation)
    if differences.empty:
        st.info("Ablation results are pending.")
    else:
        differences = differences.merge(sites[["site_id", "name"]], on="site_id", validate="many_to_one")
        summary = differences.groupby("model", as_index=False).agg(
            difference=("difference", "mean"), n_sites=("site_id", "nunique"),
        )
        axis_labels = {"model": "Model", "difference": "MAE reduction from extra variables (°C)",
                       "n_sites": "Paired sites", "name": "Site"}
        mean_figure = px.bar(summary, x="model", y="difference", color="model", color_discrete_map=COLORS,
                             hover_data=["n_sites"], labels=axis_labels)
        mean_figure.update_xaxes(tickmode="array", tickvals=summary["model"].tolist(),
                                ticktext=[model_label(model) for model in summary["model"]])
        mean_figure.add_hline(y=0, line_color="#808080")
        mean_figure.update_layout(showlegend=False)
        chart(mean_figure, "ablation_mean_difference")
        distribution = px.box(differences, x="model", y="difference", color="model", points="all",
                              color_discrete_map=COLORS, hover_data=["name"], labels=axis_labels)
        distribution.update_xaxes(tickmode="array", tickvals=summary["model"].tolist(),
                                  ticktext=[model_label(model) for model in summary["model"]])
        distribution.add_hline(y=0, line_color="#808080", line_dash="dot")
        distribution.update_layout(showlegend=False)
        chart(distribution, "ablation_site_differences")
        summary["model"] = summary["model"].map(model_label)
        st.dataframe(summary.rename(columns={"model": "Model", "difference": "Mean MAE reduction (°C)",
                                             "n_sites": "Paired sites"}), hide_index=True, width="stretch")
    st.caption("Difference = temp+calendar MAE − full MAE, paired by site and model, then averaged equally "
               "over sites. Positive values mean extra weather variables help. "
               "The box plot shows site spread.")
    st.caption("temp_calendar: lagged temperatures + target calendar; "
               "full: adds humidity, pressure and wind. Each feature group selects its own shared "
               "configuration on validation. Full matches the main models. "
               "This comparison has no exported confidence interval.")


def show_method(sites: pd.DataFrame, manifest: dict) -> None:
    """Explain the evaluation and its operational and sampling limits."""
    start, end = manifest["data_period"]
    st.subheader("Question and data")
    st.write(f"Where is next-day high temperature predictable, and where does ML beat simple rules? "
             f"ERA5 via Open-Meteo covers {start} to {end}. Five daily variables: maximum and minimum "
             "temperature, mean relative humidity, mean sea-level pressure and mean wind speed. "
             "Dates are local calendar days; temperatures are °C.")
    st.write(f"The sample has {len(sites)} cities and research stations, stratified by Köppen class. "
             "Selection prioritizes rare classes, then geographic spread with climate-group quotas, "
             "country caps and distinct ERA5 grid cells. Spacing can relax where coverage requires it. "
             "This is a designed sample, not an area-weighted global estimate.")
    st.subheader("Timing, baselines and models")
    st.write("For target date d, weather predictors use data dated at most d−1; only known calendar facts "
             "about d are allowed. Lags cover the preceding seven days. Climatology, imputers and scalers "
             "are fitted only on the training rows of the current fold.")
    split_rows = [{"Split": name, "Start": dates[0], "End": dates[1]}
                  for name, dates in manifest["splits"].items()]
    st.dataframe(pd.DataFrame(split_rows), hide_index=True, width="stretch")
    st.write("Validation predictions fit on training only. Hyperparameters are chosen on validation, "
             "then models refit on training + validation for the held-out test, evaluated once. "
             "Validation is a tuning diagnostic; it is not independent evidence of generalization. "
             "All splits are chronological, with no time shuffling.")
    st.write("Persistence predicts yesterday's Tmax; climatology is a training-fit seasonal harmonic rule. "
             "The better simple rule is whichever baseline has lower validation MAE on their common finite "
             "dates, fixed for test. Each site has a ridge linear regression and a histogram gradient "
             "boosting model (HGB); configurations are chosen using mean site validation MAE.")
    if "hgb_pooled" in manifest["models"]:
        st.write("HGB pooled (all sites) is an optional single HGB trained jointly across sites. "
                 "It is evaluated with the same chronological splits and paired references.")
    st.subheader("Skill and uncertainty")
    st.write("Skill = 1 − MAE_model / MAE_reference. ΔMAE = MAE_reference − MAE_model, in °C. "
             "Positive values favor ML. Comparisons use common finite truth/model/reference dates; "
             "skill is undefined when reference MAE is zero. Group estimates weight sites equally.")
    st.write(f"95% intervals use a paired moving-block bootstrap with {manifest['bootstrap_block_days']}-day "
             f"blocks and {manifest['bootstrap_reps']:,} replicates. "
             "Group replicates share date blocks across "
             "sites. Missing dates remain in the sampling calendar; undefined replicates are omitted. "
             "Intervals condition on the fitted and selected models, so they exclude model-selection "
             "uncertainty. An interval crossing zero does not establish an improvement.")
    st.subheader("Limits")
    st.write("ERA5 reanalysis is not station truth. This is a retrospective experiment: ERA5 has an "
             "approximately five-day delay, so these results are not an operational forecast. The grid is "
             "roughly 25 km, which smooths terrain and coastal differences. Rare classes have modest "
             "samples: Dsd is missing and Csc has one site. Sites are stratified by climate class, not by "
             "area or population, so group averages are not global averages. Bootstrap intervals do not "
             "remove these limits.")
    st.subheader("Download concurrency benchmark")
    st.dataframe(table("benchmark_concurrency"), hide_index=True, width="stretch")
    st.caption("Measured download timings for the request batch shown; speedups are relative to one thread. "
               "They are not model evaluation timings or a guarantee of future API throughput.")
    st.subheader("Sources and attribution")
    st.markdown(
        "[Weather data by Open-Meteo.com](https://open-meteo.com/) under "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
        "ERA5 / [Copernicus Climate Change Service](https://climate.copernicus.eu/). "
        "Köppen-Geiger: [Beck et al. 2023](https://doi.org/10.1038/s41597-023-02549-6), "
        "1991–2020 classification. Place names: [GeoNames](https://www.geonames.org/), CC BY 4.0. "
        "Map land and boundaries use Plotly's built-in world geography."
    )


def main() -> None:
    """Render one page and keep every data dependency inside the export directory."""
    st.set_page_config(page_title="Skillmap", page_icon="🌡️", layout="wide")
    st.html("""<style>
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background-color: #126E78; color: #FFFFFF;
        }
        @media (max-width: 600px) {
            .stMainBlockContainer {padding-left: 0.75rem; padding-right: 0.75rem;}
        }
    </style>""")
    st.title("Skillmap")
    st.write("Where is next-day high temperature predictable, and where does ML beat simple rules?")
    try:
        manifest_path = DATA / "manifest.json"
        manifest = load_json(str(manifest_path), manifest_path.stat().st_mtime_ns)
        sites = table("sites")
        n = int(sites["results_available"].sum())
        st.caption(f"{'Preliminary · ' if n < len(sites) else ''}{n} of {len(sites)} sites with results · "
                   f"{len(sites) - n} pending · Exported {manifest['generated_at'][:10]}")
        page = st.sidebar.radio("Explore", PAGES, key="page")
        if page in ("Map", "Climate zones"):
            split = st.sidebar.selectbox("Split", ["test", "val"], key="split",
                                         format_func=lambda x: "Test (held out)" if x == "test"
                                         else "Validation (used for tuning)")
            reference = st.sidebar.selectbox("Reference", list(REFERENCE_LABELS), key="reference",
                                             format_func=REFERENCE_LABELS.get)
            if split == "val":
                st.warning("Validation was used for tuning and baseline selection; treat it as a diagnostic.")
        if page == "Map":
            learned = [name for name in model_order(manifest["models"])
                       if name not in manifest["baselines"]]
            model = st.sidebar.selectbox(
                "Model", learned, format_func=model_label, key="model",
                index=learned.index("hgb") if "hgb" in learned else 0,
            )
        if page == "Map":
            show_map(sites, model, reference, split)
        elif page == "City explorer":
            show_city(sites, manifest)
        elif page == "Climate zones":
            show_climates(sites, reference, split)
        else:
            show_method(sites, manifest)
    except FileNotFoundError:
        st.info("The app data bundle is incomplete. Run scripts/export_app_data.py before starting the app.")
    st.divider()
    st.markdown("[Weather data by Open-Meteo.com](https://open-meteo.com/) · "
                "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · ERA5 / Copernicus")


if __name__ == "__main__":
    main()
