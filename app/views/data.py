from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.cultivation_status import render_cultivation_status
from lib.figures import show_figures
from lib.experiment_c import experiment_c
from lib.load import member1_status, title_case
from lib.style import AMBER, HARVEST, card_grid, page_header, pick, section_title, show_chart


def render(data: dict[str, Any]) -> None:
    coverage = data["coverage"]
    series = data["series"]
    cobweb = data["cobweb"]
    split = data["split"]
    meta = data["meta"]
    upstream = member1_status()

    page_header(
        "Panel",
        "Wholesale weeks that were scored",
        f"{meta['n_panel']:,} wholesale weekly rows after the filters. "
        "Missing prices stay missing. Only weeks with a published wholesale price are scored.",
    )

    card_grid(
        [
            {
                "title": name.title(),
                "body": f"{split[name]['n']} rows · {split[name]['start']} to {split[name]['end']}",
                "active": name == "test",
            }
            for name in ("train", "calibration", "test")
        ],
        columns=3,
    )

    if upstream.get("found"):
        st.caption(
            f"An older Member 1 extract is also on disk ({upstream['wholesale_rows']:,} wholesale rows). "
            "The chart on this page uses the real HARTI panel scored for this run."
        )

    render_cultivation_status()

    crop = None
    market = None
    with st.container(border=True):
        crops = sorted({row["crop"] for row in series})
        if not crops:
            st.caption("No price series is stored for this page.")
        else:
            crop_col, market_col = st.columns(2)
            with crop_col:
                crop = pick("Crop", crops, key="data_crop", preferred="carrot", format_func=title_case)
            markets = sorted({row["market"] for row in series if row["crop"] == crop})
            with market_col:
                market = pick("Market", markets, key="data_market", preferred="colombo", format_func=title_case) if markets else None
            chosen = next((row for row in series if row["crop"] == crop and row["market"] == market), None)
            if chosen is None:
                st.caption("No series is stored for this crop and market.")
            else:
                points = [p for p in chosen["points"] if p.get("price") is not None]
                price_df = pd.DataFrame(points)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=price_df["week"], y=price_df["price"], mode="lines", line=dict(color=HARVEST, width=2), name="Price"))
                fig.update_layout(yaxis_title="Rs. / kg", showlegend=False)
                show_chart(fig, 380)
                st.caption("Spikes in 2020–2021 are flagged as disruption, not interpolated.")

    money, rules = st.columns(2)
    cob = [] if crop is None or market is None else [row for row in cobweb if row["crop"] == crop and row["market"] == market]
    with money:
        with st.container(border=True):
            section_title("Money plot", "Season t intensity vs season t+1 price")
            cob_df = pd.DataFrame(cob)
            scatter = go.Figure()
            if not cob_df.empty:
                scatter.add_trace(
                    go.Scatter(
                        x=cob_df["intensity_t"],
                        y=cob_df["price_t1"],
                        mode="markers",
                        marker=dict(color=AMBER, size=8),
                        name="Season pairs",
                    )
                )
            scatter.update_layout(xaxis_title="Seasonal extent ratio", yaxis_title="Price next season")
            show_chart(scatter, 320)
            st.caption(
                "Each point is a finished season’s extent ratio against the next season’s average price. "
                "This is not current achieved hectares divided by target hectares."
            )

    with rules:
        with st.container(border=True):
            section_title("Data rules", "How this panel is treated")
            trained = bool(experiment_c().get("trained"))
            for item in coverage.get("exclusions", []):
                reason = item["reason"]
                if item.get("item") == "Live commitments" and trained:
                    reason = (
                        "processed/doa_cultivation_progress.csv is a national progress file, not a district-month series. "
                        "Experiment C was trained on the national as-of signal. District rows stay in the audit. "
                        "Lagged finished-season extent remains a historical feature in the full-sample Experiment B run, "
                        "not current cultivation progress."
                    )
                st.markdown(f"**{item['item']}**  \n{reason}")

    with st.container(border=True):
        section_title("Coverage", "Wholesale series used in the model")
        cov_df = pd.DataFrame(coverage["by_series"])
        cov_df["crop"] = cov_df["crop"].map(title_case)
        cov_df["market"] = cov_df["market"].map(title_case)
        cov_df["start"] = cov_df["start"].astype(str).str.slice(0, 10)
        cov_df["end"] = cov_df["end"].astype(str).str.slice(0, 10)
        st.dataframe(
            cov_df.rename(columns={"coverage_pct": "price weeks %"})[["crop", "market", "rows", "price weeks %", "start", "end"]],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "This percentage is how many weeks in the series have a published wholesale price. "
            "It is not the forecast interval, and it is not the 90% coverage target."
        )
    _missing_share_chart(series)
    show_figures("Data", skip=("wholesale_missing_share_by_year.png",))


def _missing_share_chart(series: list[dict]) -> None:
    rows = []
    for item in series:
        for point in item.get("points") or []:
            week = str(point.get("week") or "")
            if len(week) < 4 or not week[:4].isdigit():
                continue
            rows.append({"year": week[:4], "missing": point.get("price") is None})
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.caption("Missing share by year is not in the stored price series.")
        return
    grouped = frame.groupby("year", as_index=False)["missing"].mean()
    if float(grouped["missing"].sum()) == 0:
        st.caption("Every stored week has a price, so there is no missing-share bar to draw.")
        return
    fig = go.Figure(
        go.Bar(
            x=grouped["year"],
            y=grouped["missing"],
            marker_color=HARVEST,
            name="Share missing",
        )
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Share missing",
        showlegend=False,
        yaxis=dict(rangemode="tozero", tickformat=".0%"),
    )
    show_chart(fig, 320)
    st.caption(
        "Missing share of wholesale prices by year. A week counts as missing when the stored price is blank."
    )
