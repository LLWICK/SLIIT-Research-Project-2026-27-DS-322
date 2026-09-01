from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
        "Panel construction",
        "The weekly panel, as the audit left it",
        f"{meta['n_panel']:,} wholesale weekly rows after filters. Member 1’s analysis-ready extract "
        "already keeps observed prices; we still refuse to zero-fill, and we score only those observed rows.",
    )

    card_grid(
        [
            {
                "role": name.upper(),
                "title": f"{split[name]['n']} rows",
                "body": f"{split[name]['start']}  →  {split[name]['end']}",
                "active": name == "test",
            }
            for name in ("train", "calibration", "test")
        ],
        columns=3,
    )

    if upstream.get("found"):
        st.info(
            f"Member 1 extract detected: {upstream['wholesale_rows']:,} wholesale rows "
            f"({upstream['scoped_rows']:,} in Carrot / Leek / Tomato at Colombo, Badulla, Nuwara Eliya). "
            "Charts below use the frozen demo panel so scored metrics stay reproducible."
        )

    with st.container(border=True):
        crops = sorted({row["crop"] for row in series})
        crop_col, market_col = st.columns(2)
        with crop_col:
            crop = pick("Crop", crops, key="data_crop", preferred="carrot", format_func=title_case)
        markets = sorted({row["market"] for row in series if row["crop"] == crop})
        with market_col:
            market = pick("Market", markets, key="data_market", preferred="colombo", format_func=title_case)
        chosen = next(row for row in series if row["crop"] == crop and row["market"] == market)
        points = [p for p in chosen["points"] if p.get("price") is not None]
        price_df = pd.DataFrame(points)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_df["week"], y=price_df["price"], mode="lines", line=dict(color=HARVEST, width=2), name="Price"))
        fig.update_layout(yaxis_title="Rs. / kg", showlegend=False)
        show_chart(fig, 380)
        st.caption("Spikes in 2020–2021 are flagged as disruption, not interpolated.")

    money, rules = st.columns(2)
    cob = [row for row in cobweb if row["crop"] == crop and row["market"] == market]
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
            scatter.update_layout(xaxis_title="Intensity (season t)", yaxis_title="Price next season")
            show_chart(scatter, 320)
            st.caption("Planting pressure this season against the next season’s market price.")

    with rules:
        with st.container(border=True):
            section_title("Data rules", "How this panel is treated")
            for item in coverage.get("exclusions", []):
                st.markdown(f"**{item['item']}**  \n{item['reason']}")

    with st.container(border=True):
        section_title("Coverage", "Wholesale series used in the model")
        cov_df = pd.DataFrame(coverage["by_series"])
        cov_df["crop"] = cov_df["crop"].map(title_case)
        cov_df["market"] = cov_df["market"].map(title_case)
        cov_df["start"] = cov_df["start"].astype(str).str.slice(0, 10)
        cov_df["end"] = cov_df["end"].astype(str).str.slice(0, 10)
        st.dataframe(
            cov_df.rename(columns={"coverage_pct": "coverage %"})[["crop", "market", "rows", "coverage %", "start", "end"]],
            width="stretch",
            hide_index=True,
        )
