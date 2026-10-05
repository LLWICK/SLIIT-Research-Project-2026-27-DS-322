from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.load import title_case
from lib.style import HARVEST, page_header, section_title, show_chart


def render(data: dict[str, Any]) -> None:
    origin_map = data["originMap"]
    dictionary = data["dictionary"]
    importance = data["importance"]

    page_header(
        "Leak-safe features",
        "Markets observe prices. Districts grow crops.",
        "Weather and commitments never join on the market name. They are aggregated from the origin map, "
        "and intensity is as-of-week: a registration after the forecast week cannot enter the feature.",
    )

    section_title("Origin map", "Supply districts behind each modelled market")
    cols = st.columns(3, gap="small")
    for col, (crop, markets) in zip(cols, origin_map.items()):
        with col:
            with st.container(border=True):
                st.caption("CROP")
                st.markdown(f"**{title_case(crop)}**")
                for market, districts in markets.items():
                    st.caption(
                        f"{title_case(market)} ← {', '.join(title_case(d) for d in districts)}"
                    )

    formula, bars = st.columns(2)
    with formula:
        with st.container(border=True):
            section_title("Novelty feature", "Cultivation progress")
            st.code(
                "progress(market, crop, week) =\n"
                "  sum of achieved hectares in supply districts\n"
                "  already known by this week\n"
                "  ÷ sum of target hectares in those districts",
                language="text",
            )
            st.caption(
                "That monthly district table was not in the data folder, so this run does not treat "
                "seasonal Census extent as cultivation progress."
            )

    with bars:
        with st.container(border=True):
            section_title("Model columns", "What this LightGBM run uses")
            top = importance[:6]
            peak = top[0]["importance"] or 1
            fig = go.Figure(
                go.Bar(
                    x=[row["importance"] for row in reversed(top)],
                    y=[row["feature"] for row in reversed(top)],
                    orientation="h",
                    marker_color=HARVEST,
                    name="Importance",
                )
            )
            fig.update_layout(showlegend=False, xaxis_title="Permutation importance")
            show_chart(fig, 280)
            for row in top:
                st.caption(f"{row['feature']}  ·  {row['importance']:.2f}  ({100 * row['importance'] / peak:.0f}% of lag-1)")
            st.caption(
                "These are SHAP values for the Experiment B 4-week median model. "
                "Cultivation progress is not in this model, so a high rank here is not evidence for that novelty."
            )

    with st.container(border=True):
        section_title("Generated dictionary", "Panel contract")
        st.dataframe(pd.DataFrame(dictionary), width="stretch", hide_index=True)
