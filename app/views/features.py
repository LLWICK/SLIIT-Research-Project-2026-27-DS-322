from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.figures import show_figures
from lib.load import title_case
from lib.style import HARVEST, page_header, section_title, show_chart


def render(data: dict[str, Any]) -> None:
    origin_map = data["originMap"]
    dictionary = data["dictionary"]
    importance = data["importance"]

    page_header(
        "Leak-safe features",
        "Markets observe prices. Districts grow crops.",
        "Weather is taken from the supply districts in the origin map, not from the market town. "
        "A cultivation record is used only when its date is on or before the forecast Monday.",
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
                "processed/doa_cultivation_progress.csv is the national progress file. "
                "Experiment C uses a report only when the crop and official season match and the report date is on or before the forecast Monday. "
                "District rows are audit-only. The dashboard model does not use this column. "
                "Seasonal Census extent is lagged history, not current progress."
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
            fig.update_layout(showlegend=False, xaxis_title="Mean absolute SHAP")
            show_chart(fig, 280)
            for row in top:
                st.caption(f"{row['feature']}  ·  {row['importance']:.2f}  ({100 * row['importance'] / peak:.0f}% of lag-1)")
            st.caption(
                "The bars are the stored importance file. "
                "The figure below, when present, is this run's LightGBM C horizon-4 SHAP. "
                "SHAP does not replace the matched-row B* versus C MAE test."
            )

    with st.container(border=True):
        section_title("Generated dictionary", "Panel contract")
        st.dataframe(pd.DataFrame(dictionary), width="stretch", hide_index=True)
    show_figures("Features")
