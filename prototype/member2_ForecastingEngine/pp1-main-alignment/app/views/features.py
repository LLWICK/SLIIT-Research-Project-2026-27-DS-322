from __future__ import annotations

from pathlib import Path
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
        "Features",
        "Weather comes from the supply district",
        "A market price is not joined to weather at the market town. "
        "A cultivation report is used only when its date is on or before the forecast Monday.",
    )

    section_title("Origin map", "Supply districts behind each modelled market")
    cols = st.columns(3, gap="small")
    for col, (crop, markets) in zip(cols, origin_map.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title_case(crop)}**")
                for market, districts in markets.items():
                    st.caption(
                        f"{title_case(market)} ← {', '.join(title_case(d) for d in districts)}"
                    )

    formula, bars = st.columns(2)
    with formula:
        with st.container(border=True):
            section_title("As-of ratio", "Cultivation progress")
            st.code(
                "progress(crop, week) =\n"
                "  national achieved hectares already known by this week\n"
                "  ÷ national target hectares",
                language="text",
            )
            st.caption(
                "processed/doa_cultivation_progress.csv is the national progress file. "
                "Experiment C uses a report only when the crop and official season match and the report date is on or before the forecast Monday. "
                "District rows stay in the audit. "
                "The dashboard is the runtime model. It is not this Experiment C score."
            )

    with bars:
        with st.container(border=True):
            section_title("Model columns", "What this LightGBM run uses")
            top = list(importance or [])[:6]
            if not top:
                st.caption("The stored importance file has no rows.")
            else:
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
        section_title("Column dictionary", "Panel contract")
        st.dataframe(pd.DataFrame(dictionary), width="stretch", hide_index=True)
    show_figures("Features", skip=("shap_cultivation_progress_h4.png",))
    st.caption(_cultivation_shap_sentence())


def _cultivation_shap_sentence() -> str:
    path = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "experiment_c"
        / "evaluation"
        / "shap_lightgbm_c_h4.csv"
    )
    if not path.is_file():
        return "The cultivation-progress SHAP figure is an empty frame, and the SHAP table is not on disk."
    frame = pd.read_csv(path)
    match = frame.loc[frame["feature"].astype(str).eq("cultivation_progress_ratio_real")]
    if match.empty:
        return (
            "The cultivation-progress SHAP figure is an empty frame. "
            "That column is not in the stored SHAP table."
        )
    value = float(match["mean_abs_shap"].iloc[0])
    if value == 0:
        return (
            "Mean absolute SHAP for cultivation_progress_ratio_real on the horizon-4 model is 0, "
            "so that chart has no bar."
        )
    return (
        f"Mean absolute SHAP for cultivation_progress_ratio_real on the horizon-4 model is {value:.4f}."
    )
