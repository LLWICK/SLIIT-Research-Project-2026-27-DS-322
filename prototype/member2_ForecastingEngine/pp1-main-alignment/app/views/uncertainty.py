from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.figures import show_figures
from lib.load import load_predictions, title_case
from lib.scores import coverage_words
from lib.style import AMBER, HARVEST, page_header, pick, show_chart


def render(data: dict[str, Any]) -> None:
    models = data["models"]
    comparison = data["comparison"]
    lgb = next((row for row in comparison if row.get("id") == "lightgbm"), None)
    if lgb is None:
        page_header("Forecast intervals", "4-week test interval", "The LightGBM row is not in the saved comparison file.")
        return
    qhat_by_horizon = models["lightgbm"].get("qhat_by_horizon") or {}
    qhat = qhat_by_horizon.get("4", models["lightgbm"].get("qhat"))
    qhat_1 = qhat_by_horizon.get("1", models["lightgbm"].get("qhat"))
    page_header(
        "Forecast intervals",
        "4-week test interval" if qhat is not None else "Raw 5% and 95% quantiles",
        (
            "The chart is Experiment B LightGBM on the 4-week test. "
            f"The quantiles were adjusted on 2023 only (1-week q̂ = {qhat_1}; 4-week adjustment = {qhat}). "
            f"Measured calibrated coverage is {coverage_words(lgb.get('picp'))}."
        )
        if qhat is not None
        else (
            "These bands are the LightGBM 5th and 95th percentiles on the test weeks. "
            "The 2023 adjustment has not been applied, so this is not a calibrated 90% interval."
        ),
    )
    try:
        loaded = load_predictions()
    except FileNotFoundError:
        loaded = []
    preds = pd.DataFrame(loaded) if loaded else pd.DataFrame()

    picp = lgb.get("picp")
    if picp is None:
        picp_delta = "not scored"
    elif float(picp) >= 90:
        picp_delta = "at or above the 90% intended level"
    else:
        picp_delta = "below the 90% intended level"
    c1, c2, c3 = st.columns(3)
    c1.metric("Calibrated PICP", "not scored" if picp is None else f"{float(picp):.2f}%", picp_delta, delta_color="off")
    c2.metric("Mean width", f"{lgb['interval_width']:.0f}", "LKR per kg", delta_color="off")
    c3.metric(
        "4-week q-hat",
        "not run" if qhat is None else str(qhat),
        "Raw quantiles only" if qhat is None else "Added on 2023 only",
        delta_color="off",
    )

    if preds.empty:
        st.caption("The interval chart is written when Experiment B is trained. The scores above are the saved 4-week test.")
        return
    with st.container(border=True):
        crops = sorted(preds["crop"].unique())
        crop_col, market_col = st.columns(2)
        with crop_col:
            crop = pick("Crop", list(crops), key="unc_crop", preferred="carrot", format_func=title_case)
        markets = sorted(preds.loc[preds["crop"] == crop, "market"].unique())
        with market_col:
            market = pick("Market", list(markets), key="unc_market", preferred="colombo", format_func=title_case)
        slice_df = preds[(preds["crop"] == crop) & (preds["market"] == market)].tail(80)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=slice_df["week_start"], y=slice_df["upper"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=slice_df["week_start"],
                y=slice_df["lower"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(47,107,79,0.16)",
                line=dict(width=0),
                name="Calibrated interval",
            )
        )
        fig.add_trace(go.Scatter(x=slice_df["week_start"], y=slice_df["y_true"], mode="lines", line=dict(color=AMBER, width=2), name="Realised price"))
        fig.add_trace(go.Scatter(x=slice_df["week_start"], y=slice_df["point"], mode="lines", line=dict(color=HARVEST, width=2), name="Median forecast"))
        fig.update_layout(yaxis_title="Rs. / kg")
        show_chart(fig, 420)
        st.caption(
            "Realised wholesale price and the median forecast. "
            + (
                "Band: 5% to 95% after the 2023 adjustment."
                if qhat is not None
                else "Band: raw 5% to 95%. The 2023 adjustment is not applied."
            )
        )
    show_figures("Uncertainty")
