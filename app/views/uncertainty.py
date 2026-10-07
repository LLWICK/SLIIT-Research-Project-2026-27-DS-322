from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.load import load_predictions, title_case
from lib.style import AMBER, HARVEST, page_header, pick, show_chart


def render(data: dict[str, Any]) -> None:
    models = data["models"]
    comparison = data["comparison"]
    lgb = next(row for row in comparison if row["id"] == "lightgbm")
    qhat_by_horizon = models["lightgbm"].get("qhat_by_horizon") or {}
    qhat = qhat_by_horizon.get("4", models["lightgbm"].get("qhat"))
    qhat_1 = qhat_by_horizon.get("1", models["lightgbm"].get("qhat"))
    page_header(
        "Forecast intervals",
        "Raw 5% and 95% quantiles" if qhat is None else "90% targeted coverage, empirically evaluated",
        "These bands are the LightGBM 5th and 95th percentiles on the 1-week-ahead test weeks. "
        "They are not conformal yet, so the coverage is descriptive rather than a calibrated 90% guarantee."
        if qhat is None
        else (
            "We do not claim an i.i.d. mathematical guarantee on agricultural prices. "
            "The chart is the Experiment B LightGBM 4-week test. "
            f"The quantiles were conformalized on 2023 only (1-week q̂ = {qhat_1}; the 4-week adjustment is {qhat}). "
            "The PICP shown here is that calibrated 4-week test, and it is below 90%."
        ),
    )
    try:
        loaded = load_predictions()
    except FileNotFoundError:
        loaded = []
    preds = pd.DataFrame(loaded) if loaded else pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    c1.metric("Empirical PICP", f"{lgb['picp']}%", "Below the 90% target")
    c2.metric("Mean width (Rs.)", f"{lgb['interval_width']:.0f}", "Sharpness half of the result")
    c3.metric("CQR q-hat", "not run" if qhat is None else str(qhat), "Raw quantiles only" if qhat is None else "Added to both tails")

    if preds.empty:
        st.info("The interval chart is written when Experiment B is trained. The scores above are the saved 4-week test.")
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
                fillcolor="rgba(37,99,235,0.16)",
                line=dict(width=0),
                name="Calibrated interval",
            )
        )
        fig.add_trace(go.Scatter(x=slice_df["week_start"], y=slice_df["y_true"], mode="lines", line=dict(color=AMBER, width=2), name="Realised price"))
        fig.add_trace(go.Scatter(x=slice_df["week_start"], y=slice_df["point"], mode="lines", line=dict(color=HARVEST, width=2), name="Median forecast"))
        fig.update_layout(yaxis_title="Rs. / kg")
        show_chart(fig, 420)
        st.caption(
            "Amber = realised wholesale price. Blue = median forecast. "
            + ("Band = split conformal interval around the LightGBM quantiles." if qhat is not None else "Band = raw 5% to 95% quantile, not conformal.")
        )
