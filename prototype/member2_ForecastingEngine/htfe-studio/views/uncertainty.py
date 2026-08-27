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
    preds = pd.DataFrame(load_predictions())

    page_header(
        "Chronological CQR",
        "90% targeted coverage, empirically evaluated",
        "We do not claim an i.i.d. mathematical guarantee on agricultural prices. "
        f"We conformalize the 5th/95th LightGBM quantiles on a contiguous calibration block "
        f"(q̂ = {models['lightgbm']['qhat']}) and then measure PICP on the later test weeks.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Empirical PICP", f"{lgb['picp']}%", "Target 90%")
    c2.metric("Mean width (Rs.)", f"{lgb['interval_width']:.0f}", "Sharpness half of the result")
    c3.metric("CQR q-hat", str(models["lightgbm"]["qhat"]), "Added to both tails")

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
                fillcolor="rgba(198,243,26,0.18)",
                line=dict(width=0),
                name="90% interval",
            )
        )
        fig.add_trace(go.Scatter(x=slice_df["week_start"], y=slice_df["y_true"], mode="lines", line=dict(color=AMBER, width=2), name="Realised price"))
        fig.add_trace(go.Scatter(x=slice_df["week_start"], y=slice_df["point"], mode="lines", line=dict(color=HARVEST, width=2), name="Median forecast"))
        fig.update_layout(yaxis_title="Rs. / kg")
        show_chart(fig, 420)
        st.caption("Amber = realised wholesale price. Lime = median forecast. Band = calibrated 90% interval.")
