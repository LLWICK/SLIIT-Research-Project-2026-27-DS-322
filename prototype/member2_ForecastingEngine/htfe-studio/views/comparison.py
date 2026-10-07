from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.load import title_case
from lib.style import AMBER, HARVEST, SKY, page_header, section_title, show_chart


def render(data: dict[str, Any]) -> None:
    comparison = data["comparison"]
    models = data["models"]
    lgb = models["lightgbm"]

    page_header(
        "Same test weeks · observed prices only",
        "Horizon changes which model is ahead",
        "At 1 week, SARIMAX has the lower error. From 4 weeks, Experiment B LightGBM is ahead of SARIMAX. "
        "Calibrated interval coverage is still below the nominal 90%.",
    )

    with st.container(border=True):
        section_title("4-week test", "Experiment B. Naive rows are an earlier pooled run and are not the same row set.")
        table = pd.DataFrame(
            [
                {
                    "Model": row["model"],
                    "MAE": row["mae"],
                    "RMSE": row["rmse"],
                    "MAPE %": row["mape"],
                    "Pinball": row["pinball"],
                    "PICP %": row["picp"],
                    "Width": row["interval_width"],
                    "Backend": row["backend"],
                }
                for row in comparison
            ]
        )
        st.dataframe(table, width="stretch", hide_index=True)

    mae_col, picp_col = st.columns(2)
    ids = [row["id"] for row in comparison]
    with mae_col:
        with st.container(border=True):
            section_title("Point accuracy", "MAE (LKR/kg)")
            fig = go.Figure(go.Bar(x=ids, y=[row["mae"] or 0 for row in comparison], marker_color=HARVEST, name="MAE"))
            fig.update_layout(showlegend=False, yaxis_title="Rs. / kg")
            show_chart(fig, 300)
    with picp_col:
        with st.container(border=True):
            section_title("Uncertainty quality", "PICP vs interval width")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=ids, y=[row["picp"] or 0 for row in comparison], marker_color=SKY, name="PICP"))
            fig2.add_trace(go.Bar(x=ids, y=[row["interval_width"] or 0 for row in comparison], marker_color=AMBER, name="Width"))
            fig2.update_layout(barmode="group")
            show_chart(fig2, 300)

    with st.container(border=True):
        section_title("LightGBM slices", "Experiment B, 4-week test, same scored rows")
        crop_col, market_col = st.columns(2)
        with crop_col:
            st.caption("By crop")
            st.dataframe(_slice_table(lgb["metrics"].get("by_crop", {})), width="stretch", hide_index=True)
        with market_col:
            st.caption("By market")
            st.dataframe(_slice_table(lgb["metrics"].get("by_market", {})), width="stretch", hide_index=True)


def _slice_table(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Slice": title_case(key),
                "MAE": value["mae"],
                "MAPE %": value["mape"],
                "PICP %": value["picp"],
                "Width": value["interval_width"],
            }
            for key, value in rows.items()
        ]
    )
