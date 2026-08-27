from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.load import lkr, title_case
from lib.style import AMBER, HARVEST, SKY, page_header, section_title, show_chart


def render(data: dict[str, Any]) -> None:
    comparison = data["comparison"]
    models = data["models"]
    baseline = comparison[0]
    lgb_row = next(row for row in comparison if row["id"] == "lightgbm")
    lgb = models["lightgbm"]

    page_header(
        "Same folds · same seed · observed only",
        "Did we beat the baseline?",
        f"Yes. LightGBM quantile + CQR cuts MAE from {lkr(baseline['mae'])} to {lkr(lgb_row['mae'])} "
        f"and lifts PICP from {baseline['picp']}% to {lgb_row['picp']}%.",
    )

    with st.container(border=True):
        section_title("PP2 table", "Headline comparison")
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
            fig = go.Figure(go.Bar(x=ids, y=[row["mae"] for row in comparison], marker_color=HARVEST, name="MAE"))
            fig.update_layout(showlegend=False, yaxis_title="Rs. / kg")
            show_chart(fig, 300)
    with picp_col:
        with st.container(border=True):
            section_title("Uncertainty quality", "PICP vs interval width")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=ids, y=[row["picp"] for row in comparison], marker_color=SKY, name="PICP"))
            fig2.add_trace(go.Bar(x=ids, y=[row["interval_width"] for row in comparison], marker_color=AMBER, name="Width"))
            fig2.update_layout(barmode="group")
            show_chart(fig2, 300)

    with st.container(border=True):
        section_title("LightGBM slices", "Per crop and market")
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
