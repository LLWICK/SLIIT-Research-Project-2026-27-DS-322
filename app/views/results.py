from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from lib.style import page_header, section_title


def render(data: dict[str, Any]) -> None:
    evaluation = data["evaluation"]
    rows = pd.DataFrame(evaluation["rows"])
    test_b = rows[rows["experiment"].eq("B")].copy()
    page_header(
        "Test scores",
        "2024–2025, observed prices only",
        "MAE and RMSE are LKR per kg. MAPE is a percent. Coverage is the share of true prices inside the interval. "
        "The nominal interval is 90%, and the calibrated coverage is below that.",
    )

    show = test_b.rename(
        columns={
            "model": "Model",
            "horizon": "Horizon",
            "n_scored": "Rows",
            "mae": "MAE",
            "rmse": "RMSE",
            "mape": "MAPE %",
            "pinball": "Pinball",
            "picp": "PICP raw %",
            "picp_cqr": "PICP calibrated %",
            "interval_width_cqr": "Width calibrated",
        }
    )
    with st.container(border=True):
        section_title("Experiment B", "Price, season, origin weather, macros, and lagged extent")
        st.dataframe(
            show[
                [
                    "Model",
                    "Horizon",
                    "Rows",
                    "MAE",
                    "RMSE",
                    "MAPE %",
                    "Pinball",
                    "PICP raw %",
                    "PICP calibrated %",
                    "Width calibrated",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    a = rows[rows["experiment"].eq("A") & rows["model"].isin(["lightgbm", "xgboost"])][
        ["model", "horizon", "mae", "rmse", "mape"]
    ].rename(columns={"mae": "MAE A", "rmse": "RMSE A", "mape": "MAPE A"})
    b = test_b[test_b["model"].isin(["lightgbm", "xgboost"])][["model", "horizon", "mae", "rmse", "mape"]].rename(
        columns={"mae": "MAE B", "rmse": "RMSE B", "mape": "MAPE B"}
    )
    compared = a.merge(b, on=["model", "horizon"])
    compared["MAE change %"] = ((compared["MAE B"] - compared["MAE A"]) / compared["MAE A"] * 100).round(2)
    with st.container(border=True):
        section_title("Evaluation", "A against B, same test rows")
        st.dataframe(compared, width="stretch", hide_index=True)
        st.caption(
            "A negative MAE change means B is more accurate. Experiment C is absent because the monthly "
            "target and achieved file was not available. SARIMAX is not in this comparison because it does not use the extra columns."
        )
