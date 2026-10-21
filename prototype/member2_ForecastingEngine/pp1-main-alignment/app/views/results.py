from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from lib.figures import show_figures
from lib.scores import b4_coverage_sentence, find_row
from lib.style import page_header, section_title


def _line(row: dict | None, label: str) -> str:
    if row is None:
        return f"{label}: not scored on this run."
    raw = row.get("picp")
    calibrated = row.get("picp_cqr")
    coverage = ""
    if raw is not None or calibrated is not None:
        raw_text = "not scored" if raw is None else f"{float(raw):.2f}%"
        cal_text = "not scored" if calibrated is None else f"{float(calibrated):.2f}%"
        coverage = f" Raw 5–95% coverage {raw_text}. Calibrated coverage {cal_text}."
    return (
        f"{label}: MAE {float(row['mae']):.2f}, MAPE {float(row['mape']):.2f}%, "
        f"n={int(row['n_scored'])}.{coverage}"
    )


def render(data: dict[str, Any]) -> None:
    evaluation = data["evaluation"]
    rows = pd.DataFrame(evaluation["rows"])
    test_b = rows[rows["experiment"].eq("B")].copy()
    page_header(
        "Results",
        "2024–2025 full sample, observed prices only",
        "MAE is LKR per kg. MAPE is a percent. They are separate numbers. "
        + b4_coverage_sentence(evaluation)
        + " The matched-row cultivation test is on Cultivation proof, not in this table.",
    )
    with st.container(border=True):
        section_title("This run", "Full-sample rows from the saved evaluation file")
        for sentence in (
            _line(find_row(evaluation, "B", "lightgbm", 4), "LightGBM B, 4 weeks"),
            _line(find_row(evaluation, "A", "lightgbm", 4), "LightGBM A, 4 weeks"),
            _line(find_row(evaluation, "B", "sarimax", 1), "SARIMAX, 1 week"),
            _line(find_row(evaluation, "B", "xgboost", 4), "XGBoost B, 4 weeks"),
        ):
            st.markdown(sentence)
        st.caption(
            "Raw coverage is the unadjusted 5–95% interval. Calibrated coverage applies the 2023 qhat. "
            "A share below 90% is below the intended level."
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
        section_title("Full-sample A/B run", "A against B, same full test rows")
        st.dataframe(compared, width="stretch", hide_index=True)
        st.caption(
            "This table is the full-sample Experiment A and Experiment B run. "
            "A negative MAE change means B is more accurate on those rows. "
            "Experiment C is not in this table. "
            "SARIMAX is not in this comparison because it does not use the extra columns. "
            "Which model is lowest at each horizon is on the Which model page."
        )
    show_figures("Results")
