from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.experiment_c import render_common_key_benchmark
from lib.figures import show_figures
from lib.load import title_case
from lib.scores import b4_coverage_sentence, find_row
from lib.style import AMBER, HARVEST, SKY, page_header, section_title, show_chart

REPO = Path(__file__).resolve().parents[2]


def _best_sentence(table: pd.DataFrame) -> str:
    if table.empty:
        return "The common-key best-model table has not been written yet."
    parts = []
    for row in table.itertuples(index=False):
        parts.append(
            f"{int(row.horizon)} week{'s' if int(row.horizon) != 1 else ''}: {row.model} "
            f"(MAE {float(row.mae):.2f}, MAPE {float(row.mape):.2f}%, n={int(row.n_scored)})"
        )
    return "Lowest MAE on the common-key rows, by horizon — " + "; ".join(parts) + "."


def render(data: dict[str, Any]) -> None:
    comparison = data["comparison"]
    models = data["models"]
    lgb = models["lightgbm"]
    evaluation = data["evaluation"]
    best_path = REPO / "outputs" / "evaluation" / "best_model_by_horizon.csv"
    best = pd.read_csv(best_path) if best_path.is_file() else pd.DataFrame()

    page_header(
        "Which model is best",
        "Common keys, one horizon at a time",
        _best_sentence(best) + " " + b4_coverage_sentence(evaluation),
    )

    with st.container(border=True):
        section_title("Best at each horizon", "Saved common-key table from this run")
        if best.empty:
            st.info("outputs/evaluation/best_model_by_horizon.csv is missing.")
        else:
            st.dataframe(best, width="stretch", hide_index=True)
            st.caption(
                "This table names the lowest MAE on the rows shared by LightGBM, XGBoost, SARIMAX, and the last-value naive forecast. "
                "It does not name one model for every horizon."
            )

    last = next(row for row in comparison if row["id"] == "naive_last")
    seasonal = next(row for row in comparison if row["id"] == "seasonal_naive_52")
    h4_x = find_row(evaluation, "B", "xgboost", 4)
    h4_l = find_row(evaluation, "B", "lightgbm", 4)
    h4_a = find_row(evaluation, "A", "lightgbm", 4)
    with st.container(border=True):
        section_title(
            "4-week test",
            f"Last-value uses the same {last['n_scored']:,} rows as Experiment B LightGBM. "
            f"Seasonal naive uses {seasonal['n_scored']:,} of those rows, the ones with a published price 52 weeks before the target. "
            "LSTM was not run.",
        )
        if h4_x and h4_l and h4_a:
            st.caption(
                f"On the full sample, XGBoost B MAE is {float(h4_x['mae']):.2f} and LightGBM B MAE is {float(h4_l['mae']):.2f}. "
                f"LightGBM A MAE is {float(h4_a['mae']):.2f}. "
                f"Last-value naive MAE is {float(last['mae']):.2f}, MAPE {float(last['mape']):.2f}%, n={int(last['n_scored'])}."
            )
        table = pd.DataFrame(
            [
                {
                    "Model": row["model"],
                    "Rows": row["n_scored"],
                    "MAE": row["mae"],
                    "RMSE": row["rmse"],
                    "MAPE %": row["mape"],
                    "Pinball": row["pinball"],
                    "PICP calibrated %": row["picp"],
                    "Width": row["interval_width"],
                }
                for row in comparison
            ]
        )
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption(
            "PICP in this table is the calibrated interval. A value below 90 is below the intended level. "
            "Naive forecasts have no interval."
        )

    scored = [row for row in comparison if row.get("mae") is not None]
    mae_col, picp_col = st.columns(2)
    ids = [row["id"] for row in scored]
    with mae_col:
        with st.container(border=True):
            section_title("Point accuracy", "MAE (LKR/kg). Unscored models are omitted.")
            fig = go.Figure(go.Bar(x=ids, y=[row["mae"] for row in scored], marker_color=HARVEST, name="MAE"))
            fig.update_layout(showlegend=False, yaxis_title="Rs. / kg")
            show_chart(fig, 300)
    intervals = [row for row in scored if row.get("picp") is not None]
    with picp_col:
        with st.container(border=True):
            section_title("Uncertainty quality", "PICP vs interval width. Naive forecasts have no interval.")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=[row["id"] for row in intervals], y=[row["picp"] for row in intervals], marker_color=SKY, name="PICP"))
            fig2.add_trace(
                go.Bar(
                    x=[row["id"] for row in intervals],
                    y=[row["interval_width"] for row in intervals],
                    marker_color=AMBER,
                    name="Width",
                )
            )
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
    render_common_key_benchmark()
    show_figures("Comparison")


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
