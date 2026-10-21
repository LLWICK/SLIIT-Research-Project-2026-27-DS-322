from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.experiment_c import parity, render_common_key_benchmark
from lib.figures import show_figures
from lib.load import title_case
from lib.scores import b4_coverage_sentence, find_row
from lib.style import AMBER, HARVEST, SKY, facts, page_header, section_title, show_chart

_MODEL_LABEL = {
    "sarimax": "SARIMAX",
    "sarima": "SARIMAX",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "naive_last": "Last-value",
    "seasonal_naive_52": "Seasonal naive",
}


def _model_label(value: object) -> str:
    text = str(value).strip()
    return _MODEL_LABEL.get(text.casefold(), text)


def _top_decision(table: pd.DataFrame, ready: bool) -> str:
    if not ready:
        return "The saved best-model table is not on disk."
    picked: dict[int, str] = {}
    for horizon in (1, 4):
        rows = table.loc[table["horizon"].astype(int).eq(horizon)]
        if rows.empty:
            continue
        picked[horizon] = _model_label(rows.iloc[0]["model"])
    if 1 in picked and 4 in picked:
        return (
            f"{picked[1]} is best at 1 week, {picked[4]} is best at 4 weeks, "
            "and cultivation is a separate test."
        )
    return "The saved best-model table does not list both 1 week and 4 weeks. Cultivation is a separate test."


def _by_id(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("id") == row_id), None)


def render(data: dict[str, Any]) -> None:
    comparison = data["comparison"]
    models = data["models"]
    lgb = models["lightgbm"]
    evaluation = data["evaluation"]
    best = parity().get("best")
    if not isinstance(best, pd.DataFrame):
        best = pd.DataFrame()
    ready = not best.empty and {"horizon", "model", "mae", "n_scored"}.issubset(best.columns)

    page_header(
        "Which model",
        "One horizon at a time",
        _top_decision(best, ready),
    )
    st.caption(b4_coverage_sentence(evaluation))

    with st.container(border=True):
        section_title("Final decision", "Common-key rows from this run")
        if not ready:
            st.caption("outputs/evaluation/best_model_by_horizon.csv is missing.")
        else:
            facts(
                [
                    (
                        f"{int(row.horizon)} week" if int(row.horizon) == 1 else f"{int(row.horizon)} weeks",
                        f"{_model_label(row.model)}, MAE {float(row.mae):.2f}, n={int(row.n_scored)}",
                    )
                    for row in best.itertuples(index=False)
                ],
                ready=True,
            )
            st.caption("The lowest model changes with the horizon.")
            st.dataframe(best, width="stretch", hide_index=True)

    last = _by_id(comparison, "naive_last")
    seasonal = _by_id(comparison, "seasonal_naive_52")
    h4_x = find_row(evaluation, "B", "xgboost", 4)
    h4_l = find_row(evaluation, "B", "lightgbm", 4)
    h4_a = find_row(evaluation, "A", "lightgbm", 4)
    with st.container(border=True):
        section_title("4-week full sample", "Experiment B on the 2024–2025 test")
        if last and seasonal:
            st.caption(
                f"Last-value uses the same {int(last['n_scored']):,} rows as Experiment B LightGBM. "
                f"Seasonal naive uses {int(seasonal['n_scored']):,} of those rows, where a price was published 52 weeks earlier. "
                "LSTM was not run."
            )
        else:
            st.caption("The naive rows are not in the saved comparison file. LSTM was not run.")
        if h4_x and h4_l and h4_a and last:
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
    with mae_col:
        with st.container(border=True):
            section_title("Point error", "MAE, LKR per kg. Unscored models are omitted.")
            labels = [_model_label(row["id"]) for row in scored]
            fig = go.Figure(go.Bar(x=labels, y=[row["mae"] for row in scored], marker_color=HARVEST, name="MAE"))
            fig.update_layout(showlegend=False, yaxis_title="Rs. / kg")
            fig.update_xaxes(tickangle=-30)
            show_chart(fig, 300)
    intervals = [row for row in scored if row.get("picp") is not None]
    with picp_col:
        with st.container(border=True):
            section_title("Interval", "Calibrated PICP and width. Naive forecasts have no interval.")
            fig2 = go.Figure()
            interval_labels = [_model_label(row["id"]) for row in intervals]
            fig2.add_trace(go.Bar(x=interval_labels, y=[row["picp"] for row in intervals], marker_color=SKY, name="PICP"))
            fig2.add_trace(
                go.Bar(
                    x=interval_labels,
                    y=[row["interval_width"] for row in intervals],
                    marker_color=AMBER,
                    name="Width",
                )
            )
            fig2.update_xaxes(tickangle=-30)
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
