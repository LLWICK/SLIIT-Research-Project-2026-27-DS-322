from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from lib.cultivation_status import load_cultivation_status
from lib.figures import show_figures
from lib.experiment_c import experiment_c, render_protocol_note, render_saved_b
from lib.style import page_header, section_title

REPO = Path(__file__).resolve().parents[2]
SAVED_MODELS = (
    ("Experiment A", REPO / "outputs" / "experiment_a" / "models"),
    ("Experiment B", REPO / "outputs" / "experiment_b" / "models"),
    ("B* matched rows", REPO / "outputs" / "experiment_c" / "models_b_star"),
    ("Experiment C", REPO / "outputs" / "experiment_c" / "models"),
)


def _scope() -> tuple[str, str]:
    try:
        from htfe.data.build.scope import CROPS, EVAL_MARKETS

        return ", ".join(CROPS), ", ".join(EVAL_MARKETS)
    except Exception:
        return "carrot, leeks, tomato", "Dambulla, Colombo, Meegoda, Nuwara Eliya"


def _saved_model_rows() -> list[dict[str, Any]]:
    rows = []
    for label, folder in SAVED_MODELS:
        files = sorted(folder.glob("*.joblib")) if folder.is_dir() else []
        rows.append(
            {
                "Fit": label,
                "Folder": str(folder.relative_to(REPO)) if folder.is_dir() else "not saved",
                "Joblib files": len(files),
            }
        )
    sarimax = REPO / "outputs" / "experiment_a" / "predictions.csv"
    rows.append(
        {
            "Fit": "SARIMAX",
            "Folder": "outputs/experiment_a/predictions.csv" if sarimax.is_file() else "not saved",
            "Joblib files": 0,
        }
    )
    return rows


def render(data: dict[str, Any]) -> None:
    evaluation = data["evaluation"]
    cultivation = load_cultivation_status()
    run = experiment_c()
    crops, markets = _scope()
    c_status = "trained on matched national rows" if run.get("trained") else "not trained"
    split = evaluation.get("split") or {}
    page_header(
        "What we did",
        "Question, data, split, and the models that were saved",
        "Weekly wholesale price, LKR per kg. The series is not shuffled.",
    )

    with st.container(border=True):
        section_title("Two functions", "What this forecast component does")
        st.markdown(
            """
**Issue-week row.**
Input: wholesale price, origin weather, published macros, and lagged extent. A value is used only when it was already public that week. Experiment C also takes the national achieved/target ratio when the report date is on or before that Monday.
Output: one row for each crop, market, issue week, and horizon.

**Horizon score.**
Input: those rows, split by the week the forecast is issued.
Output: MAE, MAPE, and calibrated coverage for Experiment A, Experiment B, and SARIMAX. Experiment C is scored on its own matched rows. The lowest MAE is stated one horizon at a time.
            """
        )

    with st.container(border=True):
        section_title("Question and design", "One chronological protocol")
        st.markdown(
            f"""
- Question: forecast the wholesale price at horizons {", ".join(str(item) for item in evaluation["horizons"])} weeks.
- Crops: {crops}. Score markets: {markets}.
- Label: {evaluation["label"]}.
- Issue-week split: train {split.get("train", "")}; calibration {split.get("calibration", "")}; test {split.get("test", "")}.
- Leakage rule: a column is used only when it was already public at the forecast issue week. Tree size is chosen on 2022 target weeks. 2023 sets the interval adjustment and is not used to choose the trees. The 2024–2025 test is not used to fit.
- {evaluation.get("tree_protocol", "")}.
- {evaluation["optuna"]}.
- LightGBM and XGBoost use quantiles {", ".join(str(item) for item in evaluation["quantiles"])}. The point forecast is the log change from the last observed price, converted back to LKR/kg.
- Experiment A is price history and season. Experiment B adds origin weather, diesel, the exchange rate, inflation, and lagged finished-season extent. Experiment C is a separate matched-row test: {c_status}. {cultivation["headline"]}
- SARIMAX is the statistical order (1, 1, 1) x (1, 0, 0, 52). It is stored with the Experiment A predictions, not as a tree file. LSTM was not run.
            """
        )

    with st.container(border=True):
        section_title("Feature sets", "What each experiment was allowed to see")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Experiment": "A", "Inputs": "Price lags and season", "Status": "trained"},
                    {
                        "Experiment": "B",
                        "Inputs": "A plus origin weather, diesel, USD/LKR, inflation, and finished-season extent",
                        "Status": "trained",
                    },
                    {
                        "Experiment": "C",
                        "Inputs": "Same eligible rows as B*, plus national as-of achieved/target. District rows are not features.",
                        "Status": c_status,
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "A later forecast loads the joblib files and does not fit them again. "
            "A new training run writes a temporary folder and replaces these files only after that run succeeds."
        )
    with st.container(border=True):
        section_title("Models saved after training", "LightGBM and XGBoost quantiles 0.05, 0.50, and 0.95")
        st.dataframe(pd.DataFrame(_saved_model_rows()), width="stretch", hide_index=True)
        st.caption("SARIMAX has no tree file. Its forecasts stay in the Experiment A prediction table.")
    render_protocol_note()
    render_saved_b()
    show_figures("Training")
