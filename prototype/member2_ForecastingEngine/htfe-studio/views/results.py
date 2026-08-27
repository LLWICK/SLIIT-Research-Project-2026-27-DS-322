from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from lib.load import lkr
from lib.style import card_grid, page_header, section_title

LABELS = {
    "sarima": ("Seasonal baseline", "SARIMA-family reference"),
    "lightgbm": ("LightGBM", "Primary engine"),
    "xgboost": ("XGBoost", "Boosting comparison"),
    "lstm": ("Sequential network", "Optional deep comparison"),
}


def render(data: dict[str, Any]) -> None:
    comparison = data["comparison"]
    split = data["split"]
    lgb = next((row for row in comparison if row["id"] == "lightgbm"), None)
    lstm = next((row for row in comparison if row["id"] == "lstm"), None)

    page_header(
        "Model results",
        "Four models, same weeks, same rules",
        f"All four models scored on {split['test']['start']} to {split['test']['end']} — "
        f"{split['test']['n']} observed test weeks. LightGBM is shipped because the 90% range holds.",
    )

    card_grid(
        [
            {
                "role": LABELS.get(row["id"], (row["model"], row["backend"]))[1],
                "title": LABELS.get(row["id"], (row["model"], row["backend"]))[0],
                "active": row["id"] == "lightgbm",
                "note": "Used on the dashboard" if row["id"] == "lightgbm" else "",
                "rows": [
                    ("MAE", lkr(row["mae"])),
                    ("MAPE", f"{row['mape']}%"),
                    ("Coverage", f"{row['picp']}%"),
                    ("Width", lkr(row["interval_width"])),
                ],
            }
            for row in comparison
        ]
    )

    with st.container(border=True):
        section_title("Same test set", "Side by side")
        table = pd.DataFrame(
            [
                {
                    "Model": LABELS.get(row["id"], (row["model"],))[0],
                    "MAE": row["mae"],
                    "RMSE": row["rmse"],
                    "MAPE %": row["mape"],
                    "Coverage %": row["picp"],
                    "Width": row["interval_width"],
                }
                for row in comparison
            ]
        )
        st.dataframe(table, width="stretch", hide_index=True)
        if lgb and lstm:
            st.caption(
                f"The sequential network can look better on MAE, but coverage drops to {lstm['picp']}% "
                f"with a tight band. LightGBM stays near the 90% target ({lgb['picp']}%) "
                f"with a width of about {lkr(lgb['interval_width'])}."
            )

    note, eval_col = st.columns(2)
    with note:
        with st.container(border=True):
            section_title("Data notes", "What this extract covers")
            st.caption(
                "Commitments are a labelled historical simulation, not live HARTI registrations. "
                "Colombo is the consumer-market series; Badulla and Nuwara Eliya are origin-adjacent."
            )
    with eval_col:
        with st.container(border=True):
            section_title("Evaluation", "How the numbers were scored")
            st.markdown(
                """
- Time order only — train, then calibration, then test.
- Only originally observed wholesale prices.
- 2020–2021 kept as a disruption slice.
- Coverage and width reported together.
                """
            )
