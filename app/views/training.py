from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from lib.style import page_header, section_title


def render(data: dict[str, Any]) -> None:
    evaluation = data["evaluation"]
    page_header(
        "Chronological training",
        "What was fit, and on which weeks",
        "Train through 2022. Calibrate on 2023. Test on 2024–2025. The series is not shuffled.",
    )

    with st.container(border=True):
        section_title("Procedure", "Same rows for every model")
        st.markdown(
            f"""
- Label: {evaluation["label"]}.
- Horizons: {", ".join(str(item) for item in evaluation["horizons"])} weeks.
- The model learns the log change from the last observed price, then converts it back to LKR/kg.
- {evaluation["optuna"]}.
- LightGBM and XGBoost use quantiles {", ".join(str(item) for item in evaluation["quantiles"])}.
- The interval adjustment is fit on 2023 only and applied to the test weeks.
- Experiment C: {evaluation["experiment_c"]}.
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
                        "Inputs": "B plus current achieved hectares / target hectares",
                        "Status": "not trained",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("SARIMAX stays a price-only seasonal baseline, so it does not change between A and B.")
