from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from lib.style import page_header, section_title


def render(data: dict[str, Any]) -> None:
    ablation = data["ablation"]
    weeks = data["weeks"] or []
    if isinstance(ablation, dict) and ablation.get("status") == "not_run":
        page_header("Later experiment", "Ablation is not in this run", ablation.get("note") or "Not run.")
        st.info(ablation.get("note") or "Not run yet.")
        return

    page_header(
        "Same model, different columns",
        "Feature sets A, B, and C",
        "LightGBM only. Same hyperparameters, same weeks, same metrics. C is proposed only when cultivation progress is a real column.",
    )
    if isinstance(ablation, dict) and ablation.get("note"):
        st.info(ablation["note"])

    arms = []
    for arm in ("A", "B", "C"):
        block = ablation.get(arm) if isinstance(ablation, dict) else None
        metrics = (block or {}).get("metrics") if isinstance(block, dict) else None
        arms.append(
            {
                "Arm": arm,
                "Status": "trained" if metrics else "not trained",
                "MAE": None if not metrics else metrics.get("mae"),
                "RMSE": None if not metrics else metrics.get("rmse"),
                "MAPE %": None if not metrics else metrics.get("mape"),
                "Pinball": None if not metrics else metrics.get("pinball"),
                "PICP %": None if not metrics else metrics.get("picp"),
                "Width": None if not metrics else metrics.get("interval_width"),
            }
        )
    with st.container(border=True):
        section_title("Ablation", "A historical · B weather, macros, and lagged extent · C not trained")
        st.dataframe(pd.DataFrame(arms), width="stretch", hide_index=True)
        st.caption(
            "The PICP in this table is the raw 5–95% interval. "
            "After the 2023 adjustment, Experiment B LightGBM covers 81.79% of the 4-week test prices. That is below 90%."
        )
        by_horizon = ablation.get("by_horizon") if isinstance(ablation, dict) else None
        if by_horizon:
            st.caption("Pooled test weeks. Horizon-level figures are in the run folder.")

    with st.container(border=True):
        section_title("Monthly re-forecast", "Update cultivation progress, do not retrain")
        if not weeks:
            st.warning(
                "Experiment C was not trained. There is no district-month file of target and achieved hectares, "
                "so these rows are empty and week 2 / 5 / 8 / 12 is not this experiment. "
                "The dashboard writes a new achieved/target record into a separate runtime model and scores it again. "
                "That is the update mechanism. It is not an accuracy result."
            )
        else:
            st.dataframe(pd.DataFrame(weeks), width="stretch", hide_index=True)
