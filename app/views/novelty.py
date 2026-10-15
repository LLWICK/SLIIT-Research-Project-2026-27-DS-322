from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from lib.cultivation_status import load_cultivation_status
from lib.experiment_c import (
    render_c_diagnostics,
    render_experiment_c_scores,
    render_member3_handoff,
    render_reforecast,
    render_saved_model_use,
)
from lib.figures import show_figures
from lib.scores import b4_coverage_sentence
from lib.style import page_header, section_title

REPO = Path(__file__).resolve().parents[2]


def render(data: dict[str, Any]) -> None:
    ablation = data["ablation"]
    weeks = data["weeks"] or []
    page_header(
        "Cultivation proof",
        "Matched national rows, separate from the full-sample table",
        "B* and C share the same eligible rows. District rows were not features. "
        "The dashboard forecast is the runtime model and is not this accuracy result.",
    )
    render_member3_handoff()
    render_experiment_c_scores()
    shock_path = REPO / "outputs" / "figures" / "shock_note.json"
    if shock_path.is_file():
        shock = json.loads(shock_path.read_text(encoding="utf-8"))
        st.caption(shock.get("note") or "The 2024 Q1 shock note is on disk.")

    if not (isinstance(ablation, dict) and ablation.get("status") == "not_run"):
        arms = []
        for arm in ("A", "B", "C"):
            block = ablation.get(arm) if isinstance(ablation, dict) else None
            metrics = (block or {}).get("metrics") if isinstance(block, dict) else None
            arms.append(
                {
                    "Arm": arm,
                    "Status": "trained" if metrics else "not in this table",
                    "MAE": None if not metrics else metrics.get("mae"),
                    "RMSE": None if not metrics else metrics.get("rmse"),
                    "MAPE %": None if not metrics else metrics.get("mape"),
                    "PICP raw %": None if not metrics else metrics.get("picp"),
                }
            )
        with st.container(border=True):
            section_title("Full-sample ablation", "A and B only. C is not in this table.")
            st.dataframe(pd.DataFrame(arms), width="stretch", hide_index=True)
            st.caption(
                "PICP here is the raw 5–95% interval. "
                + b4_coverage_sentence(data["evaluation"])
                + " "
                + str((ablation or {}).get("note") or "")
            )

    render_c_diagnostics()
    with st.container(border=True):
        section_title("Monthly re-forecast", "Runtime update mechanism, separate from Experiment C")
        if not weeks:
            cultivation = load_cultivation_status()
            st.caption(
                "The commitment-week list on this page is empty. "
                "The dashboard slider scores the runtime model again. That slider is not Experiment C. "
                + cultivation["headline"]
            )
        else:
            st.dataframe(pd.DataFrame(weeks), width="stretch", hide_index=True)
            st.caption("These rows are not the matched-row Experiment C test.")
    render_reforecast()
    render_saved_model_use()
    show_figures("Novelty")
