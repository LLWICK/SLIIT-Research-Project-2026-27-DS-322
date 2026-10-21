"""Member 1 handoff — analysis-ready extract that Member 2 scores."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from lib.cultivation_status import load_cultivation_status
from lib.scores import b4_row, coverage_words
from lib.experiment_c import experiment_c
from lib.load import member1_status, title_case
from lib.nav import go_member
from lib.style import page_header, section_title


def render(data: dict[str, Any]) -> None:
    coverage = data["coverage"]
    split = data["split"]
    meta = data["meta"]
    upstream = member1_status()

    page_header(
        "Member 1",
        "Weekly panel passed to the forecast",
        "Member 1 prepares HARTI wholesale prices, seasonal commitments, and NASA POWER weather. "
        "Member 2 scores originally observed wholesale weeks and does not re-clean those files.",
    )

    k1, k2, k3, k4 = st.columns(4)
    if upstream.get("found"):
        k1.metric("Price rows in extract", f"{upstream['price_rows']:,}")
        k2.metric("Wholesale rows", f"{upstream['wholesale_rows']:,}")
        k3.metric("Scoped model rows", f"{upstream['scoped_rows']:,}", "Carrot / Leek / Tomato")
        k4.metric("Commitment rows", f"{upstream['commitment_rows']:,}")
        st.caption(
            f"Member 1 extract detected at {upstream['price_path']}. "
            f"Weather file {'present' if upstream['has_weather'] else 'missing'}."
        )
    else:
        k1.metric("Frozen demo panel", f"{meta['n_panel']:,}")
        k2.metric("Scored test weeks", f"{meta['n_scored_test']:,}")
        measured = None if b4_row(data["evaluation"]) is None else b4_row(data["evaluation"]).get("picp_cqr")
        k3.metric("4-week calibrated PICP", "not scored" if measured is None else f"{float(measured):.2f}%")
        cultivation = load_cultivation_status()
        run = experiment_c()
        if run.get("trained"):
            k4.metric("Experiment C", "Trained", "National matched-row test")
        elif cultivation.get("doa_present"):
            k4.metric("DOA progress file", f"{cultivation['n_rows']} rows", "National rows scored. District rows audit-only.")
        else:
            k4.metric("District-month file", "Not on disk", "Experiment C has not been trained")
        st.caption(
            "The Member 1 extract is not on this path. Counts are the frozen scored panel. "
            + coverage_words(measured)
            + "."
        )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            section_title("Issue-week split", "Train, calibration, then the 2024–2025 test")
            table = pd.DataFrame(
                [
                    {
                        "Window": name.title(),
                        "Rows": split[name]["n"],
                        "Start": split[name]["start"],
                        "End": split[name]["end"],
                    }
                    for name in ("train", "calibration", "test")
                ]
            )
            st.dataframe(table, width="stretch", hide_index=True)
            st.caption("Train, then a calibration block for CQR, then 2024–2025 test weeks.")

    with right:
        with st.container(border=True):
            section_title("Series used downstream", "What Member 2 is allowed to score")
            cov_df = pd.DataFrame(coverage["by_series"])
            cov_df["crop"] = cov_df["crop"].map(title_case)
            cov_df["market"] = cov_df["market"].map(title_case)
            st.dataframe(
                cov_df.rename(columns={"coverage_pct": "price weeks %"})[
                    ["crop", "market", "rows", "price weeks %"]
                ],
                width="stretch",
                hide_index=True,
            )

    with st.container(border=True):
        section_title("Handoff contract", "1 → 2")
        st.markdown(
            """
- Grain: **crop × market × week**
- Price type: **wholesale only**
- Missing prices stay missing — never zero-filled
- Commitments and weather join through an **origin map**, not the market name
- Colombo is the consumer-market series; Badulla and Nuwara Eliya are origin-adjacent
            """
        )

    if st.button("Continue to Member 2", type="primary", key="cont_m2_from_m1"):
        go_member("2")
