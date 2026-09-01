"""Member 1 handoff — analysis-ready extract that Member 2 scores."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from lib.load import member1_status, title_case
from lib.nav import go_member
from lib.style import page_header, section_title, stage_row

STAGES = [
    {"role": "01", "title": "Data fusion", "body": "Weekly panel and reliability.", "active": True},
    {"role": "02", "title": "Price forecast", "body": "Calibrated interval for the week."},
    {"role": "03", "title": "Crop viability", "body": "Uses this forecast with weather."},
    {"role": "04", "title": "Season simulation", "body": "Tests advice at market scale."},
]


def render(data: dict[str, Any]) -> None:
    coverage = data["coverage"]
    split = data["split"]
    meta = data["meta"]
    upstream = member1_status()

    page_header(
        "Member 1 · data fusion",
        "The weekly panel Member 2 inherits",
        "Member 1 prepares HARTI wholesale prices, seasonal commitments, and NASA POWER weather. "
        "Member 2 does not re-clean those files — it scores only originally observed wholesale weeks.",
    )
    stage_row(STAGES)

    k1, k2, k3, k4 = st.columns(4)
    if upstream.get("found"):
        k1.metric("Price rows in extract", f"{upstream['price_rows']:,}")
        k2.metric("Wholesale rows", f"{upstream['wholesale_rows']:,}")
        k3.metric("Scoped model rows", f"{upstream['scoped_rows']:,}", "Carrot / Leek / Tomato")
        k4.metric("Commitment rows", f"{upstream['commitment_rows']:,}")
        st.markdown(
            '<div class="safe-alert"><b>✅ Member 1 extract detected</b><br>'
            f"Reading <code>{upstream['price_path']}</code>. "
            f"Weather file {'present' if upstream['has_weather'] else 'missing'}.</div>",
            unsafe_allow_html=True,
        )
    else:
        k1.metric("Frozen demo panel", f"{meta['n_panel']:,}")
        k2.metric("Scored test weeks", f"{meta['n_scored_test']:,}")
        k3.metric("Confidence target", "90%")
        k4.metric("Commitment source", title_case(str(meta.get("commitment_source", "simulated"))))
        st.markdown(
            '<div class="warn-alert"><b>⚠️ Live extract not on this path</b><br>'
            "Charts still use the frozen scored panel so metrics stay reproducible. "
            "Place <code>data/v1/analysis_ready_price.csv</code> at the repo root to light up the live count.</div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            section_title("Chronological split", "Time order only — never random")
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
                cov_df.rename(columns={"coverage_pct": "coverage %"})[
                    ["crop", "market", "rows", "coverage %"]
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

    if st.button("Continue to Member 2 · Price forecast", type="primary"):
        go_member("2")
