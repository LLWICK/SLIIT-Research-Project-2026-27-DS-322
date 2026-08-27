from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.style import AMBER, HARVEST, page_header, section_title, show_chart


def render(data: dict[str, Any]) -> None:
    ablation = data["ablation"]
    weeks = data["weeks"]
    importance = data["importance"]
    intensity_rank = next((i + 1 for i, row in enumerate(importance) if row["feature"] == "cultivation_intensity"), "—")

    page_header(
        "The two experiments that carry the mark",
        "A/B/C ablation and mid-season re-forecast",
        "Same LightGBM family, same hyperparameters, same chronological folds, same seed. "
        "Only the column list changes. A rigorous near-tie is still a result — we do not hide it.",
    )

    with st.container(border=True):
        section_title("Ablation", "A historical · B multi-source · C + intensity")
        table = pd.DataFrame(
            [
                {
                    "Arm": arm,
                    "MAE": ablation[arm]["metrics"]["mae"],
                    "RMSE": ablation[arm]["metrics"]["rmse"],
                    "MAPE %": ablation[arm]["metrics"]["mape"],
                    "Pinball": ablation[arm]["metrics"]["pinball"],
                    "PICP %": ablation[arm]["metrics"]["picp"],
                    "Width": ablation[arm]["metrics"]["interval_width"],
                }
                for arm in ("A", "B", "C")
            ]
        )
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption(
            f"On this extract, B is slightly best on MAE. C stays within 0.2 LKR and keeps PICP near 90%. "
            f"Intensity ranks #{intensity_rank} in permutation importance. The operational proof is the week experiment, "
            "not a forced accuracy win."
        )

    left, right = st.columns(2)
    week_df = pd.DataFrame(weeks)
    with left:
        with st.container(border=True):
            section_title("Week 2 / 5 / 8 / 12", "Error falls as commitments accumulate")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=week_df["commitment_week"], y=week_df["mae"], mode="lines+markers", line=dict(color=HARVEST, width=2), name="MAE"))
            fig.add_trace(go.Scatter(x=week_df["commitment_week"], y=week_df["rmse"], mode="lines+markers", line=dict(color=AMBER, width=2), name="RMSE"))
            fig.update_layout(xaxis_title="Commitment week", yaxis_title="Error")
            show_chart(fig, 320)
            st.caption(
                f"MAE {week_df.iloc[0]['mae']} at week 2 → {week_df.iloc[-1]['mae']} at week 12. "
                "The forecast is updated from new supply information, not from later prices."
            )

    with right:
        with st.container(border=True):
            section_title("Information states", "What the farmer would have known")
            for row in weeks:
                st.markdown(
                    f"**Week {row['commitment_week']}** · mean intensity {row['mean_intensity']}  \n"
                    f"MAE {row['mae']} · PICP {row['picp']}%"
                )
