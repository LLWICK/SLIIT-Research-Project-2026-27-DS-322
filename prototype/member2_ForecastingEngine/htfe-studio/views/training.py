from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.style import HARVEST, card_grid, page_header, pick, section_title, show_chart

STEPS = [
    ("sarima", "Seasonal baseline", "idle", "52-week seasonal naive. Locked protocol, price only."),
    ("lightgbm", "LightGBM quantiles", "idle", "Quantile 0.05 / 0.50 / 0.95, then chronological CQR."),
    ("xgboost", "XGBoost quantiles", "idle", "Same folds, same seed. Boosting comparison."),
    ("lstm", "LSTM-style MLP", "idle", "Optional. Same features; no conformal wrap."),
]


def render(data: dict[str, Any]) -> None:
    models = data["models"]
    comparison = data["comparison"]
    meta = data["meta"]

    page_header(
        "Azure-style experiment console",
        "Watch the registry fill",
        f"Replay of the real training run (seed {meta['seed']}, {meta['elapsed_s']}s wall time). "
        "Metrics come from the frozen demo artifacts.",
    )

    if st.button("Replay training", type="primary", key="train_replay"):
        st.session_state.train_replayed = True

    replayed = bool(st.session_state.get("train_replayed"))
    cards = []
    for model_id, title, _state, detail in STEPS:
        row = next((item for item in comparison if item["id"] == model_id), None)
        card = {
            "role": "done" if replayed else "idle",
            "title": title,
            "active": replayed and model_id == "lightgbm",
        }
        if row and replayed:
            card["rows"] = [("MAE", str(row["mae"])), ("PICP", f"{row['picp']}%")]
        else:
            card["body"] = detail
        cards.append(card)
    card_grid(cards)

    with st.container(border=True):
        section_title("Live console", "Recorded training curve")
        chosen = pick(
            "Model",
            [step[0] for step in STEPS],
            key="train_model",
            preferred="xgboost",
            format_func=lambda mid: next(s[1] for s in STEPS if s[0] == mid),
        )
        raw = models.get(chosen, {}).get("train_curve") or []
        curve = pd.DataFrame(
            [
                {
                    "iter": int(row.get("iter") or row.get("epoch") or index + 1),
                    "mae": row.get("mae"),
                }
                for index, row in enumerate(raw)
                if row.get("mae") is not None
            ]
        )
        fig = go.Figure()
        if not curve.empty:
            fig.add_trace(go.Scatter(x=curve["iter"], y=curve["mae"], mode="lines", line=dict(color=HARVEST, width=2), name="MAE"))
            fig.update_layout(yaxis_title="MAE", xaxis_title="Iteration")
            show_chart(fig, 300)
        elif chosen == "lightgbm":
            overall = models.get("lightgbm", {}).get("metrics", {}).get("overall", {})
            st.caption("This artifact did not log per-iteration LightGBM MAE. Final scored test metrics are below. XGBoost and LSTM recorded curves.")
            m1, m2 = st.columns(2)
            m1.metric("Final test MAE", overall.get("mae"))
            m2.metric("PICP", f"{overall.get('picp')}%")
        elif chosen == "sarima":
            st.caption("SARIMA is a locked seasonal protocol. The log is per series, not a boosting curve.")
            st.dataframe(pd.DataFrame(raw), width="stretch", hide_index=True)
        else:
            st.caption("No recorded training curve for this model.")

        if replayed:
            for model_id, title, _, _ in STEPS:
                model = models.get(model_id, {})
                metrics = model.get("metrics", {}).get("overall", {})
                st.code(
                    f"[{model_id}] backend={model.get('backend')} qhat={model.get('qhat')} "
                    f"MAE={metrics.get('mae')} PICP={metrics.get('picp')}"
                )
            st.success("Registry flushed · LightGBM + CQR is the shipped engine")
