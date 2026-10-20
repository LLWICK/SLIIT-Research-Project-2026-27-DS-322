from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from lib.cultivation_status import render_cultivation_status
from lib.scores import b4_row, coverage_words, find_row
from lib.live_match import handoff_packet, handoff_request, match_forecast
from lib.load import lkr, title_case, write_forecast_packet
from lib.nav import go_member
from lib.style import fact, facts, page_header, pick, range_bar, section_title


def _horizon_label(match: dict[str, Any]) -> str:
    issue = str(match.get("forecast_issue_week") or "")[:10]
    target = str(match.get("forecast_week") or "")[:10]
    try:
        days = (date.fromisoformat(target) - date.fromisoformat(issue)).days
    except ValueError:
        return "1 week"
    if days == 7:
        return "1 week"
    if days > 0 and days % 7 == 0:
        weeks = days // 7
        return "1 week" if weeks == 1 else f"{weeks} weeks"
    if days > 0:
        return f"{days} days"
    return "1 week"


def _saved_test(evaluation: dict[str, Any]) -> None:
    rows: list[tuple[str, str]] = []
    sarimax = find_row(evaluation, "B", "sarimax", 1)
    xgb = find_row(evaluation, "B", "xgboost", 4)
    if sarimax is not None:
        rows.append(("1 week", f"SARIMAX, MAE {float(sarimax['mae']):.2f}"))
    if xgb is not None:
        rows.append(("4 weeks", f"XGBoost, MAE {float(xgb['mae']):.2f}"))
    if not rows:
        return
    st.caption("Saved full-sample test. The lowest model changes with the horizon. Not this runtime price.")
    facts(rows, ready=True)


def render(data: dict[str, Any]) -> None:
    live = data["live"]
    crops = sorted({row["crop"] for row in live})

    page_header(
        "Member 2",
        "Price forecast",
        "Choose a crop and market. This interval is the saved runtime model. "
        "It is not the Experiment B score and it is not the Experiment C score.",
    )
    _saved_test(data["evaluation"])
    render_cultivation_status(compact=True)

    if not crops:
        st.caption("No runtime forecast is on disk for this studio.")
        if st.button("Continue to Member 3", type="primary", key="dash_to_m3_empty"):
            go_member("3")
        return

    if "dash_intensity_pct" not in st.session_state:
        st.session_state.dash_intensity_pct = 85

    if "dash_crop" not in st.session_state:
        st.session_state.dash_crop = "carrot" if "carrot" in crops else crops[0]
    preview_crop = st.session_state.dash_crop if st.session_state.dash_crop in crops else crops[0]
    preview_markets = sorted({row["market"] for row in live if row["crop"] == preview_crop})
    if st.session_state.get("dash_market") not in preview_markets:
        st.session_state.dash_market = "colombo" if "colombo" in preview_markets else preview_markets[0]
    intensity = st.session_state.dash_intensity_pct / 100.0
    match = match_forecast(live, st.session_state.dash_crop, st.session_state.dash_market, intensity)
    request = handoff_request(match, intensity) if match else None
    packet = handoff_packet(match) if match else None
    if request and packet:
        write_forecast_packet(request, packet)

    inputs, forecast, inbox = st.columns(3, gap="small", vertical_alignment="top")
    with inputs:
        with st.container(border=True):
            section_title("Inputs", "Market")
            crop = pick("Crop", crops, key="dash_crop", preferred="carrot", format_func=title_case)
            markets = sorted({row["market"] for row in live if row["crop"] == crop})
            market = pick("Market", markets, key="dash_market", preferred="colombo", format_func=title_case)
            progress_in_model = bool(match and match.get("progress_in_model"))
            intensity_pct = st.slider(
                "Cultivation progress" if progress_in_model else "Cultivation progress (not in this model)",
                min_value=25,
                max_value=150,
                step=5,
                format="%d%%",
                key="dash_intensity_pct",
                help="Achieved hectares divided by target hectares. Ignored unless that column was in the trained model.",
                disabled=not progress_in_model,
            )
            intensity = intensity_pct / 100.0

    with forecast:
        with st.container(border=True):
            section_title("Forecast", "Expected wholesale price")
            if not match:
                st.caption("No live interval for this crop and market.")
            else:
                expected = int(match["predicted_price"])
                low = int(match["lower_price"])
                high = int(match["upper_price"])
                facts(
                    [
                        ("Crop", title_case(match["crop"])),
                        ("Market", title_case(match["market"])),
                        ("Horizon", _horizon_label(match)),
                        ("Week", str(match["forecast_week"])),
                    ]
                )
                st.markdown(
                    f"""
<div class="price-row">
  <div class="price-tile">
    <div class="price-label">Low</div>
    <div class="price-value">{lkr(low)}</div>
  </div>
  <div class="price-tile accent">
    <div class="price-label">Expected</div>
    <div class="price-value" data-expected-price="{expected}">{lkr(expected)}</div>
  </div>
  <div class="price-tile">
    <div class="price-label">High</div>
    <div class="price-value">{lkr(high)}</div>
  </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
                range_bar(low, expected, high)
                fact(
                    "Saved B horizon-4 coverage, not this forecast",
                    coverage_words(None if b4_row() is None else b4_row().get("picp_cqr")),
                )
                fact("Cultivation update applied", "Yes" if match.get("applied_cultivation_update") else "No")
                fact("Progress source", str(match.get("commitment_source") or "not available"))
                st.caption(match.get("note") or "")

    with inbox:
        with st.container(border=True):
            section_title("Next stage", "Crop viability")
            st.caption("Member 3 would read this range. This button opens the note on this page. It does not start Member 3's app.")
            if packet:
                fact("Crop", title_case(packet["crop"]))
                fact("Market", title_case(packet["market"]))
                fact("Expected price", lkr(packet["predicted_price"]))
                fact("Range", f"{lkr(packet['lower_price'])} – {lkr(packet['upper_price'])}")
                fact("Status", "Ready", ready=True)
                st.markdown(
                    "<p class='mist'>Member 4 can reuse the same weekly price. That simulator is not started here.</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No runtime forecast is stored for this crop and market.")
            if st.button("Continue to Member 3", type="primary", key="dash_to_m3", use_container_width=True):
                go_member("3")
