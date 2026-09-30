from __future__ import annotations

from typing import Any

import streamlit as st

from lib.live_match import handoff_packet, handoff_request, match_forecast
from lib.load import lkr, title_case, write_forecast_packet
from lib.nav import go_member
from lib.style import fact, page_header, pick, range_bar, section_title, stage_row

STAGES = [
    {"role": "01", "title": "Data fusion", "body": "Weekly panel and reliability."},
    {"role": "02", "title": "Price forecast", "body": "Calibrated interval for the week.", "active": True},
    {"role": "03", "title": "Crop viability", "body": "Uses this forecast with weather."},
    {"role": "04", "title": "Season simulation", "body": "Tests advice at market scale."},
]


def render(data: dict[str, Any]) -> None:
    live = data["live"]
    crops = sorted({row["crop"] for row in live})

    page_header(
        "Cobweb decision support",
        "Price Forecast",
        "Choose a crop and market. The price is the trained LightGBM forecast. "
        "A cultivation-progress control only changes that price when the model was trained with monthly achieved/target hectares.",
    )
    stage_row(STAGES)

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

    st.markdown('<div id="dash-trio"></div>', unsafe_allow_html=True)
    inputs, forecast, inbox = st.columns(3, gap="medium", vertical_alignment="top")
    with inputs:
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
        st.markdown('<div class="slider-ends"><span>Low</span><span>Crowded</span></div>', unsafe_allow_html=True)

    with forecast:
        section_title("Forecast", "Expected wholesale price")
        if not match:
            st.warning("No live interval for this crop and market.")
        else:
            expected = int(match["predicted_price"])
            low = int(match["lower_price"])
            high = int(match["upper_price"])
            st.caption(
                f"{title_case(match['crop'])} · {title_case(match['market'])} · week of {match['forecast_week']}"
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
            fact("Coverage target", "90%")
            fact("Cultivation update applied", "Yes" if match.get("applied_cultivation_update") else "No")
            fact("Progress source", str(match.get("commitment_source") or "not available"))
            st.caption(match.get("note") or "")

    with inbox:
        section_title("Next stage", "Crop viability")
        st.caption("This forecast is ready for the viability component.")
        if packet:
            fact("Crop", title_case(packet["crop"]))
            fact("Market", title_case(packet["market"]))
            fact("Expected price", lkr(packet["predicted_price"]))
            fact("Range", f"{lkr(packet['lower_price'])} – {lkr(packet['upper_price'])}")
            fact("Status", "Ready", ready=True)
            st.markdown(
                "<p class='mist'>Viability scoring will combine this range with weather. "
                "The season simulation can reuse the same forecast at market scale.</p>",
                unsafe_allow_html=True,
            )
            if st.button("Open Member 3’s DSS →", type="primary", key="dash_to_m3"):
                go_member("3")
