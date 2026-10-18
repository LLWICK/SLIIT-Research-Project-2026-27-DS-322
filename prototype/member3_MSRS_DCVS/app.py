"""Small Streamlit client for the Member 3 FastAPI prototype."""

from __future__ import annotations

import os
import uuid
from datetime import date

import requests
import streamlit as st


API_URL = os.getenv("MEMBER3_API_URL", "http://127.0.0.1:8000")
SUPPORTED_CROPS = ["carrot", "leek", "tomato"]
SUPPORTED_MARKETS = ["colombo", "dambulla", "meegoda", "nuwara_eliya"]
DEMO_FARMERS = ["demo-farmer-01", "demo-farmer-02", "demo-farmer-03"]

st.set_page_config(page_title="Member 3 Crop Viability Prototype", layout="wide")
st.title("Crowding-Aware Crop Recommendation Prototype")
st.caption("Experimental MSRS/DCVS. Demo capacities, weights, and thresholds are not HARTI-validated demand estimates.")
if "confirmation_key" not in st.session_state:
    st.session_state.confirmation_key = f"demo-{uuid.uuid4().hex[:12]}"

with st.form("proposal"):
    first, second, third = st.columns(3)
    with first:
        district = st.selectbox(
            "District / planning location",
            SUPPORTED_MARKETS,
            format_func=lambda value: value.replace("_", " ").title(),
            help="Prototype scope is limited to locations represented in the supplied Member 2 forecast artifact.",
        )
        market = st.selectbox("Market used by Member 2 forecast", SUPPORTED_MARKETS)
        season = st.selectbox("Season", ["Maha", "Yala"])
    with second:
        crop = st.selectbox("Requested crop", SUPPORTED_CROPS)
        area = st.number_input("Proposed area (ha)", min_value=0.1, value=20.0, step=0.5)
        horizon = st.selectbox("Forecast horizon (weeks)", [1, 2, 4, 8, 12], index=2)
    with third:
        st.caption("Demo suitability inputs until Member 1 provides a validated suitability service.")
        suitability = {item: st.slider(f"{item.title()} suitability", 0.0, 1.0, 0.7, 0.05) for item in SUPPORTED_CROPS}
    assessed = st.form_submit_button("Assess proposal")

assessment_payload = {
    "district": district,
    "market": market,
    "crop": crop,
    "season": season,
    "proposed_area_ha": area,
    "horizon_weeks": horizon,
    "suitability": suitability,
}

if assessed:
    response = requests.post(f"{API_URL}/recommendations/assess", json=assessment_payload, timeout=10)
    if response.ok:
        st.session_state.assessment = response.json()
    else:
        st.error(response.text)

if "assessment" in st.session_state:
    result = st.session_state.assessment
    requested = result["requested_crop_assessment"]
    st.subheader("Requested crop assessment")
    one, two, three = st.columns(3)
    one.metric("Experimental pressure risk", requested["pressure"]["score"])
    two.metric("Experimental DCVS", f"{requested['dcvs']:.1f}%")
    three.metric("Post-proposal confirmed area", f"{requested['proposed_total_area_ha']:.1f} ha")
    st.warning("The shown reference capacity is a visible prototype placeholder, not a measured market-demand estimate.")
    st.subheader("Ranked alternatives")
    st.dataframe([
        {"crop": item["crop"], "DCVS": item["dcvs"], "pressure": item["pressure"]["label"], "forecast price": item["forecast"]["predicted_price"]}
        for item in result["recommendations"]
    ], hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Confirm planting")
    st.caption("Assessment does not save anything. Confirmation creates the only commitment record.")
    with st.form("confirmation"):
        farmer_reference = st.selectbox("Farmer/demo reference", DEMO_FARMERS)
        st.caption(f"Confirmation key: {st.session_state.confirmation_key}")
        planting_date = st.date_input("Planting date", value=date.today())
        harvest_date = st.date_input("Expected harvest date")
        confirmed = st.checkbox("I confirm this planting decision")
        submit = st.form_submit_button("Save confirmed commitment")
    if submit:
        commitment = assessment_payload | {
            "idempotency_key": st.session_state.confirmation_key,
            "farmer_reference": farmer_reference,
            "planting_date": str(planting_date),
            "harvest_date": str(harvest_date),
            "confirmed": confirmed,
        }
        response = requests.post(f"{API_URL}/commitments", json=commitment, timeout=10)
        if response.ok:
            st.success(f"Commitment {response.json()['id']} saved. Future assessments use the updated aggregate.")
            st.session_state.confirmation_key = f"demo-{uuid.uuid4().hex[:12]}"
        else:
            st.error(response.text)
