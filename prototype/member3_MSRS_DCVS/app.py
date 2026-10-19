"""Interactive multi-paradigm recommendation dashboard for Member 3 prototype.

Demonstrates:
1. Agro-climatic suitability mapped from planting areas (altitude, rainfall, climate, temperature, soil).
2. Market Saturation Risk Score (MSRS) evaluation for requested crops.
3. Multi-paradigm recommendation comparisons:
   - Baseline Weighted DCVS
   - MCDM / TOPSIS (Multi-Criteria Ideal Distance)
   - Non-Linear BPR Congestion + Crop Vector Similarity Recommender
   - Markowitz Portfolio Land-Splitting / Hedging
   - Game-Theoretic Softmax Load Balancing (Herding Prevention)
4. Confirmed commitment workflow with SQLite persistence.
"""

from __future__ import annotations

import os
import uuid
from datetime import date
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("MEMBER3_API_URL", "http://127.0.0.1:8000")
SUPPORTED_CROPS = ["carrot", "leek", "tomato"]
SUPPORTED_DISTRICTS = ["nuwara_eliya", "badulla", "dambulla", "colombo", "meegoda"]
SUPPORTED_MARKETS = ["colombo", "dambulla", "meegoda", "nuwara_eliya"]
DEMO_FARMERS = ["demo-farmer-01", "demo-farmer-02", "demo-farmer-03"]

st.set_page_config(
    page_title="Crowding-Aware Crop Recommendation Engine",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Crowding-Aware Crop Recommendation System")
st.markdown(
    "**SLIIT Research Project J26-DS-322 | Member 3 Prototype**  \n"
    "*Evaluate market saturation risk (Cobweb effect), compute agro-climatic suitability from environmental parameters, and compare multiple recommendation paradigms.*"
)

if "confirmation_key" not in st.session_state:
    st.session_state.confirmation_key = f"demo-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------
# Sidebar: Location Environmental Profile & Suitability
# ---------------------------------------------------------
with st.sidebar:
    st.header("📍 Planting Area & Climate")
    selected_district = st.selectbox(
        "Select Planting District",
        SUPPORTED_DISTRICTS,
        format_func=lambda x: x.replace("_", " ").title(),
        index=0,
    )

    # Fetch planting area environmental profile from API or local fallback
    area_profile: dict[str, Any] = {}
    try:
        resp = requests.get(f"{API_URL}/planting-areas/{selected_district}", timeout=3)
        if resp.ok:
            area_data = resp.json()
            area_profile = area_data["profile"]
            suitability_defaults = {
                crop: area_data["suitability_breakdown"][crop]["composite_suitability"]
                for crop in SUPPORTED_CROPS
            }
        else:
            suitability_defaults = {"carrot": 0.9, "leek": 0.85, "tomato": 0.45}
    except Exception:
        suitability_defaults = {"carrot": 0.9, "leek": 0.85, "tomato": 0.45}

    if area_profile:
        st.subheader(area_profile.get("name", selected_district.title()))
        st.caption(area_profile.get("description", ""))
        st.metric("Elevation / Altitude", f"{area_profile.get('altitude_m', 0):.0f} m")
        st.metric("Mean Temperature", f"{area_profile.get('mean_temp_c', 0):.1f} °C")
        st.metric("Seasonal Rainfall", f"{area_profile.get('seasonal_rainfall_mm', 0):.0f} mm")
        st.caption(f"**Zone:** {area_profile.get('climate_zone', 'N/A')}")
        st.caption(f"**Soil:** {area_profile.get('soil_type', 'N/A')}")
        st.divider()

    st.subheader("🌱 Agro-Climatic Suitability")
    use_climate_model = st.checkbox(
        "Auto-derive suitability from climate & soil",
        value=True,
        help="Calculates continuous suitability scores using the district's altitude, rainfall, and temperature profiles.",
    )

    suitability_inputs = {}
    if use_climate_model:
        st.info("Using automated environmental suitability mapping:")
        for c in SUPPORTED_CROPS:
            score = suitability_defaults.get(c, 0.7)
            st.progress(score, text=f"{c.title()}: {score * 100:.1f}%")
            suitability_inputs[c] = score
    else:
        st.warning("Manual suitability override mode:")
        for c in SUPPORTED_CROPS:
            suitability_inputs[c] = st.slider(
                f"{c.title()} Suitability",
                0.0,
                1.0,
                float(suitability_defaults.get(c, 0.7)),
                0.05,
            )

# ---------------------------------------------------------
# Farmer Proposal Form
# ---------------------------------------------------------
st.subheader("1. Farmer Crop Proposal & Planning Inputs")
with st.form("farmer_proposal"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        crop = st.selectbox("Requested Crop", SUPPORTED_CROPS, format_func=lambda x: x.title())
        area = st.number_input("Proposed Area (ha)", min_value=0.5, value=25.0, step=0.5)
    with c2:
        market = st.selectbox(
            "Target Terminal Market",
            SUPPORTED_MARKETS,
            format_func=lambda x: x.replace("_", " ").title(),
            index=0,
        )
        season = st.selectbox("Season", ["Maha", "Yala"])
    with c3:
        horizon = st.selectbox("Price Forecast Horizon (weeks)", [1, 2, 4, 8, 12], index=2)
        farmer_ref = st.selectbox("Farmer Reference", DEMO_FARMERS)
    with c4:
        st.write("")
        st.write("")
        assessed = st.form_submit_button("🔍 Evaluate Saturation Risk & Alternatives", use_container_width=True)

assessment_payload = {
    "district": selected_district,
    "market": market,
    "crop": crop,
    "season": season,
    "proposed_area_ha": area,
    "horizon_weeks": horizon,
    "suitability": suitability_inputs,
}

if assessed:
    with st.spinner("Querying market commitments, Member 2 price forecasts, and running recommendation paradigms..."):
        try:
            response = requests.post(f"{API_URL}/recommendations/assess", json=assessment_payload, timeout=10)
            if response.ok:
                st.session_state.assessment = response.json()
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as ex:
            st.error(f"Failed to connect to recommendation backend at {API_URL}: {ex}")

# ---------------------------------------------------------
# Results Section
# ---------------------------------------------------------
if "assessment" in st.session_state:
    result = st.session_state.assessment
    requested = result["requested_crop_assessment"]
    approaches = result.get("approaches", {})
    pressure_info = requested["pressure"]
    pressure_score = pressure_info["score"]
    pressure_ratio = pressure_info["pressure_ratio"]
    pressure_label = pressure_info["label"]

    st.divider()
    st.subheader(f"2. Market Saturation Evaluation for Requested Crop: **{crop.title()}**")

    # Saturation Risk Alert Banner
    if pressure_score >= 0.7 or pressure_ratio >= 1.2:
        st.error(
            f"🚨 **CRITICAL MARKET SATURATION RISK DETECTED!**  \n"
            f"Current commitments + your proposed area ({requested['proposed_total_area_ha']:.1f} ha) will reach "
            f"**{pressure_ratio * 100:.1f}%** of market capacity ({requested['reference_capacity_ha']:.0f} ha).  \n"
            f"Risk Score: **{pressure_score:.2f} (High)**. Planting this will induce the **Cobweb oversupply crash**, leading to severe price drops. "
            f"**Action:** Switch to a recommended alternative or hedge using land-splitting below."
        )
    elif pressure_score >= 0.3 or pressure_ratio >= 0.8:
        st.warning(
            f"⚠️ **MODERATE CROWDING DETECTED.**  \n"
            f"Proposed area pushes aggregate commitments to **{pressure_ratio * 100:.1f}%** of market capacity. "
            f"Risk Score: **{pressure_score:.2f} (Moderate)**. Consider lower acreage or viable alternatives."
        )
    else:
        st.success(
            f"✅ **SAFE PLANTING ZONE.**  \n"
            f"Aggregate commitments remain healthy at **{pressure_ratio * 100:.1f}%** of market capacity. "
            f"Risk Score: **{pressure_score:.2f} (Low)**. Market demand is sufficient."
        )

    # Overview Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confirmed Area (District/Season)", f"{requested['current_confirmed_area_ha']:.1f} ha")
    m2.metric("Post-Proposal Total Area", f"{requested['proposed_total_area_ha']:.1f} ha")
    m3.metric("Capacity Overshoot Ratio", f"{pressure_ratio * 100:.1f}%")
    m4.metric("Forecast Price (Member 2)", f"LKR {requested['forecast']['predicted_price']:.2f}/kg")

    st.write("")

    # ---------------------------------------------------------
    # 3. Multi-Paradigm Recommendation Comparison
    # ---------------------------------------------------------
    st.subheader("3. Alternative Recommendations Across Different Algorithmic Paradigms")
    st.caption("Inspect and compare how 5 distinct academic and operational solutions formulate crop recommendations.")

    tab_overview, tab_topsis, tab_bpr, tab_portfolio, tab_softmax, tab_baseline, tab_climate = st.tabs([
        "📊 Approach Comparison Matrix",
        "🎯 MCDM / TOPSIS",
        "🚜 Non-Linear BPR & Crop Similarity",
        "💼 Portfolio Land-Splitting (Hedging)",
        "🌐 Softmax Load Balancing",
        "⚖️ Baseline Weighted DCVS",
        "🗺️ Climate Suitability Breakdown",
    ])

    # ------------------ Tab 1: Comparison Matrix ------------------
    with tab_overview:
        st.markdown("#### Cross-Paradigm Recommendation Summary")
        st.write("Compare the top-ranked choices and rationale generated by each approach for your proposal:")

        baseline_top = approaches.get("baseline_dcvs", [{}])[0].get("crop", "N/A").title()
        topsis_top = approaches.get("topsis_mcdm", [{}])[0].get("crop", "N/A").title()
        bpr_top = approaches.get("bpr_similarity", [{}])[0].get("crop", "N/A").title()
        softmax_top = approaches.get("softmax_load_balancing", [{}])[0].get("crop", "N/A").title()
        portfolio_info = approaches.get("portfolio_hedging", {})

        summary_data = [
            {
                "Approach": "1. Baseline DCVS",
                "Domain Paradigm": "Linear Multi-Attribute Scoring",
                "Top Recommendation": baseline_top,
                "Core Mechanism": "Weighted combination of price, linear MSRS, and climate suitability.",
            },
            {
                "Approach": "2. MCDM / TOPSIS",
                "Domain Paradigm": "Multi-Criteria Decision Analysis",
                "Top Recommendation": topsis_top,
                "Core Mechanism": "Minimizes geometric distance to ideal solution, penalizes forecast price uncertainty interval.",
            },
            {
                "Approach": "3. BPR & Profile Similarity",
                "Domain Paradigm": "Traffic Congestion & Content Recommender",
                "Top Recommendation": bpr_top,
                "Core Mechanism": "Uses non-linear traffic congestion curves and matches farmer operational profile vectors.",
            },
            {
                "Approach": "4. Portfolio Land-Splitting",
                "Domain Paradigm": "Modern Portfolio Theory (Markowitz)",
                "Top Recommendation": f"Split: {', '.join(a['crop'].title() + ' (' + str(a['allocation_pct']) + '%)' for a in portfolio_info.get('allocations', []))}",
                "Core Mechanism": "Caps saturated crop to safe threshold and hedges the rest into low-risk alternatives.",
            },
            {
                "Approach": "5. Softmax Load Balancing",
                "Domain Paradigm": "Game Theory / Macro Equilibrium",
                "Top Recommendation": f"{softmax_top} ({approaches.get('softmax_load_balancing', [{}])[0].get('rank_score', 0)}% cohort share)",
                "Core Mechanism": "Probabilistic load balancing to prevent the Herding Effect and secondary oversupply.",
            },
        ]
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    # ------------------ Tab 2: TOPSIS ------------------
    with tab_topsis:
        st.markdown("#### Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)")
        st.markdown(
            "Evaluates candidate crops against conflicting objectives: **Maximize Suitability**, **Maximize Forecast Price**, "
            "**Minimize Price Uncertainty Spread** ($\Delta P = \\text{Upper} - \\text{Lower}$ from Member 2), and **Minimize Saturation Ratio**."
        )
        topsis_list = approaches.get("topsis_mcdm", [])
        if topsis_list:
            df_topsis = pd.DataFrame([
                {
                    "Rank": idx + 1,
                    "Crop": item["crop"].title(),
                    "TOPSIS Closeness": f"{item['rank_score']:.2f}%",
                    "Distance to Best (S+)": item["dist_ideal_best"],
                    "Distance to Worst (S-)": item["dist_ideal_worst"],
                    "Price Uncertainty Spread": f"LKR {item['price_spread']:.1f}",
                    "Forecast Price": f"LKR {item['predicted_price']:.2f}",
                    "Suitability": f"{item['suitability']*100:.1f}%",
                    "Explanation": item["explanation"],
                }
                for idx, item in enumerate(topsis_list)
            ])
            st.dataframe(df_topsis, hide_index=True, use_container_width=True)

    # ------------------ Tab 3: BPR & Similarity ------------------
    with tab_bpr:
        st.markdown("#### Non-Linear Traffic Congestion (BPR) & Crop Profile Similarity")
        st.markdown(
            "Borrowing from GPS navigation algorithms (Waze/Google Maps), this approach replaces arbitrary linear thresholds with the "
            "**Bureau of Public Roads (BPR)** congestion function:  \n"
            "$$\\text{Congestion} = 1 - \\frac{1}{1 + 0.15 \\times (V/C)^4}$$  \n"
            "It computes **Cosine Vector Similarity** on operational dimensions (*growth duration, water demand, labor intensity, capital investment*) "
            "to find crops that require similar farming equipment and skills as the requested crop."
        )
        bpr_list = approaches.get("bpr_similarity", [])
        if bpr_list:
            df_bpr = pd.DataFrame([
                {
                    "Rank": idx + 1,
                    "Crop": item["crop"].title(),
                    "Composite BPR Score": f"{item['rank_score']:.2f}%",
                    "Profile Similarity to Requested": f"{item['profile_similarity']*100:.1f}%",
                    "BPR Market Congestion Penalty": f"{item['bpr_congestion']*100:.1f}%",
                    "Suitability": f"{item['suitability']*100:.1f}%",
                    "Forecast Price": f"LKR {item['predicted_price']:.2f}",
                    "Explanation": item["explanation"],
                }
                for idx, item in enumerate(bpr_list)
            ])
            st.dataframe(df_bpr, hide_index=True, use_container_width=True)

    # ------------------ Tab 4: Portfolio Land-Splitting ------------------
    with tab_portfolio:
        st.markdown("#### Markowitz-Inspired Land-Splitting & Risk Hedging")
        st.markdown(
            "Rather than treating crop selection as an all-or-nothing binary decision, finance portfolio theory hedges risk by **splitting land allocation**. "
            "The system caps planting of the saturated crop to the safe 85% capacity margin and reallocates remaining acreage to the top viable alternative."
        )
        if portfolio_info:
            p_cols = st.columns(len(portfolio_info.get("allocations", [])))
            for idx, alloc in enumerate(portfolio_info.get("allocations", [])):
                with p_cols[idx]:
                    st.metric(
                        f"Allocated: {alloc['crop'].title()}",
                        f"{alloc['allocated_area_ha']:.1f} ha ({alloc['allocation_pct']}%)",
                    )
                    st.caption(alloc["role"])

            st.info(f"💡 **Strategy Insight:** {portfolio_info.get('explanation', '')}")
            if portfolio_info.get("estimated_risk_reduction_pct", 0) > 0:
                st.success(f"🛡️ **Estimated Over-Saturation Risk Reduction:** {portfolio_info['estimated_risk_reduction_pct']:.1f}%")

    # ------------------ Tab 5: Softmax Load Balancing ------------------
    with tab_softmax:
        st.markdown("#### Game-Theoretic Herding Prevention (Boltzmann/Softmax)")
        st.markdown(
            "**The Research Problem:** If a recommendation engine directs 100% of farmers in a district to the same #1 alternative, "
            "it induces a secondary glut and causes the alternative crop's market to crash next season.  \n"
            "**Solution:** A randomized Softmax policy distributes recommendation traffic across viable crops proportionally to their viability score."
        )
        softmax_list = approaches.get("softmax_load_balancing", [])
        if softmax_list:
            chart_data = pd.DataFrame([
                {"Crop": item["crop"].title(), "Recommendation Probability (%)": item["rank_score"]}
                for item in softmax_list
            ])
            st.bar_chart(chart_data.set_index("Crop"), use_container_width=True)

            df_softmax = pd.DataFrame([
                {
                    "Crop": item["crop"].title(),
                    "Systemic Quota / Share": f"{item['rank_score']:.1f}%",
                    "Underlying Utility (DCVS)": f"{item['utility_dcvs']:.1f}%",
                    "Macroeconomic Role": item["explanation"],
                }
                for item in softmax_list
            ])
            st.dataframe(df_softmax, hide_index=True, use_container_width=True)

    # ------------------ Tab 6: Baseline DCVS ------------------
    with tab_baseline:
        st.markdown("#### Baseline Dynamic Crop Viability Score (DCVS)")
        st.markdown(
            "Uses piecewise linear MSRS (0.0 to 1.0 between 80% and 120% capacity) and computes:  \n"
            "$$\\text{DCVS} = (0.34 \\times P_{\\text{norm}}) + (0.33 \\times S) + (0.33 \\times (1 - \\text{MSRS}))$$"
        )
        baseline_list = approaches.get("baseline_dcvs", [])
        if baseline_list:
            df_base = pd.DataFrame([
                {
                    "Rank": idx + 1,
                    "Crop": item["crop"].title(),
                    "DCVS": f"{item['rank_score']:.1f}%",
                    "Pressure Score": f"{item['pressure_risk']:.3f}",
                    "Pressure Label": item["pressure_label"].title(),
                    "Forecast Price": f"LKR {item['predicted_price']:.2f}",
                    "Suitability": f"{item['suitability']*100:.1f}%",
                    "Explanation": item["explanation"],
                }
                for idx, item in enumerate(baseline_list)
            ])
            st.dataframe(df_base, hide_index=True, use_container_width=True)

    # ------------------ Tab 7: Climate Suitability Breakdown ------------------
    with tab_climate:
        st.markdown("#### District Environmental Factors & Agronomic Mapping")
        st.markdown(
            f"Suitability scores for **{selected_district.replace('_', ' ').title()}** computed from elevation, temperature, and rainfall."
        )
        dist_profiles = result.get("district_environmental_profiles", {})
        if dist_profiles:
            climate_rows = []
            for c_name, profile_data in dist_profiles.items():
                subs = profile_data.get("sub_scores", {})
                env = profile_data.get("environmental_factors", {})
                climate_rows.append({
                    "Crop": c_name.title(),
                    "Composite Suitability": f"{profile_data.get('composite_suitability', 0)*100:.1f}%",
                    "Altitude Match": f"{subs.get('altitude', 0)*100:.1f}%",
                    "Temperature Match": f"{subs.get('temperature', 0)*100:.1f}%",
                    "Rainfall Match": f"{subs.get('rainfall', 0)*100:.1f}%",
                    "District Altitude": f"{env.get('altitude_m', 0):.0f} m",
                    "District Mean Temp": f"{env.get('mean_temp_c', 0):.1f} °C",
                    "District Rainfall": f"{env.get('seasonal_rainfall_mm', 0):.0f} mm",
                })
            st.dataframe(pd.DataFrame(climate_rows), hide_index=True, use_container_width=True)

    # ---------------------------------------------------------
    # 4. Planting Confirmation & Persistence Section
    # ---------------------------------------------------------
    st.divider()
    st.subheader("4. Confirm Planting Commitment")
    st.caption("Assessments are exploratory. Confirmation commits the planting record into SQLite, updating the market aggregate.")

    with st.form("commitment_confirmation"):
        c_left, c_right = st.columns(2)
        with c_left:
            st.text_input("Farmer Reference", value=farmer_ref, disabled=True)
            st.text_input("Confirmation Idempotency Key", value=st.session_state.confirmation_key, disabled=True)
            chosen_crop = st.selectbox(
                "Final Confirmed Crop Decision",
                SUPPORTED_CROPS,
                index=SUPPORTED_CROPS.index(crop) if crop in SUPPORTED_CROPS else 0,
                format_func=lambda x: x.title(),
                help="You can either confirm your original crop or switch to a recommended alternative.",
            )
        with c_right:
            planting_date = st.date_input("Planting Date", value=date.today())
            harvest_date = st.date_input("Expected Harvest Date")
            confirmed_check = st.checkbox("I confirm this planting commitment on behalf of the farmer")

        submit_commitment = st.form_submit_button("💾 Save Confirmed Commitment", use_container_width=True)

    if submit_commitment:
        commitment_payload = {
            "district": selected_district,
            "market": market,
            "crop": chosen_crop,
            "season": season,
            "proposed_area_ha": area,
            "horizon_weeks": horizon,
            "suitability": suitability_inputs,
            "idempotency_key": st.session_state.confirmation_key,
            "farmer_reference": farmer_ref,
            "planting_date": str(planting_date),
            "harvest_date": str(harvest_date),
            "confirmed": confirmed_check,
        }
        try:
            c_resp = requests.post(f"{API_URL}/commitments", json=commitment_payload, timeout=10)
            if c_resp.ok:
                st.success(
                    f"🎉 Commitment #{c_resp.json().get('id')} successfully confirmed and saved to SQLite! "
                    f"Future assessments in {selected_district.title()} will reflect this +{area} ha."
                )
                st.session_state.confirmation_key = f"demo-{uuid.uuid4().hex[:12]}"
            else:
                st.error(f"Commitment rejected: {c_resp.text}")
        except Exception as ex:
            st.error(f"Failed to save commitment: {ex}")
