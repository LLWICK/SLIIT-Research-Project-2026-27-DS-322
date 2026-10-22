"""Interactive crop recommendation dashboard for Member 3 prototype.

Demonstrates:
1. Agro-climatic suitability mapped from planting areas (altitude, rainfall, climate, temperature, soil).
2. Market Saturation Risk Score (MSRS) evaluation for requested crops.
3. Simplified Recommendation Workflow:
   - Primary Recommendation (Safe Cultivation).
   - Recommended Alternative (Triggered only when Market Saturation Risk is High).
   - Commitment Registration with SQLite Persistence.
4. Optional Advanced Multi-Paradigm Research Expander (TOPSIS, BPR, Softmax, Weight Trainer).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Member3Data"
DB_PATH = Path(os.getenv("MEMBER3_DB_PATH", ROOT / "member3_commitments.sqlite3"))

# Local module imports for robust offline fallback
from member3.config import ScoringConfig
from member3.forecast_adapter import ForecastAdapter
from member3.repository import CommitmentRepository
from member3.service import RecommendationService
from member3.suitability import PLANTING_AREAS, calculate_suitability, get_area_suitability_map

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

# Custom CSS for clean, premium styling
st.markdown(
    """
    <style>
    .main-card {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
    }
    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-safe {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .badge-danger {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .badge-alt {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        background-color: #e0e7ff;
        color: #3730a3;
        border: 1px solid #a5b4fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌾 Crowding-Aware Crop Recommendation System")
st.markdown(
    "**SLIIT Research Project J26-DS-322 | Member 3 Prototype**  \n"
    "*Proactively evaluates market saturation risk to prevent the Cobweb price crash and recommends optimal crop alternatives.*"
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
    suitability_defaults: dict[str, float] = {}
    try:
        resp = requests.get(f"{API_URL}/planting-areas/{selected_district}", timeout=2)
        if resp.ok:
            area_data = resp.json()
            area_profile = area_data["profile"]
            suitability_defaults = {
                crop: area_data["suitability_breakdown"][crop]["composite_suitability"]
                for crop in SUPPORTED_CROPS
            }
        else:
            raise ValueError("API returned non-200")
    except Exception:
        # Seamless local fallback
        key = selected_district.strip().lower().replace(" ", "_")
        if key in PLANTING_AREAS:
            area_profile = asdict(PLANTING_AREAS[key])
            suitability_defaults = {
                c: calculate_suitability(c, key)["composite_suitability"]
                for c in SUPPORTED_CROPS
            }
        else:
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
        st.info("Automated environmental suitability mapping:")
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
# 1. Farmer Proposal Form
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
        assessed = st.form_submit_button("🔍 Evaluate Viability & Recommendation", use_container_width=True)

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
    with st.spinner("Evaluating market commitments, Member 2 price forecasts, and saturation risk..."):
        # Attempt API request first, fallback to local engine seamlessly
        assessment_done = False
        try:
            response = requests.post(f"{API_URL}/recommendations/assess", json=assessment_payload, timeout=3)
            if response.ok:
                st.session_state.assessment = response.json()
                assessment_done = True
        except Exception:
            pass

        if not assessment_done:
            try:
                repo = CommitmentRepository(DB_PATH)
                adapter = ForecastAdapter(DATA_DIR / "member3_forecast_c.json")
                local_service = RecommendationService(repo, adapter)
                st.session_state.assessment = local_service.assess(assessment_payload)
            except Exception as ex:
                st.error(f"Failed to compute recommendations: {ex}")

# ---------------------------------------------------------
# 2. Results Section: Recommendation & Alternative
# ---------------------------------------------------------
if "assessment" in st.session_state:
    result = st.session_state.assessment
    requested = result["requested_crop_assessment"]
    approaches = result.get("approaches", {})
    pressure_info = requested["pressure"]
    pressure_score = pressure_info["score"]
    pressure_ratio = pressure_info["pressure_ratio"]
    pressure_label = pressure_info["label"]

    # Saturation risk condition: score >= 0.7 or ratio >= 1.2 or label == "high"
    is_high_risk = pressure_score >= 0.7 or pressure_ratio >= 1.2 or pressure_label == "high"

    st.divider()

    # Identify best alternative among other crops
    candidates = result.get("recommendations", [])
    alternatives = [item for item in candidates if item["crop"] != crop]
    # Rank alternatives by lowest saturation risk, then highest viability score (DCVS)
    alternatives.sort(key=lambda x: (x["pressure"]["score"] < 0.7, x["dcvs"]), reverse=True)
    best_alt = alternatives[0] if alternatives else None

    # CASE A: Saturation Risk is NOT High -> Just show Recommendation
    if not is_high_risk:
        st.subheader("2. Recommendation")
        st.success(
            f"✅ **RECOMMENDATION: PROCEED WITH {crop.upper()}**  \n\n"
            f"**Safe Cultivation Zone:** Current market commitments plus your proposed area "
            f"({requested['proposed_total_area_ha']:.1f} ha) will utilize **{pressure_ratio * 100:.1f}%** of market capacity "
            f"({requested['reference_capacity_ha']:.0f} ha).  \n"
            f"The Market Saturation Risk is **{pressure_score:.2f} ({pressure_label.title()})**, indicating healthy market absorption "
            f"with negligible risk of Cobweb price crash."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended Crop", crop.title())
        m2.metric("Dynamic Viability Score (DCVS)", f"{requested['dcvs']:.1f}%")
        m3.metric("Forecast Price (Member 2)", f"LKR {requested['forecast']['predicted_price']:.2f}/kg")
        m4.metric("Agro-Climatic Suitability", f"{requested['suitability'] * 100:.1f}%")

    # CASE B: Saturation Risk IS High -> Show Recommendation Warning AND Recommended Alternative
    else:
        st.subheader("2. Recommendation & Alternative")
        st.error(
            f"🚨 **RECOMMENDATION: DO NOT PLANT {crop.upper()} — HIGH MARKET SATURATION RISK DETECTED**  \n\n"
            f"Current commitments plus your proposed area ({requested['proposed_total_area_ha']:.1f} ha) will reach "
            f"**{pressure_ratio * 100:.1f}%** of market capacity ({requested['reference_capacity_ha']:.0f} ha).  \n"
            f"Risk Score: **{pressure_score:.2f} (High / Critical)**. Planting this crop will induce the "
            f"**Cobweb oversupply crash**, leading to severe wholesale price collapse at terminal markets."
        )

        if best_alt:
            alt_crop = best_alt["crop"]
            alt_pressure = best_alt["pressure"]
            alt_forecast = best_alt["forecast"]

            st.markdown(
                f"""
                <div style="background-color: #f0fdf4; border: 2px solid #86efac; border-radius: 10px; padding: 1.25rem; margin-top: 1rem; margin-bottom: 1.25rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <h3 style="margin: 0; color: #166534;">🔄 Recommended Alternative: Switch to {alt_crop.upper()}</h3>
                        <span class="badge-safe">Safe Capacity & High Viability</span>
                    </div>
                    <p style="margin-top: 0.5rem; color: #14532d; font-size: 1rem;">
                        Switching to <strong>{alt_crop.title()}</strong> protects the farmer from wholesale price collapse while ensuring profitable farmgate returns.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Comparative Metrics Table
            st.markdown("#### ⚖️ Decision Comparison: Requested vs Recommended Alternative")
            comp_df = pd.DataFrame([
                {
                    "Decision Factor": "Market Saturation Risk",
                    f"Requested: {crop.title()}": f"🚨 High ({pressure_score:.2f})",
                    f"Recommended Alternative: {alt_crop.title()}": f"✅ {alt_pressure['label'].title()} ({alt_pressure['score']:.2f})",
                },
                {
                    "Decision Factor": "Market Capacity Utilization",
                    f"Requested: {crop.title()}": f"{pressure_ratio * 100:.1f}% (Over-Capacity)",
                    f"Recommended Alternative: {alt_crop.title()}": f"{alt_pressure['pressure_ratio'] * 100:.1f}% (Healthy / Safe)",
                },
                {
                    "Decision Factor": "Forecast Wholesale Price",
                    f"Requested: {crop.title()}": f"LKR {requested['forecast']['predicted_price']:.2f} / kg",
                    f"Recommended Alternative: {alt_crop.title()}": f"LKR {alt_forecast['predicted_price']:.2f} / kg",
                },
                {
                    "Decision Factor": "Agro-Climatic Suitability",
                    f"Requested: {crop.title()}": f"{requested['suitability'] * 100:.1f}%",
                    f"Recommended Alternative: {alt_crop.title()}": f"{best_alt['suitability'] * 100:.1f}%",
                },
                {
                    "Decision Factor": "Dynamic Viability Score (DCVS)",
                    f"Requested: {crop.title()}": f"{requested['dcvs']:.1f}%",
                    f"Recommended Alternative: {alt_crop.title()}": f"{best_alt['dcvs']:.1f}%",
                },
            ])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

            # Hedging insight
            portfolio_info = approaches.get("portfolio_hedging", {})
            if portfolio_info and portfolio_info.get("allocations"):
                allocs = portfolio_info.get("allocations", [])
                if len(allocs) > 1:
                    alloc_req = next((a for a in allocs if a["crop"] == crop), None)
                    alloc_alt = next((a for a in allocs if a["crop"] != crop), None)
                    if alloc_req and alloc_alt:
                        st.info(
                            f"💡 **Land-Splitting Hedging Alternative:** If you still wish to grow some {crop.title()}, you can hedge your risk: "
                            f"plant **{alloc_req['allocated_area_ha']:.1f} ha of {crop.title()}** (safe quota) and **{alloc_alt['allocated_area_ha']:.1f} ha of {alloc_alt['crop'].title()}** "
                            f"(hedging crop) to reduce saturation risk by **{portfolio_info.get('estimated_risk_reduction_pct', 0):.1f}%**."
                        )

    # ---------------------------------------------------------
    # 3. Confirm Planting Commitment (In the end)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("3. Confirm Planting Commitment")
    st.caption("Record your confirmed planting in the SQLite market ledger to guarantee aggregate coordination.")

    # Determine default crop choice
    if is_high_risk and best_alt:
        crop_options = [best_alt["crop"], crop] + [c for c in SUPPORTED_CROPS if c not in (best_alt["crop"], crop)]
    else:
        crop_options = [crop] + [c for c in SUPPORTED_CROPS if c != crop]

    def format_crop_option(c: str) -> str:
        if is_high_risk and best_alt and c == best_alt["crop"]:
            return f"{c.title()} (Recommended Alternative - Safe)"
        elif is_high_risk and c == crop:
            return f"{c.title()} (Original Requested - High Saturation Risk)"
        elif not is_high_risk and c == crop:
            return f"{c.title()} (Recommended - Safe)"
        return c.title()

    with st.form("commitment_confirmation"):
        c_left, c_right = st.columns(2)
        with c_left:
            st.text_input("Farmer Reference", value=farmer_ref, disabled=True)
            chosen_crop = st.selectbox(
                "Selected Crop for Commitment",
                crop_options,
                index=0,
                format_func=format_crop_option,
                help="Pre-selected to the optimal recommendation. You may change this if desired.",
            )
            committed_area = st.number_input("Confirmed Area (ha)", min_value=0.5, value=float(area), step=0.5)
        with c_right:
            planting_date = st.date_input("Planting Date", value=date.today())
            harvest_date = st.date_input("Expected Harvest Date", value=date.today() + pd.Timedelta(days=90))
            confirmed_check = st.checkbox("I confirm this planting commitment on behalf of the farmer", value=True)

        submit_commitment = st.form_submit_button("💾 Save Confirmed Commitment", use_container_width=True)

    if submit_commitment:
        if not confirmed_check:
            st.warning("Please check the confirmation box before saving.")
        else:
            commitment_payload = {
                "district": selected_district,
                "market": market,
                "crop": chosen_crop,
                "season": season,
                "proposed_area_ha": committed_area,
                "horizon_weeks": horizon,
                "suitability": suitability_inputs,
                "idempotency_key": st.session_state.confirmation_key,
                "farmer_reference": farmer_ref,
                "planting_date": str(planting_date),
                "harvest_date": str(harvest_date),
                "confirmed": True,
            }
            saved_id = None
            try:
                c_resp = requests.post(f"{API_URL}/commitments", json=commitment_payload, timeout=3)
                if c_resp.ok:
                    saved_id = c_resp.json().get("id")
            except Exception:
                pass

            if saved_id is None:
                # Local SQLite persistence fallback
                try:
                    repo = CommitmentRepository(DB_PATH)
                    record = repo.create({
                        "idempotency_key": st.session_state.confirmation_key,
                        "farmer_reference": farmer_ref,
                        "district": selected_district,
                        "market": market,
                        "crop": chosen_crop,
                        "season": season,
                        "planting_date": str(planting_date),
                        "harvest_date": str(harvest_date),
                        "area_ha": committed_area,
                        "status": "confirmed",
                    })
                    saved_id = record.get("id")
                except Exception as ex:
                    st.error(f"Failed to record commitment: {ex}")

            if saved_id:
                st.success(
                    f"🎉 **Commitment #{saved_id} successfully confirmed and registered in SQLite!**  \n"
                    f"Registered **{committed_area:.1f} ha** of **{chosen_crop.title()}** for farmer **{farmer_ref}** "
                    f"in **{selected_district.replace('_', ' ').title()}** ({season} season).  \n"
                    f"Market aggregate commitments are now updated."
                )
                st.session_state.confirmation_key = f"demo-{uuid.uuid4().hex[:12]}"

    # ---------------------------------------------------------
    # 4. Optional: Multi-Paradigm Research Models (Collapsed)
    # ---------------------------------------------------------
    st.write("")
    with st.expander("🔬 View Advanced Multi-Paradigm Research Models & Calibration (Optional)", expanded=False):
        st.caption("Inspect and compare how 5 distinct academic and operational solutions formulate crop recommendations.")

        tab_overview, tab_topsis, tab_bpr, tab_portfolio, tab_softmax, tab_baseline, tab_climate, tab_trainer = st.tabs([
            "📊 Approach Comparison Matrix",
            "🎯 MCDM / TOPSIS",
            "🚜 Non-Linear BPR & Crop Similarity",
            "💼 Portfolio Land-Splitting",
            "🌐 Softmax Load Balancing",
            "⚖️ Baseline Weighted DCVS",
            "🗺️ Climate Suitability Breakdown",
            "🧠 Data-Driven Weight Trainer",
        ])

        # Tab 1: Comparison Matrix
        with tab_overview:
            st.markdown("#### Cross-Paradigm Recommendation Summary")
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

        # Tab 2: TOPSIS
        with tab_topsis:
            st.markdown("#### Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)")
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

        # Tab 3: BPR & Similarity
        with tab_bpr:
            st.markdown("#### Non-Linear Traffic Congestion (BPR) & Crop Profile Similarity")
            bpr_list = approaches.get("bpr_similarity", [])
            if bpr_list:
                df_bpr = pd.DataFrame([
                    {
                        "Rank": idx + 1,
                        "Crop": item["crop"].title(),
                        "Composite BPR Score": f"{item['rank_score']:.2f}%",
                        "Profile Similarity": f"{item['profile_similarity']*100:.1f}%",
                        "BPR Congestion Penalty": f"{item['bpr_congestion']*100:.1f}%",
                        "Suitability": f"{item['suitability']*100:.1f}%",
                        "Forecast Price": f"LKR {item['predicted_price']:.2f}",
                        "Explanation": item["explanation"],
                    }
                    for idx, item in enumerate(bpr_list)
                ])
                st.dataframe(df_bpr, hide_index=True, use_container_width=True)

        # Tab 4: Portfolio Land-Splitting
        with tab_portfolio:
            st.markdown("#### Markowitz-Inspired Land-Splitting & Risk Hedging")
            if portfolio_info:
                p_cols = st.columns(len(portfolio_info.get("allocations", [])))
                for idx, alloc in enumerate(portfolio_info.get("allocations", [])):
                    with p_cols[idx]:
                        st.metric(
                            f"Allocated: {alloc['crop'].title()}",
                            f"{alloc['allocated_area_ha']:.1f} ha ({alloc['allocation_pct']}%)",
                        )
                        st.caption(alloc["role"])
                st.info(f"💡 Strategy Insight: {portfolio_info.get('explanation', '')}")

        # Tab 5: Softmax Load Balancing
        with tab_softmax:
            st.markdown("#### Game-Theoretic Herding Prevention (Softmax Load Balancing)")
            softmax_list = approaches.get("softmax_load_balancing", [])
            if softmax_list:
                chart_data = pd.DataFrame([
                    {"Crop": item["crop"].title(), "Recommendation Probability (%)": item["rank_score"]}
                    for item in softmax_list
                ])
                st.bar_chart(chart_data.set_index("Crop"), use_container_width=True)

        # Tab 6: Baseline DCVS
        with tab_baseline:
            st.markdown("#### Baseline Dynamic Crop Viability Score (DCVS)")
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

        # Tab 7: Climate Suitability Breakdown
        with tab_climate:
            st.markdown(f"#### District Environmental Factors & Agronomic Mapping ({selected_district.replace('_', ' ').title()})")
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

        # Tab 8: Weight Trainer & Calibration
        with tab_trainer:
            st.markdown("#### 🧠 Data-Driven Algorithm Weight Derivation & Model Calibration")
            train_col1, _ = st.columns([1, 3])
            with train_col1:
                run_training = st.button("🚀 Run Multi-Model Weight Calibration", use_container_width=True)

            if "trained_weights" not in st.session_state and not run_training:
                try:
                    tw_resp = requests.get(f"{API_URL}/weights/trained", timeout=3)
                    if tw_resp.ok:
                        st.session_state.trained_weights = tw_resp.json()
                except Exception:
                    pass

            if run_training:
                with st.spinner("Processing Vegetable Prices.xlsx and calibrating weights..."):
                    try:
                        from member3.weight_trainer import train_all_models
                        st.session_state.trained_weights = train_all_models()
                        st.success("Calibration complete across all 4 paradigms!")
                    except Exception as ex:
                        st.error(f"Training failed: {ex}")

            if "trained_weights" in st.session_state:
                tw_data = st.session_state.trained_weights
                p_stats = tw_data.get("price_statistics", {})
                models = tw_data.get("derived_models", {})

                st.markdown("##### 1. Empirical Wholesale Price Volatility")
                stat_rows = [
                    {
                        "Crop": crop_name.title(),
                        "Weekly Price Records": stat.get("samples", 0),
                        "Mean Price (Rs/kg)": f"LKR {stat.get('mean', 0):.2f}",
                        "Std. Deviation": f"LKR {stat.get('stdev', 0):.2f}",
                        "Historical Price Range": f"LKR {stat.get('min', 0):.1f} - {stat.get('max', 0):.1f}",
                    }
                    for crop_name, stat in p_stats.items()
                ]
                st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)
