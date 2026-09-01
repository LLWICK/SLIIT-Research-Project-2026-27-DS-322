import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from engine import calculate_msrs, calculate_dcvs, check_crash_risk

# Page Configuration
st.set_page_config(
    page_title="Farmer Advisory DSS (Member 3)",
    page_icon="👨‍🌾",
    layout="wide"
)

# Load Dummy Data
@st.cache_data
def load_dummy_data():
    with open("dummy_data.json", "r") as f:
        return json.load(f)

dummy_data = load_dummy_data()
crops_data = dummy_data["crops"]

# Styling
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .safe-alert {
        background: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 8px;
    }
    .warn-alert {
        background: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 15px;
        border-radius: 8px;
    }
    .critical-alert {
        background: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
    }
    .recommendation-card {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("👨‍🌾 Farmer Decision Support System (DSS)")
st.caption("Member 3: Senaratna S.T.S | SLIIT Research Project J26-DS-322")

st.markdown("""
Welcome to the Farmer Advisory portal. Enter the crop you intend to plant and the land extent. 
The system will analyze current national commitments and provide a **Market Saturation Risk Score (MSRS)** and 
a **Dynamic Crop Viability Score (DCVS)**. If your chosen crop is at risk of oversupply, we will recommend a safer alternative.
""")

st.sidebar.header("⚙️ Market State Simulation")
st.sidebar.markdown("Adjust these sliders to simulate the *current* market conditions before the farmer plants.")
current_market_commitments = {}
for crop, data in crops_data.items():
    current_market_commitments[crop] = st.sidebar.slider(
        f"{crop} Current Market Commitment (ha)",
        min_value=0.0,
        max_value=data["market_demand_capacity_ha"] * 1.5,
        value=data["market_demand_capacity_ha"] * 0.6,
        step=50.0
    )

# --- Farmer Input Section ---
st.subheader("1. Your Planting Intentions")
col_in1, col_in2 = st.columns(2)
with col_in1:
    selected_crop = st.selectbox("Target Crop to Plant", list(crops_data.keys()))
with col_in2:
    farmer_area = st.number_input("Amount to Plant (Hectares)", min_value=0.1, value=5.0, step=0.5)

# --- Calculations for Target Crop ---
crop_baseline = crops_data[selected_crop]
demand_capacity = crop_baseline["market_demand_capacity_ha"]
weather_index = crop_baseline["weather_suitability_base"]
price_index = crop_baseline["price_forecast_base"]

# Calculate new total commitment if farmer plants this
new_total_area = current_market_commitments[selected_crop] + farmer_area

msrs_val = calculate_msrs(new_total_area, demand_capacity)
dcvs_val = calculate_dcvs(msrs_val, weather_index, price_index)
risk_assessment = check_crash_risk(msrs_val)

st.markdown("---")
st.subheader("2. Market Viability Analysis")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("New Total Market Commitment", f"{new_total_area:,.1f} ha", f"Capacity: {demand_capacity:,.1f} ha", delta_color="off")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Market Saturation Risk (MSRS)", f"{msrs_val:.3f}", "0.0 is Safe, 1.0 is Critical", delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Dynamic Viability Score (DCVS)", f"{dcvs_val}%", "Higher is better")
    st.markdown('</div>', unsafe_allow_html=True)

# Risk Assessment & Recommendation
if msrs_val >= 0.7:
    st.markdown(f'<div class="critical-alert"><b>🚨 {risk_assessment}</b><br>High risk of market crash due to oversupply.</div>', unsafe_allow_html=True)
    
    st.subheader("💡 Recommended Alternatives")
    st.markdown("We scanned other crops based on current market data. Here is the best alternative to plant instead:")
    
    # Calculate DCVS for all other crops
    best_alt_crop = None
    best_alt_dcvs = -1
    best_alt_msrs = 1.0
    
    for alt_crop in crops_data.keys():
        if alt_crop == selected_crop:
            continue
            
        alt_baseline = crops_data[alt_crop]
        alt_demand = alt_baseline["market_demand_capacity_ha"]
        alt_weather = alt_baseline["weather_suitability_base"]
        alt_price = alt_baseline["price_forecast_base"]
        
        # Add farmer's area to alt crop's current commitment
        alt_new_total = current_market_commitments[alt_crop] + farmer_area
        
        alt_msrs = calculate_msrs(alt_new_total, alt_demand)
        alt_dcvs = calculate_dcvs(alt_msrs, alt_weather, alt_price)
        
        if alt_dcvs > best_alt_dcvs:
            best_alt_dcvs = alt_dcvs
            best_alt_crop = alt_crop
            best_alt_msrs = alt_msrs
            
    if best_alt_crop:
        st.markdown(f"""
        <div class="recommendation-card">
            <h3 style="margin-top:0;">🌱 Switch to: {best_alt_crop}</h3>
            <p>If you plant <b>{farmer_area} ha</b> of {best_alt_crop} instead:</p>
            <ul>
                <li><b>New MSRS:</b> {best_alt_msrs:.3f} (Safe)</li>
                <li><b>New DCVS:</b> {best_alt_dcvs}% (Highly Viable)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
elif msrs_val >= 0.3:
    st.markdown(f'<div class="warn-alert"><b>⚠️ {risk_assessment}</b><br>Monitor closely. DCVS is dropping.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="safe-alert"><b>✅ {risk_assessment}</b><br>Good to plant. Market demand is healthy.</div>', unsafe_allow_html=True)

st.markdown("---")

# Visualizations
tab1, tab2 = st.tabs(["📊 Saturation Gauges", "🔌 Mocked REST API"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_msrs = go.Figure(go.Indicator(
            mode = "gauge+number", value = msrs_val, title = {'text': "MSRS (Risk of Oversupply)"},
            gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 0.3], 'color': "#10b981"}, {'range': [0.3, 0.7], 'color': "#f59e0b"}, {'range': [0.7, 1.0], 'color': "#ef4444"}]}
        ))
        st.plotly_chart(fig_msrs, use_container_width=True)

    with col_g2:
        fig_dcvs = go.Figure(go.Indicator(
            mode = "gauge+number", value = dcvs_val, title = {'text': "DCVS (Overall Recommendation %)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 40], 'color': "#ef4444"}, {'range': [40, 70], 'color': "#f59e0b"}, {'range': [70, 100], 'color': "#10b981"}]}
        ))
        st.plotly_chart(fig_dcvs, use_container_width=True)

with tab2:
    st.subheader("Mocked REST API Payload")
    api_payload = {
        "farmer_request": {
            "crop": selected_crop,
            "intended_area_ha": farmer_area
        },
        "market_state": {
            "current_commitment_ha": current_market_commitments[selected_crop],
            "market_demand_capacity_ha": demand_capacity
        },
        "outputs": {
            "MSRS": msrs_val,
            "DCVS_percentage": dcvs_val,
            "risk_assessment": risk_assessment
        }
    }
    st.json(api_payload)
