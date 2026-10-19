# Geo-Spatial Decision Support System (DSS) Dashboard

**SLIIT Research Project 2026/27 (DS-322)**  
**Research Title**: *Data-Driven Decision Support System to Mitigate the Cobweb Effect in Sri Lankan Vegetable Markets*  
**Member 1 Responsibility**: Data Engineering and Geo-Spatial Decision Support System (DSS)  

---

## 1. Executive Summary

This interactive dashboard serves as the operational command center for **Member 1**, synthesizing real agricultural price reports, continuous NASA POWER meteorological streams, and regional farmer planting commitments to identify and mitigate the Cobweb Effect across Sri Lanka's 25 administrative districts.

### Key Capabilities:
- **Sri Lanka District Geo-Spatial Mapping**: Visualizes real-time Market Saturation Risk (MSRS) and Dynamic Crop Viability (DCVS) across all 25 districts using Plotly.
- **Cobweb Effect Phase Dynamics**: Plots cyclical phase spirals (Price vs. Supply Extent) confirming oversupply oscillations.
- **Evidence-Based Farmer Advisory**: Provides transparent planting guidance (Favorable, Caution, Diversify) with alternative crop suggestions.
- **Data Readiness Index (DRI) Quality Audit**: Direct integration with the empirical DRI audit ledger (`data/processed/prices/dri_audit_ledger.jsonl`).
- **Member 2 Forecasting Adapter**: Decoupled contract bridge for forward price forecasts with uncertainty bands, with clearly watermarked demo simulations.

---

## 2. Research Data Schemas

The dashboard directly interfaces with the primary research data streams without duplication:

| Stream | Local File Path | Records | Key Schema Fields |
| :--- | :--- | :--- | :--- |
| **Weekly Prices** | `data/v1/analysis_ready_price.csv` | 110,019 | `price_date` (Monday), `district`, `crop`, `price_type` (`retail`/`wholesale`), `granularity` (`Retail - W`), `price` (Rs./kg), `source` (`harti`) |
| **Commitments** | `data/v1/analysis_ready_commitment.csv` | 2,875 | `district`, `crop_name`, `season_type` (`Maha`/`Yala`), `season_year` (2014–2025), `target_hectares`, `committed_hectares`, `source` (`synthetic_proxy`) |
| **Weather** | `data/v1/weather_data.json` | 100,225 | `district_name_raw`, `obs_date`, `rainfall_mm`, `temp_c`, `source` (`nasa_power`) |
| **DRI Audits** | `data/processed/prices/dri_audit_ledger.jsonl` | Audited | `calculated_dri`, `reliability_tier`, `source_agency`, `completeness_score`, `timeliness_score`, `consistency_score` |

---

## 3. Windows PowerShell Setup & Local Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12 (or Anaconda3 environment)
- Windows PowerShell

### Step 1: Open PowerShell and Navigate to the Repository Root
```powershell
cd c:\Users\SAKUNI\Desktop\Research_Project\SLIIT-Research-Project-2026-27-DS-322
```

### Step 2: (Optional) Install Dependencies
If running in a fresh virtual environment:
```powershell
pip install -r geospatial-dss-dashboard/requirements.txt
```

### Step 3: Launch the Streamlit Dashboard
Execute the following PowerShell command:
```powershell
C:\Users\SAKUNI\anaconda3\python.exe -m streamlit run geospatial-dss-dashboard/app.py --server.port 8501
```
*(Or if standard `streamlit` is in your PATH)*:
```powershell
streamlit run geospatial-dss-dashboard/app.py --server.port 8501
```

### Step 4: Open in Web Browser
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 4. Architecture & Module Structure

```
geospatial-dss-dashboard/
├── app.py                      # Main executive Streamlit application
├── config.py                   # 25 district coordinates, crop palettes, CSS tokens
├── data_loader.py              # Cached loaders for price, commitment, weather, and DRI ledger
├── requirements.txt            # Dedicated dependencies
├── README.md                   # Setup, execution, and research documentation
├── engine/
│   ├── __init__.py
│   ├── msrs_engine.py          # Market Saturation Risk Score computation (Member 3 bridge)
│   ├── dcvs_engine.py          # Dynamic Crop Viability Score computation (Member 3 bridge)
│   ├── dri_bridge.py           # Data Readiness Index auditor (Member 1 quality bridge)
│   ├── forecast_bridge.py      # Price forecasting adapter (Member 2 bridge)
│   └── advisor.py              # Farmer advisory engine with alternative crop routing
├── components/
│   ├── __init__.py
│   ├── maps.py                 # Plotly Sri Lanka district geospatial maps
│   ├── charts.py               # Price trends, Cobweb phase plot, commitment bars, weather trends
│   ├── metrics_cards.py        # Executive KPI widgets with provenance tags
│   └── recommendations.py      # Farmer advisory UI cards
└── tests/
    ├── __init__.py
    └── test_dashboard_modules.py # Automated test suite for dashboard engines & adapters
```

---

## 5. Research Provenance & Transparency Standards

In accordance with academic rigor for SLIIT research viva:
- **`REAL DATA`**: High-frequency weekly prices (HARTI) and daily weather (NASA POWER).
- **`SYNTHETIC PROXY`**: Planning commitment quotas (`analysis_ready_commitment.csv`) simulating farmer registrations.
- **`PENDING ADAPTER`**: Member 2 econometric forecasting bridge.
- **`DEMO DATA`**: Clearly watermarked simulation placeholders toggled during live viva demonstrations.
