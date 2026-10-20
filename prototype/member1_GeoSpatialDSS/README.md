# Geo-Spatial Decision Support System (DSS) Dashboard

**SLIIT Research Project 2026/27 (J26-DS-322)**
**Research Title**: *Data-Driven Decision Support System to Mitigate the Cobweb Effect in Sri Lankan Vegetable Markets*
**Collaboration**: Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI) collaboration | SLIIT | J26-DS-322
**Component**: Member 1 — Data Engineering & Geo-Spatial Decision Support System

---

## 1. Executive Summary

This research-grade Geo-Spatial Decision Support System (DSS) serves as the operational intelligence dashboard for analyzing agricultural market dynamics and mitigating the Cobweb Effect across Sri Lanka's 25 administrative districts.

The dashboard integrates real weekly retail and wholesale market prices from HARTI, continuous NASA POWER daily agro-meteorological records, and seasonal cultivation planning records into standardized indices:
- **Market Saturation Risk Score (MSRS)** (0.0–100.0): Quantifies oversupply risk against regional market demand capacity.
- **Dynamic Crop Viability Score (DCVS)** (0.0–100.0): Balances agronomic weather favorability, price momentum, and market capacity.
- **Data Reliability Index (DRI)** (0.0–100.0): Core empirical quality audit based on Completeness (40%), Recency (30%), and Consistency (30%).

---

## 2. Dashboard Structure (6 Research Tabs)

The dashboard provides six decision-focused, professional views (without emojis):
1. **Overview**: Executive summary with 5 standardized KPI cards (`Market Price`, `MSRS`, `DCVS`, `DRI`, `Confidence`), district saturation preview, and evidence-based advisory banner.
2. **Market Saturation**: Interactive Plotly district map (sized by cultivation density, colored by MSRS or DRI), MSRS/DRI layer toggle, CSV export, and complete 25-district quota breakdown table.
3. **Price Dynamics**: Chronologically sorted weekly price trends with 8-week moving average (computed only when $\ge 8$ observations exist), price spread analysis, and Cobweb phase-space plot with clear label spacing.
4. **Agro-Meteorology**: District-specific weekly rainfall accumulation, mean temperature, and seasonal distribution from NASA POWER records.
5. **Recommendations**: Sortable multi-crop planning table across all 5 research crops (`Crop | MSRS | DCVS | DRI | Confidence | Recommendation | Key Reason`), actionable farmer guidance with 2–3 alternative crops, and district drill-down panel.
6. **Data Reliability**: Empirical 3-component DRI breakdown (Completeness 0.4, Recency 0.3, Consistency 0.3), category audits (Price, Weather, Cultivation, and neutral unavailable for Yield), dynamic "Why this score?" explanation, and 25-district reliability table.
---

## 3. Standardized Scale & Risk Bands

All primary indicators are standardized to a continuous **0.0 to 100.0** scale, displayed to one decimal place:

| Risk Band | Scale Range | Saturation Status | Viability Status | Operational Guidance |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | 0.0 – 20.0 | Safe / Low Saturation | Critical Viability Concern | Favorable for Planting / Quota Available |
| **Moderate** | 20.0 – 60.0 | Moderate Saturation | Moderate Viability | Proceed with Caution / Maintain Quotas |
| **High** | 60.0 – 80.0 | High Saturation | High Viability | High Risk / Restrict Expansion |
| **Critical** | 80.0 – 100.0 | Critical Oversupply Glut | Superior Agronomic Viability | Avoid Expansion / Diversify to Alternatives |

### Plain-Language Confidence Tiers
- **80.0 or above**: High confidence
- **60.0 to below 80.0**: Moderate confidence
- **Below 60.0**: Low confidence — verify with local officer

---

## 4. Empirical Data Sources & Provenance

| Data Category | Primary Source | Granularity | Scope & Records | Provenance Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Vegetable Prices** | HARTI Colombo & District Markets | Weekly (Retail & Wholesale) | 110,019 records (2014–2025) | Real Measured Observations |
| **Agro-Meteorology** | NASA POWER Agroclimatology | Daily ($0.5^\circ \times 0.5^\circ$) | 100,225 records (25 districts) | Real Measured Observations |
| **Cultivation Baseline** | HARTI / DOA District Records | Seasonal (Maha / Yala) | 2,875 seasonal quotas | Planning Baseline |
| **Yield Data** | Field Agronomic Surveys | Seasonal | Unavailable in current intake | Neutral Unavailable State |

> **Operational Refresh Note**: Cultivation commitment values for the prototype are derived from historical HARTI district records and refreshed per season.

---

## 5. Local Setup and Execution Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 (or Anaconda environment)
- Windows PowerShell or Bash terminal

### Step 1: Install Dependencies
From the repository root:
```bash
pip install -r prototype/member1_GeoSpatialDSS/requirements.txt
```

### Step 2: Run the Streamlit Dashboard
Execute from the repository root:
```bash
streamlit run prototype/member1_GeoSpatialDSS/app.py --server.port 8501
```
The application will open automatically in your browser at `http://localhost:8501`.

### Step 3: Run the Optional Standalone FastAPI Microservice
A decoupled FastAPI service is provided in `api_server.py` exposing REST endpoints for external cross-component integration:
```bash
uvicorn prototype.member1_GeoSpatialDSS.api_server:app --host 127.0.0.1 --port 8000 --reload
```
Available REST Endpoints:
- `GET http://127.0.0.1:8000/health`: Health status and project metadata.
- `GET http://127.0.0.1:8000/dri?district=Badulla&crop=Carrot`: Empirical 3-component DRI score and audit breakdown.
- `GET http://127.0.0.1:8000/fused-data?crop=Carrot&season=Maha&year=2024`: Unified multi-stream fused dataset.
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`

*Note: The Streamlit dashboard runs completely independently and does NOT require the FastAPI service to launch.*

### Step 4: Run the Automated Pytest Suite
Execute all 31 unit, regression, and data-reliability tests:
```bash
pytest prototype/member1_GeoSpatialDSS/tests/test_dashboard_modules.py -v
```

---

## 6. Project Boundaries & Collaboration

- **Collaboration**: Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI) collaboration | SLIIT | J26-DS-322
- **Isolated Target**: All modifications and assets are strictly contained within `prototype/member1_GeoSpatialDSS/`.
