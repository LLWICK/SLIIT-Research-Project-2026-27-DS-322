"""
FastAPI Microservice for Geo-Spatial Decision Support System.

Exposes REST endpoints for:
- GET /health: Health check and service metadata.
- GET /dri: Empirical Data Reliability Index assessment (district and national).
- GET /fused-data: Fused multi-source market saturation, crop viability, and price dataset.

Run command:
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path
from typing import Optional

_dashboard_root = str(Path(__file__).resolve().parent)
if _dashboard_root not in sys.path:
    sys.path.insert(0, _dashboard_root)

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data_loader import (
    load_price_data,
    load_commitment_data,
    load_weather_data,
    load_dri_audit_ledger,
)
from engine.dri_bridge import (
    assess_district_data_readiness,
    compute_district_dri_table,
)
from engine.msrs_engine import (
    compute_district_msrs_table,
    compute_national_msrs_summary,
)
from engine.dcvs_engine import (
    compute_district_dcvs_table,
)

app = FastAPI(
    title="SLIIT Geo-Spatial DSS API (J26-DS-322)",
    description="REST Service providing empirical DRI scores, market saturation metrics, and fused datasets.",
    version="1.0.0",
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Service health and diagnostic status."""
    return {
        "status": "healthy",
        "service": "SLIIT-GeoSpatialDSS-API",
        "version": "1.0.0",
        "project": "J26-DS-322",
        "institution": "Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI) collaboration | SLIIT",
    }


@app.get("/dri")
def get_dri_assessment(
    district: str = Query("All Island (National Average)", description="Target district or national aggregate"),
    crop: str = Query("Carrot", description="Target research crop"),
    season_year: int = Query(2024, description="Planning season year"),
):
    """
    Retrieve empirical 3-component DRI score and category audits:
    - Completeness (40%)
    - Recency (30%)
    - Consistency (30%)
    """
    try:
        df_price = load_price_data()
        df_weather = load_weather_data()
        df_comm = load_commitment_data()

        result = assess_district_data_readiness(
            district=district,
            crop=crop,
            df_price=df_price,
            df_weather=df_weather,
            df_comm=df_comm,
            season_year=season_year,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fused-data")
def get_fused_data(
    crop: str = Query("Carrot", description="Target vegetable crop"),
    season_type: str = Query("Maha", description="Season type: Maha or Yala"),
    season_year: int = Query(2024, description="Planning year"),
    price_type: str = Query("retail", description="Quotation level: retail or wholesale"),
):
    """
    Retrieve fused analytical slice including MSRS, DCVS, and DRI across all 25 districts.
    """
    try:
        df_price = load_price_data()
        df_weather = load_weather_data()
        df_comm = load_commitment_data()

        df_msrs = compute_district_msrs_table(df_comm, crop, season_type, season_year)
        df_dcvs = compute_district_dcvs_table(df_comm, df_price, df_weather, crop, season_type, season_year, price_type=price_type)
        df_dri = compute_district_dri_table(df_price, df_weather, df_comm, crop, season_year=season_year)

        fused = df_msrs[["district", "province", "committed_hectares", "target_hectares", "msrs", "risk_label"]].merge(
            df_dcvs[["district", "dcvs", "viability_tier", "current_price"]],
            on="district",
            how="left",
        ).merge(
            df_dri[["district", "dri_score", "completeness", "recency", "consistency", "confidence_label"]],
            on="district",
            how="left",
        )

        return {
            "crop": crop,
            "season_type": season_type,
            "season_year": season_year,
            "price_type": price_type,
            "districts_count": len(fused),
            "data": fused.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
