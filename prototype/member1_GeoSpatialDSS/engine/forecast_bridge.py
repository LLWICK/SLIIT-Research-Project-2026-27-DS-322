"""
Member 2 Price Forecasting Adapter & Contract Bridge.

Provides a decoupled interface for connecting Member 2's price forecasting models
(ARIMA, SARIMAX, LSTM, Prophet) into Member 1's Geo-Spatial DSS.

CRITICAL RESEARCH INTEGRITY RULE:
Never invent data, forecasts, or confidence bands. Only display forecasts and
confidence corridors when a verified model file or validated REST endpoint response exists.
When no live service is connected, return a professional empty state:
'Forecast service not connected' and show verified historical price observations only.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Ensure dashboard root is on sys.path
_dashboard_root = str(Path(__file__).resolve().parent.parent)
if _dashboard_root not in sys.path:
    sys.path.insert(0, _dashboard_root)

from data_loader import get_project_root


@dataclass
class ForecastResult:
    """Standardized forecast container matching Member 2's output contract."""
    is_live: bool
    status_label: str
    provenance_tag: str
    warning_banner: Optional[str]
    forecast_df: pd.DataFrame
    historical_df: pd.DataFrame
    model_name: str
    confidence_level: float
    narrative_summary: str


class ForecastAdapter:
    """
    Decoupled adapter bridging Member 2's forecasting models to the Geo-Spatial DSS.
    Strictly reports genuine forecast availability without synthetic placeholders.
    """

    def __init__(self):
        self.root = get_project_root()
        self.forecast_dir = self.root / "data" / "processed" / "forecasts"
        self.m2_output_packet = (
            self.root / "prototype" / "member2_ForecastingEngine" / "htfe-studio" / "outputs" / "forecast_packet.json"
        )

    def check_live_model_available(self, crop: str) -> bool:
        """Check if real empirical forecast files exist on disk for the crop."""
        if self.forecast_dir.exists():
            crop_file = self.forecast_dir / f"price_forecast_{crop.lower().replace(' ', '_')}.csv"
            if crop_file.exists():
                return True
        if self.m2_output_packet.exists():
            return True
        return False

    def get_forecast(
        self,
        crop: str,
        district: str,
        historical_price_df: pd.DataFrame,
        price_type: str = "retail",
        enable_demo_simulation: bool = False,
        weeks_ahead: int = 12,
    ) -> ForecastResult:
        """
        Retrieve price forecast for the target crop and district.

        Follows documented contract:
        - If live model file exists and is validated, loads real projections and confidence bounds.
        - If no working service or valid response exists, returns honest empty state:
          'Forecast service not connected'.
        - Never fabricates numbers, random walks, or fake confidence bands.

        Args:
            crop: Selected crop name.
            district: Selected administrative district or National Aggregate.
            historical_price_df: Available historical weekly prices.
            price_type: 'retail' or 'wholesale' quotation level.
            enable_demo_simulation: Kept for contract compatibility; does NOT generate fake numbers.
            weeks_ahead: Prediction horizon in weeks.

        Returns:
            ForecastResult containing projection data, uncertainty bounds, and provenance tags.
        """
        # Step 0: Extract recent historical price baseline matching selected filters
        sub_p = historical_price_df[historical_price_df["crop"] == crop].copy()
        if "price_type" in sub_p.columns and price_type:
            sub_p = sub_p[sub_p["price_type"] == price_type]
        if district and district not in ["All Island (National Average)", "National", "All Island"]:
            dist_sub = sub_p[sub_p["district"] == district]
            if not dist_sub.empty:
                sub_p = dist_sub
            else:
                sub_p = pd.DataFrame(columns=historical_price_df.columns)

        if sub_p.empty:
            hist_context = pd.DataFrame(columns=["price_date", "price"])
        else:
            sub_p["price_date"] = pd.to_datetime(sub_p["price_date"])
            # Aggregate by price_date to obtain genuine single weekly observation values
            grouped_p = sub_p.groupby("price_date")["price"].mean().reset_index()
            grouped_p = grouped_p.sort_values(by="price_date")
            # Keep up to the last 52 weekly dates (1 full seasonal year)
            hist_context = grouped_p.tail(52).copy().reset_index(drop=True)

        # Step 1: Check for real model file from Member 2
        if self.check_live_model_available(crop):
            crop_file = self.forecast_dir / f"price_forecast_{crop.lower().replace(' ', '_')}.csv"
            if crop_file.exists():
                try:
                    df_live = pd.read_csv(crop_file)
                    req_cols = {"forecast_date", "predicted_price"}
                    if req_cols.issubset(set(df_live.columns)):
                        return ForecastResult(
                            is_live=True,
                            status_label="LIVE MODEL OUTPUT (Price Forecast Model)",
                            provenance_tag="REAL MODEL PREDICTION",
                            warning_banner=None,
                            forecast_df=df_live,
                            historical_df=hist_context,
                            model_name=str(df_live.get("model_name", ["Ensemble"])[0]),
                            confidence_level=0.95,
                            narrative_summary=f"Live econometric model output integrated for {crop} in {district}.",
                        )
                except Exception:
                    pass

        # Step 2: Honest empty state when service is not connected
        empty_df = pd.DataFrame(columns=[
            "forecast_date", "predicted_price", "lower_bound_95", "upper_bound_95", "data_tag"
        ])
        return ForecastResult(
            is_live=False,
            status_label="Forecast service not connected",
            provenance_tag="HISTORICAL BASELINE ONLY",
            warning_banner=None,
            forecast_df=empty_df,
            historical_df=hist_context,
            model_name="Econometric Price Forecasting Model (Pending Connection)",
            confidence_level=0.0,
            narrative_summary=(
                f"Observing verified historical price baseline for {crop} ({district}). "
                "Predictive forecast REST endpoint is currently disconnected. "
                "Showing verified historical price observations only."
            ),
        )
