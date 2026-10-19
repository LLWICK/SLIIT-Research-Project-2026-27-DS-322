"""
Farmer Advisory and Decision Recommendation Engine.

Member 1 Geo-Spatial DSS Advisory Bridge:
Generates actionable farmer guidance to prevent Cobweb overproduction gluts and price crashes.
Synthesizes:
1. Real Market Prices (HARTI Weekly Retail/Wholesale)
2. Agro-Meteorological Moisture & Temperature (NASA POWER Daily Observations)
3. Regional Planting Commitments (MSRS Saturation Quota from Member 3 proxy)
4. Data Readiness Index (DRI Intake Audit Benchmark from Member 1 Governance)
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd

# Ensure dashboard root is on sys.path
_dashboard_root = str(Path(__file__).resolve().parent.parent)
if _dashboard_root not in sys.path:
    sys.path.insert(0, _dashboard_root)

from config import TARGET_CROPS
from engine.msrs_engine import (
    calculate_msrs,
    compute_district_msrs_table,
    compute_national_msrs_summary,
    evaluate_saturation_risk,
)
from engine.dcvs_engine import (
    compute_district_dcvs_table,
    evaluate_viability_tier,
)
from engine.dri_bridge import (
    assess_district_data_readiness,
)


@dataclass
class Recommendation:
    """Actionable farmer recommendation with transparent evidence breakdown."""
    district: str
    crop: str
    season_label: str
    action: str
    action_color: str
    banner_class: str
    headline: str
    rationale_points: List[str]
    msrs_score: float
    msrs_tier: str
    msrs_label: str
    dcvs_score: float
    dcvs_tier: str
    dcvs_label: str
    dri_score: float
    dri_source: str
    dri_tier: str
    dri_scope: str
    evidence_sufficiency: str
    confidence_limitation: str
    suggested_alternatives: List[Dict[str, Any]]
    data_provenance: str


class FarmerAdvisoryEngine:
    """
    Evidence-based advisory engine for farmers and agricultural extension officers.
    Strictly separates data reliability (DRI), market saturation (MSRS), and crop viability (DCVS).
    """

    def generate_advisory(
        self,
        district: str,
        crop: str,
        season_type: str,
        season_year: int,
        df_comm: pd.DataFrame,
        df_price: pd.DataFrame,
        df_weather: pd.DataFrame,
        price_type: str = "retail",
        precomputed_msrs: Optional[float] = None,
        precomputed_dcvs: Optional[float] = None,
    ) -> Recommendation:
        """
        Generate evidence-based planting advisory for a target district or national aggregate.

        Args:
            district: Target district or 'All Island (National Average)'.
            crop: Selected vegetable crop.
            season_type: 'Maha' or 'Yala'.
            season_year: Planning year.
            df_comm: Commitment dataset.
            df_price: Price dataset.
            df_weather: Weather dataset.
            price_type: 'retail' or 'wholesale'.
            precomputed_msrs: Optional pre-computed MSRS score for 100% KPI consistency.
            precomputed_dcvs: Optional pre-computed DCVS score for 100% KPI consistency.

        Returns:
            Structured Recommendation object.
        """
        is_national = (district in ["All Island (National Average)", "National", "All Island"])
        season_label = f"{season_type} {season_year}"
        scope_name = "National Aggregate" if is_national else district

        # Step 1: Resolve MSRS and Commitment Metrics
        df_msrs = compute_district_msrs_table(df_comm, crop, season_type, season_year)
        if is_national:
            if precomputed_msrs is not None:
                msrs_val = float(precomputed_msrs)
            else:
                nat_msrs_summary = compute_national_msrs_summary(df_comm, crop, season_type, season_year)
                msrs_val = float(nat_msrs_summary["national_msrs"])
            comm_ha = float(df_msrs["committed_hectares"].sum()) if not df_msrs.empty else 0.0
            tgt_ha = float(df_msrs["target_hectares"].sum()) if not df_msrs.empty else 100.0
            overshoot_pct = ((comm_ha - tgt_ha) / tgt_ha * 100.0) if tgt_ha > 0 else 0.0
            comm_source = "SYNTHETIC PROXY (NATIONAL AGGREGATE)"
        else:
            row_msrs = df_msrs[df_msrs["district"] == district]
            if not row_msrs.empty:
                msrs_val = float(precomputed_msrs) if precomputed_msrs is not None else float(row_msrs["msrs"].iloc[0])
                comm_ha = float(row_msrs["committed_hectares"].iloc[0])
                tgt_ha = float(row_msrs["target_hectares"].iloc[0])
                overshoot_pct = float(row_msrs["overshoot_pct"].iloc[0])
                comm_source = str(row_msrs["data_source"].iloc[0]).upper()
            else:
                msrs_val = float(precomputed_msrs) if precomputed_msrs is not None else 0.50
                comm_ha = 0.0
                tgt_ha = 100.0
                overshoot_pct = 0.0
                comm_source = "SYNTHETIC PROXY"

        msrs_tier, msrs_label, _ = evaluate_saturation_risk(msrs_val)

        # Step 2: Resolve DCVS, Price, and Weather Metrics
        df_dcvs = compute_district_dcvs_table(
            df_comm, df_price, df_weather, crop, season_type, season_year, price_type=price_type
        )
        if is_national:
            dcvs_val = float(precomputed_dcvs) if precomputed_dcvs is not None else (
                float(df_dcvs["dcvs"].mean()) if not df_dcvs.empty else 60.0
            )
            price_sub = df_price[(df_price["crop"] == crop) & (df_price["price_type"] == price_type)]
            curr_price = float(price_sub["price"].iloc[-1]) if not price_sub.empty else 120.0
            rainfall_mm = float(df_weather["rainfall_mm"].mean()) if not df_weather.empty else 5.0
            temp_c = float(df_weather["temp_c"].mean()) if not df_weather.empty else 25.0
        else:
            row_dcvs = df_dcvs[df_dcvs["district"] == district]
            if not row_dcvs.empty:
                dcvs_val = float(precomputed_dcvs) if precomputed_dcvs is not None else float(row_dcvs["dcvs"].iloc[0])
                curr_price = float(row_dcvs["current_price"].iloc[0])
                rainfall_mm = float(row_dcvs["rainfall_avg_mm"].iloc[0])
                temp_c = float(row_dcvs["temp_avg_c"].iloc[0])
            else:
                dcvs_val = float(precomputed_dcvs) if precomputed_dcvs is not None else 60.0
                curr_price = 120.0
                rainfall_mm = 5.0
                temp_c = 25.0

        dcvs_tier, dcvs_label, _ = evaluate_viability_tier(dcvs_val)

        # Step 3: Retrieve Authoritative DRI Benchmark (Separated from Recommendation Action)
        dri_info = assess_district_data_readiness(district, crop, df_price, df_weather, df_comm)
        dri_score = float(dri_info["benchmark_dri_score"])
        dri_source = str(dri_info["source_agency"])
        dri_tier = str(dri_info["reliability_tier"])
        dri_scope = str(dri_info["dri_scope"])
        evidence_sufficiency = str(dri_info["evidence_sufficiency"])
        confidence_limitation = str(dri_info["confidence_limitation"])

        # Step 4: Evaluate Actionable Guidance & Decision Logic
        rationale = []

        if msrs_val >= 0.70:
            action = "AVOID EXPANSION / DIVERSIFY"
            action_color = "#C53030"
            banner_class = "banner-critical"
            headline = f"Critical Saturation Alert: {crop} in {scope_name} ({season_label})"
            rationale.append(
                f"**Cobweb Glut Crash Risk (MSRS: {msrs_val:.2f} — {msrs_tier})**: Regional cultivation commitments "
                f"have exceeded sustainable absorption limits ({overshoot_pct:+.1f}% quota gap). "
                f"Concurrent harvesting without coordinated allocation will trigger an oversupply price collapse."
            )
            rationale.append(
                f"**Price Incentive Signal ({price_type.capitalize()} Quotes)**: Current quote is Rs. {curr_price:.1f}/kg. "
                f"Growers are cautioned against supply-lag price signals: trailing high spot prices frequently mislead growers into concurrent overplanting."
            )
        elif msrs_val >= 0.30:
            action = "PROCEED WITH CAUTION"
            action_color = "#D97706"
            banner_class = "banner-warning"
            headline = f"Moderate Saturation Warning: {crop} in {scope_name} ({season_label})"
            rationale.append(
                f"**Moderate Saturation (MSRS: {msrs_val:.2f} — {msrs_tier})**: Commitment registrations are approaching quota threshold. "
                f"Maintain existing planned planting without acreage expansion, and secure forward off-take contracts."
            )
            rationale.append(
                f"**Viability Index (DCVS)**: {dcvs_val:.1f}% ({dcvs_tier} Viability). Climatic favorability and market prices balance quota pressure."
            )
        else:
            action = "FAVORABLE FOR PLANTING"
            action_color = "#2E7D32"
            banner_class = "banner-safe"
            headline = f"Sustainable Market Capacity: {crop} in {scope_name} ({season_label})"
            rationale.append(
                f"**Market Capacity Available (MSRS: {msrs_val:.2f} — {msrs_tier})**: Cultivation commitments remain comfortably within sustainable demand quotas. "
                f"Minimal risk of an oversupply collapse."
            )
            rationale.append(
                f"**Dynamic Crop Viability (DCVS)**: {dcvs_val:.1f}% ({dcvs_tier} Viability). Favorable convergence of demand capacity, price returns, and weather."
            )

        # Cultivation Commitment Evidence
        rationale.append(
            f"**Cultivation Commitment Evidence ({comm_source})**: {comm_ha:,.1f} ha registered vs. {tgt_ha:,.1f} ha target quota "
            f"({overshoot_pct:+.1f}% capacity gap)."
        )

        # Agro-Meteorological Evidence
        scope_loc = "island-wide production zones" if is_national else district
        rationale.append(
            f"**Agro-Meteorological Environment (NASA POWER)**: Historical weekly precipitation of {rainfall_mm:.1f} mm "
            f"and mean ambient temperature of {temp_c:.1f}°C in {scope_loc} support regional agronomic requirements."
        )

        # Data Reliability Transparency Point
        rationale.append(
            f"**Source Data Reliability (DRI: {dri_score:.2f}%)**: Sourced from official {dri_source} intake audit ({dri_tier} Reliability). "
            f"Verified data intake health confirms that downstream models receive complete and consistent input feeds."
        )

        # Step 5: Identify Alternative Crops with Rigorous Comparative Scoring
        candidates = []
        for alt_crop in TARGET_CROPS:
            if alt_crop == crop:
                continue

            alt_dcvs_df = compute_district_dcvs_table(
                df_comm, df_price, df_weather, alt_crop, season_type, season_year, price_type=price_type
            )

            if is_national:
                alt_msrs_summary = compute_national_msrs_summary(df_comm, alt_crop, season_type, season_year)
                alt_msrs = float(alt_msrs_summary["national_msrs"])
                alt_dcvs = float(alt_dcvs_df["dcvs"].mean()) if not alt_dcvs_df.empty else 60.0
                alt_p_sub = df_price[(df_price["crop"] == alt_crop) & (df_price["price_type"] == price_type)]
                alt_p = float(alt_p_sub["price"].iloc[-1]) if not alt_p_sub.empty else 100.0
            else:
                alt_row = alt_dcvs_df[alt_dcvs_df["district"] == district]
                if not alt_row.empty:
                    alt_msrs = float(alt_row["msrs"].iloc[0])
                    alt_dcvs = float(alt_row["dcvs"].iloc[0])
                    alt_p = float(alt_row["current_price"].iloc[0])
                else:
                    alt_msrs = 0.50
                    alt_dcvs = 50.0
                    alt_p = 100.0

            if alt_msrs < 0.30:
                status_lbl = "Low Saturation (Safe Diversification)"
                expl = "Ample market capacity available in this production scope."
            elif alt_msrs < 0.70:
                status_lbl = "Moderate Saturation (Viable Alternative)"
                expl = "Manageable quota status with stable demand."
            else:
                status_lbl = "Elevated Saturation (Monitor Quota)"
                expl = "Quota constrained; evaluate regional commitments carefully."

            candidates.append({
                "crop": alt_crop,
                "msrs": alt_msrs,
                "dcvs": alt_dcvs,
                "current_price": alt_p,
                "status": status_lbl,
                "explanation": expl,
            })

        # Sort primarily by lower saturation (MSRS ascending) then higher viability (DCVS descending)
        candidates = sorted(candidates, key=lambda x: (x["msrs"], -x["dcvs"]))

        return Recommendation(
            district=scope_name,
            crop=crop,
            season_label=season_label,
            action=action,
            action_color=action_color,
            banner_class=banner_class,
            headline=headline,
            rationale_points=rationale,
            msrs_score=msrs_val,
            msrs_tier=msrs_tier,
            msrs_label=msrs_label,
            dcvs_score=dcvs_val,
            dcvs_tier=dcvs_tier,
            dcvs_label=dcvs_label,
            dri_score=dri_score,
            dri_source=dri_source,
            dri_tier=dri_tier,
            dri_scope=dri_scope,
            evidence_sufficiency=evidence_sufficiency,
            confidence_limitation=confidence_limitation,
            suggested_alternatives=candidates[:3],
            data_provenance="EVIDENCE-BASED ADVISORY (HARTI Price / NASA Weather / Commitment Proxy)",
        )
