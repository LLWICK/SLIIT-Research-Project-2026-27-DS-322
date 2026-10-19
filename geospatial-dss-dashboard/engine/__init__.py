"""
Decision Support System computational engines, risk scoring models, and analytical bridges.
"""

from engine.msrs_engine import (
    calculate_msrs,
    evaluate_saturation_risk,
    compute_district_msrs_table,
    compute_national_msrs_summary,
)
from engine.dcvs_engine import (
    calculate_dcvs,
    compute_district_dcvs_table,
)
from engine.dri_bridge import (
    get_dri_evaluation_summary,
    assess_district_data_readiness,
)
from engine.forecast_bridge import (
    ForecastAdapter,
    ForecastResult,
)
from engine.advisor import (
    FarmerAdvisoryEngine,
    Recommendation,
)

__all__ = [
    "calculate_msrs",
    "evaluate_saturation_risk",
    "compute_district_msrs_table",
    "compute_national_msrs_summary",
    "calculate_dcvs",
    "compute_district_dcvs_table",
    "get_dri_evaluation_summary",
    "assess_district_data_readiness",
    "ForecastAdapter",
    "ForecastResult",
    "FarmerAdvisoryEngine",
    "Recommendation",
]
