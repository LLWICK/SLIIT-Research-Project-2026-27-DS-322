"""
Decision Support System computational engines, risk scoring models, and analytical bridges.
"""

from .msrs_engine import (
    calculate_msrs,
    evaluate_saturation_risk,
    compute_district_msrs_table,
    compute_national_msrs_summary,
)
from .dcvs_engine import (
    calculate_dcvs,
    compute_district_dcvs_table,
)
from .dri_bridge import (
    get_dri_evaluation_summary,
    assess_district_data_readiness,
)
from .forecast_bridge import (
    ForecastAdapter,
    ForecastResult,
)
from .advisor import (
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
