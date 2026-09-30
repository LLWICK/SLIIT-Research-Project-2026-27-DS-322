"""Locked progress-presentation scope. Defaults from the build playbook."""
from __future__ import annotations

CROPS = ("carrot", "leeks", "tomato")
EVAL_MARKETS = ("Dambulla", "Colombo", "Meegoda", "Nuwara Eliya")
TRAIN_POOL_MARKETS = ("Kandy", "Keppetipola", "Thambuttegama")
ALL_MARKETS = EVAL_MARKETS + TRAIN_POOL_MARKETS
HORIZONS = (1, 2, 4, 8, 12)

TRAIN_END = "2022-12-31"
CAL_START = "2023-01-01"
CAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"
SHOCK_START = "2024-01-01"
SHOCK_END = "2024-03-31"

WEATHER_KEEP = (
    "precip_mm",
    "tmean_c",
    "tmax_c",
    "tmin_c",
    "et0_mm",
    "shortwave_mj_m2",
    "rh_mean_pct",
    "soil_moist_0_7cm_m3m3",
    "wet_days_1mm",
    "heavy_rain_days_50mm",
    "complete",
)
