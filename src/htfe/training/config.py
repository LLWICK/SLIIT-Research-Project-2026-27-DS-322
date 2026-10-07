"""Training paths and the locked chronological split."""
from __future__ import annotations

import pandas as pd

from htfe.config import ARTIFACTS, STUDIO_DATA, built_file
from htfe.data.build.scope import (
    ALL_MARKETS,
    CAL_END,
    CAL_START,
    CROPS,
    EVAL_MARKETS,
    HORIZONS,
    SHOCK_END,
    SHOCK_START,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_POOL_MARKETS,
)

PANEL_PATH = built_file("analysis_ready_panel.parquet")
FEATURE_PATH = built_file("feature_frame.parquet")

WEATHER_LEVELS = (
    "weather_precip_mm",
    "weather_tmean_c",
    "weather_tmax_c",
    "weather_tmin_c",
    "weather_et0_mm",
    "weather_shortwave_mj_m2",
    "weather_rh_mean_pct",
    "weather_soil_moist_0_7cm_m3m3",
    "weather_wet_days_1mm",
    "weather_heavy_rain_days_50mm",
)
WEATHER_LAGS = (1, 2, 4, 8)
PRICE_LAGS = (1, 2, 4, 8, 12, 52)


def assign_split(week_start: pd.Series) -> pd.DataFrame:
    stamps = pd.to_datetime(week_start)
    split = pd.Series("other", index=stamps.index)
    split = split.mask(stamps <= TRAIN_END, "train")
    split = split.mask((stamps >= CAL_START) & (stamps <= CAL_END), "calibrate")
    split = split.mask((stamps >= TEST_START) & (stamps <= TEST_END), "test")
    shock = (stamps >= SHOCK_START) & (stamps <= SHOCK_END)
    return pd.DataFrame({"split": split, "is_shock": shock})
