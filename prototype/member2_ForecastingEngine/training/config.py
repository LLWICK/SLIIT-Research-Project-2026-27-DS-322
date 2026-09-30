"""Paths and the locked progress-presentation split."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HTFE_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = HTFE_DIR.parent
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))

from data_pipeline.build.scope import (  # noqa: E402
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
from data_pipeline.config import processed_dir  # noqa: E402

ARTIFACTS = HTFE_DIR / "artifacts"
STUDIO_DATA = HTFE_DIR / "htfe-studio" / "data"
PANEL_PATH = processed_dir() / "latest" / "analysis_ready_panel.parquet"
FEATURE_PATH = processed_dir() / "latest" / "feature_frame.parquet"

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
