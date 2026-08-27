"""Prototype MSRS, DCVS and DRI calculations for the Geo-Spatial DSS demo.

These are transparent placeholder scores for the research proposal
presentation. They are not the final validated research models.

TODO: get_msrs(), get_dcvs() and get_forecast() are integration hooks.
Later they can call the other members' APIs instead of the local formulas.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# Prototype source reliability weights. Not final research weights.
SOURCE_RELIABILITY = {
    "synthetic_proxy": 0.35,
    "harti": 0.90,
    "nasa_power": 0.85,
}

DRI_WEIGHTS = {
    "commitment": 0.25,
    "price": 0.45,
    "weather": 0.30,
}


def _clip01to100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _safe_div(num: float, den: float) -> Optional[float]:
    if den is None or den == 0 or (isinstance(den, float) and not np.isfinite(den)):
        return None
    if num is None or (isinstance(num, float) and not np.isfinite(num)):
        return None
    return float(num) / float(den)


def _norm_around_one(ratio: Optional[float]) -> Optional[float]:
    """Map ratio 0.5 → 0, 1.0 → ~33, 2.0 → 100."""
    if ratio is None or not np.isfinite(ratio):
        return None
    return _clip01to100((ratio - 0.5) / 1.5 * 100.0)


def compute_msrs_table(commitment_slice: pd.DataFrame) -> pd.DataFrame:
    """Prototype MSRS from one season/year slice of the commitment file.

    crowding  = committed_hectares / median committed hectares of the same crop
    overshoot = committed_hectares / target_hectares
    MSRS      = 0.5 * crowding_norm + 0.5 * overshoot_norm, clipped to 0–100
    """
    df = commitment_slice.copy()
    if df.empty:
        return df

    crop_median = df.groupby("crop_name")["committed_hectares"].transform("median")
    df["district_crop_median"] = crop_median
    df["crowding"] = df["committed_hectares"] / df["district_crop_median"].replace(0, np.nan)
    df["overshoot"] = df["committed_hectares"] / df["target_hectares"].replace(0, np.nan)
    df["crowding_norm"] = df["crowding"].map(_norm_around_one)
    df["overshoot_norm"] = df["overshoot"].map(_norm_around_one)

    def _msrs(row) -> Optional[float]:
        parts = [p for p in (row["crowding_norm"], row["overshoot_norm"]) if pd.notna(p)]
        if not parts:
            return np.nan
        return _clip01to100(float(np.mean(parts)))

    df["msrs"] = df.apply(_msrs, axis=1)
    return df


def _price_score(mean_price: Optional[float], baseline: Optional[float]) -> Optional[float]:
    ratio = _safe_div(mean_price, baseline)
    return _norm_around_one(ratio)


def _weather_score(
    rainfall: Optional[float],
    temp: Optional[float],
    rain_median: Optional[float],
    rain_iqr: Optional[float],
    temp_median: Optional[float],
    temp_iqr: Optional[float],
) -> Optional[float]:
    """How typical this season is vs the district's own historical climate.

    This is NOT crop agronomic suitability. It only uses NASA POWER summaries.
    """
    if rainfall is None or temp is None or rain_median is None or temp_median is None:
        return None
    rain_scale = rain_iqr if rain_iqr and rain_iqr > 0 else max(abs(rain_median) * 0.25, 1.0)
    temp_scale = temp_iqr if temp_iqr and temp_iqr > 0 else 1.0
    rain_dev = abs(rainfall - rain_median) / rain_scale
    temp_dev = abs(temp - temp_median) / temp_scale
    return _clip01to100(100.0 - 35.0 * rain_dev - 35.0 * temp_dev)


def compute_score_table(
    commitment: pd.DataFrame,
    price_seasonal: pd.DataFrame,
    price_latest: pd.DataFrame,
    price_counts: pd.DataFrame,
    weather_seasonal: pd.DataFrame,
    year: int,
    season: str,
    price_type: str,
    crop: Optional[str] = None,
) -> pd.DataFrame:
    """One row per district–crop for the selected season/year."""
    slice_df = commitment[
        (commitment["season_year"] == year) & (commitment["season_type"] == season)
    ].copy()
    if crop:
        slice_df = slice_df[slice_df["crop_name"] == crop]
    if slice_df.empty:
        return slice_df

    scored = compute_msrs_table(slice_df)

    ps = price_seasonal[price_seasonal["price_type"] == price_type].copy()
    current_price = ps[(ps["season_year"] == year) & (ps["season_type"] == season)]
    hist_price = ps[ps["season_type"] == season]
    baseline = (
        hist_price.groupby(["district", "crop"])["median_price"]
        .median()
        .rename("price_baseline")
        .reset_index()
    )

    latest = price_latest[price_latest["price_type"] == price_type][
        ["district", "crop", "latest_price", "price_date", "granularity", "price_source"]
    ]
    counts = price_counts[price_counts["price_type"] == price_type][
        ["district", "crop", "n_obs", "completeness"]
    ].rename(columns={"n_obs": "price_n_obs", "completeness": "price_completeness"})

    current_price = current_price.rename(
        columns={
            "mean_price": "season_mean_price",
            "n_obs": "season_price_n_obs",
            "crop": "crop_name",
        }
    )
    latest = latest.rename(columns={"crop": "crop_name"})
    counts = counts.rename(columns={"crop": "crop_name"})
    baseline = baseline.rename(columns={"crop": "crop_name"})

    merge_keys = ["district", "crop_name"]
    scored = scored.merge(
        current_price[merge_keys + ["season_mean_price", "season_price_n_obs"]],
        on=merge_keys,
        how="left",
    )
    scored = scored.merge(latest, on=merge_keys, how="left")
    scored = scored.merge(counts, on=merge_keys, how="left")
    scored = scored.merge(baseline, on=merge_keys, how="left")

    ws = weather_seasonal.copy()
    current_wx = ws[(ws["season_year"] == year) & (ws["season_type"] == season)][
        ["district", "rainfall_mm", "temp_c", "n_days", "completeness", "source"]
    ].rename(
        columns={
            "rainfall_mm": "season_rainfall_mm",
            "temp_c": "season_temp_c",
            "n_days": "weather_n_days",
            "completeness": "weather_completeness",
            "source": "weather_source",
        }
    )
    hist_wx = ws[ws["season_type"] == season]
    wx_stats = (
        hist_wx.groupby("district")
        .agg(
            rain_median=("rainfall_mm", "median"),
            rain_q25=("rainfall_mm", lambda s: s.quantile(0.25)),
            rain_q75=("rainfall_mm", lambda s: s.quantile(0.75)),
            temp_median=("temp_c", "median"),
            temp_q25=("temp_c", lambda s: s.quantile(0.25)),
            temp_q75=("temp_c", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    wx_stats["rain_iqr"] = wx_stats["rain_q75"] - wx_stats["rain_q25"]
    wx_stats["temp_iqr"] = wx_stats["temp_q75"] - wx_stats["temp_q25"]

    scored = scored.merge(current_wx, on="district", how="left")
    scored = scored.merge(wx_stats, on="district", how="left")

    scored["price_score"] = scored.apply(
        lambda r: _price_score(r.get("season_mean_price"), r.get("price_baseline")),
        axis=1,
    )
    scored["weather_score"] = scored.apply(
        lambda r: _weather_score(
            r.get("season_rainfall_mm"),
            r.get("season_temp_c"),
            r.get("rain_median"),
            r.get("rain_iqr"),
            r.get("temp_median"),
            r.get("temp_iqr"),
        ),
        axis=1,
    )
    scored["anti_crowding"] = scored["msrs"].map(
        lambda x: _clip01to100(100.0 - float(x)) if pd.notna(x) else np.nan
    )
    scored["dcvs"] = scored.apply(_combine_dcvs, axis=1)
    scored["dri"] = scored.apply(_combine_dri, axis=1)
    scored["cultivation_density"] = scored["committed_hectares"]
    return scored


def _combine_dcvs(row: pd.Series) -> float:
    price_s = row.get("price_score")
    weather_s = row.get("weather_score")
    anti = row.get("anti_crowding")
    parts: list[tuple[float, float]] = []
    if pd.notna(price_s):
        parts.append((0.45, float(price_s)))
    if pd.notna(weather_s):
        parts.append((0.35, float(weather_s)))
    if pd.notna(anti):
        parts.append((0.20, float(anti)))
    if not parts:
        return np.nan
    weight_sum = sum(w for w, _ in parts)
    return _clip01to100(sum(w * v for w, v in parts) / weight_sum)


def _combine_dri(row: pd.Series) -> float:
    commit_source = str(row.get("source") or "")
    price_source = str(row.get("price_source") or "")
    weather_source = str(row.get("weather_source") or "")

    commit_rel = SOURCE_RELIABILITY.get(commit_source, 0.20)
    commit_comp = 1.0 if pd.notna(row.get("committed_hectares")) else 0.0

    has_price = pd.notna(row.get("latest_price")) or pd.notna(row.get("season_mean_price"))
    price_rel = SOURCE_RELIABILITY.get(price_source, 0.0) if has_price else 0.0
    price_comp = float(row["price_completeness"]) if pd.notna(row.get("price_completeness")) else 0.0
    if has_price and price_comp == 0:
        price_comp = 0.4

    has_weather = pd.notna(row.get("season_rainfall_mm"))
    weather_rel = SOURCE_RELIABILITY.get(weather_source, 0.0) if has_weather else 0.0
    weather_comp = (
        float(row["weather_completeness"]) if pd.notna(row.get("weather_completeness")) else 0.0
    )

    dri = 100.0 * (
        DRI_WEIGHTS["commitment"] * commit_rel * commit_comp
        + DRI_WEIGHTS["price"] * price_rel * price_comp
        + DRI_WEIGHTS["weather"] * weather_rel * weather_comp
    )
    return _clip01to100(dri)


def dri_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    commit_source = str(row.get("source") or "unknown")
    reasons.append(
        f"Commitment source is `{commit_source}` "
        f"(prototype reliability {SOURCE_RELIABILITY.get(commit_source, 0.20):.2f})."
    )
    if pd.isna(row.get("latest_price")) and pd.isna(row.get("season_mean_price")):
        reasons.append("No HARTI price rows for this district / crop / price type.")
    else:
        n_obs = row.get("price_n_obs")
        n_txt = f"{int(n_obs)} observations" if pd.notna(n_obs) else "unknown count"
        src = row.get("price_source") or "harti"
        reasons.append(
            f"Price source is `{src}` "
            f"(prototype reliability {SOURCE_RELIABILITY.get(str(src), 0.90):.2f}); {n_txt}."
        )
    if pd.isna(row.get("season_rainfall_mm")):
        reasons.append("No NASA POWER weather summary for this district / season / year.")
    else:
        src = row.get("weather_source") or "nasa_power"
        days = row.get("weather_n_days")
        day_txt = f"{int(days)} daily rows in season" if pd.notna(days) else "season coverage unknown"
        reasons.append(
            f"Weather source is `{src}` "
            f"(prototype reliability {SOURCE_RELIABILITY.get(str(src), 0.85):.2f}); {day_txt}."
        )
    if pd.isna(row.get("price_score")):
        reasons.append("Prototype DCVS reweighted because seasonal price was unavailable.")
    if pd.isna(row.get("weather_score")):
        reasons.append("Prototype DCVS reweighted because seasonal weather was unavailable.")
    reasons.append("These weights are a presentation placeholder, not a validated reliability model.")
    return reasons


def get_msrs(
    district: str,
    crop: str,
    season: str,
    year: int,
    score_table: Optional[pd.DataFrame] = None,
    **_: Any,
) -> Optional[float]:
    """Return prototype MSRS for one district–crop.

    TODO: Replace the local lookup with Member 3 crowding / MSRS API, e.g.
          GET /api/msrs?district=&crop=&season=&year=
    """
    if score_table is None or score_table.empty:
        return None
    match = score_table[
        (score_table["district"] == district) & (score_table["crop_name"] == crop)
    ]
    if match.empty or pd.isna(match.iloc[0].get("msrs")):
        return None
    return float(match.iloc[0]["msrs"])


def get_dcvs(
    district: str,
    crop: str,
    season: str,
    year: int,
    score_table: Optional[pd.DataFrame] = None,
    **_: Any,
) -> Optional[float]:
    """Return prototype DCVS for one district–crop.

    TODO: Replace the local lookup with the final DCVS research component API.
    """
    if score_table is None or score_table.empty:
        return None
    match = score_table[
        (score_table["district"] == district) & (score_table["crop_name"] == crop)
    ]
    if match.empty or pd.isna(match.iloc[0].get("dcvs")):
        return None
    return float(match.iloc[0]["dcvs"])


def get_forecast(
    district: str,
    crop: str,
    season: str,
    year: int,
    **_: Any,
) -> dict[str, Any]:
    """Forecasting is owned by another team member.

    TODO: Connect to the forecasting API, e.g.
          GET /api/forecast?district=&crop=&season=&year=
    Do not invent a forecast series in this Geo-Spatial DSS prototype.
    """
    return {
        "available": False,
        "district": district,
        "crop": crop,
        "season": season,
        "year": year,
        "values": None,
        "message": (
            "Price / production forecasts are not calculated in this Geo-Spatial DSS "
            "prototype. They will be supplied later by the forecasting component."
        ),
    }
