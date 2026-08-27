"""Rank crops for a selected district using prototype DCVS / MSRS / DRI."""

from __future__ import annotations

import pandas as pd

from scores import dri_reasons


def _status(dcvs: float | None, msrs: float | None) -> str:
    if dcvs is None or (isinstance(dcvs, float) and pd.isna(dcvs)):
        return "Insufficient data"
    if msrs is not None and pd.notna(msrs) and float(msrs) >= 70:
        return "High Risk / Avoid"
    if float(dcvs) < 35:
        return "High Risk / Avoid"
    if float(dcvs) >= 60 and (msrs is None or pd.isna(msrs) or float(msrs) < 55):
        return "Recommended"
    return "Consider"


def _reason(row: pd.Series) -> str:
    bits: list[str] = []
    if pd.notna(row.get("crowding")):
        bits.append(f"crowding {float(row['crowding']):.2f}× crop median")
    if pd.notna(row.get("overshoot")):
        bits.append(f"commitment/target {float(row['overshoot']):.2f}")
    if pd.isna(row.get("season_mean_price")) and pd.isna(row.get("latest_price")):
        bits.append("no HARTI price for this district–crop")
    elif pd.notna(row.get("price_score")):
        bits.append(f"price score {float(row['price_score']):.0f}")
    if pd.isna(row.get("season_rainfall_mm")):
        bits.append("no seasonal weather summary")
    elif pd.notna(row.get("weather_score")):
        bits.append(f"weather typicality {float(row['weather_score']):.0f}")
    if pd.notna(row.get("dri")) and float(row["dri"]) < 45:
        bits.append("low prototype reliability")
    if not bits:
        bits.append("limited overlapping data for this combination")
    return "; ".join(bits)


def recommend_crops(district_scores: pd.DataFrame) -> pd.DataFrame:
    """Return ranked recommendation table for one district (all crops)."""
    if district_scores.empty:
        return pd.DataFrame(
            columns=["Crop", "DCVS", "MSRS", "DRI", "Status", "Reason"]
        )

    ranked = district_scores.sort_values(
        ["dcvs", "dri"], ascending=[False, False], na_position="last"
    ).copy()

    rows = []
    for _, row in ranked.iterrows():
        dcvs = row.get("dcvs")
        msrs = row.get("msrs")
        dri = row.get("dri")
        rows.append(
            {
                "Crop": row.get("crop_name"),
                "DCVS": None if pd.isna(dcvs) else round(float(dcvs), 1),
                "MSRS": None if pd.isna(msrs) else round(float(msrs), 1),
                "DRI": None if pd.isna(dri) else round(float(dri), 1),
                "Status": _status(
                    None if pd.isna(dcvs) else float(dcvs),
                    None if pd.isna(msrs) else float(msrs),
                ),
                "Reason": _reason(row),
                "_dri_reasons": dri_reasons(row),
            }
        )
    return pd.DataFrame(rows)
