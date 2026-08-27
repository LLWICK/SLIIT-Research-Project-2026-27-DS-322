"""Approximate Sri Lankan district centroids for the prototype map.

PROTOTYPE ASSUMPTION: the project does not include official district
GeoJSON boundaries. These coordinates are approximate administrative
centroids used only for visualization. They are not cadastral or
survey-grade locations.
"""

from __future__ import annotations

from typing import Optional

import folium
from branca.colormap import LinearColormap


# Approximate district centroids (latitude, longitude).
DISTRICT_CENTROIDS: dict[str, tuple[float, float]] = {
    "Ampara": (7.2975, 81.6820),
    "Anuradhapura": (8.3114, 80.4037),
    "Badulla": (6.9934, 81.0550),
    "Batticaloa": (7.7310, 81.6747),
    "Colombo": (6.9271, 79.8612),
    "Galle": (6.0535, 80.2210),
    "Gampaha": (7.0912, 80.0081),
    "Hambantota": (6.1241, 81.1185),
    "Jaffna": (9.6615, 80.0255),
    "Kalutara": (6.5854, 79.9607),
    "Kandy": (7.2906, 80.6337),
    "Kegalle": (7.2513, 80.3464),
    "Kilinochchi": (9.3803, 80.3770),
    "Kurunegala": (7.4863, 80.3647),
    "Mannar": (8.9810, 79.9044),
    "Matale": (7.4675, 80.6234),
    "Matara": (5.9549, 80.5550),
    "Monaragala": (6.8726, 81.3509),
    "Mullaitivu": (9.2671, 80.8142),
    "Nuwara Eliya": (6.9497, 80.7891),
    "Polonnaruwa": (7.9403, 81.0188),
    "Puttalam": (8.0362, 79.8398),
    "Ratnapura": (6.7056, 80.3847),
    "Trincomalee": (8.5874, 81.2152),
    "Vavuniya": (8.7542, 80.4982),
}

SRI_LANKA_CENTER = (7.87, 80.77)


def get_centroid(district: str) -> Optional[tuple[float, float]]:
    return DISTRICT_CENTROIDS.get(district)


def _metric_colormap(metric_key: str, vmin: float, vmax: float) -> LinearColormap:
    """Colour scales: risk metrics are red-high; viability metrics are green-high."""
    if vmax <= vmin:
        vmax = vmin + 1.0
    if metric_key in {"msrs", "cultivation_density"}:
        colors = ["#FEF3C7", "#F59E0B", "#DC2626"]
        caption = "Higher = more crowding / saturation risk"
    elif metric_key in {"dcvs", "dri"}:
        colors = ["#FECACA", "#FDE68A", "#059669"]
        caption = "Higher = more viable / more reliable"
    else:
        colors = ["#DBEAFE", "#2563EB"]
        caption = metric_key
    cmap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=caption)
    return cmap


def _fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:  # NaN
            return "n/a"
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def build_district_map(
    score_df,
    metric_key: str,
    metric_label: str,
    selected_district: str,
) -> folium.Map:
    """Circle-marker map at approximate district centroids."""
    fmap = folium.Map(
        location=list(SRI_LANKA_CENTER),
        zoom_start=7,
        tiles="CartoDB positron",
        control_scale=True,
    )

    values = score_df[metric_key].dropna()
    vmin = float(values.min()) if len(values) else 0.0
    vmax = float(values.max()) if len(values) else 100.0
    cmap = _metric_colormap(metric_key, vmin, vmax)
    cmap.add_to(fmap)

    ha_values = score_df["committed_hectares"].fillna(0)
    ha_max = float(ha_values.max()) if len(ha_values) else 1.0
    ha_max = ha_max if ha_max > 0 else 1.0

    for _, row in score_df.iterrows():
        district = row["district"]
        centroid = get_centroid(district)
        if centroid is None:
            continue

        metric_val = row.get(metric_key)
        color = "#9CA3AF" if metric_val is None or metric_val != metric_val else cmap(float(metric_val))
        radius = 8 + 14 * (float(row.get("committed_hectares") or 0) / ha_max)
        is_selected = district == selected_district

        popup_html = f"""
        <div style="font-family:Segoe UI,sans-serif;min-width:210px">
          <div style="font-weight:700;font-size:14px;margin-bottom:6px">{district}</div>
          <table style="font-size:12px;border-collapse:collapse">
            <tr><td>Committed</td><td style="padding-left:8px"><b>{_fmt(row.get('committed_hectares'), 1)} ha</b></td></tr>
            <tr><td>Target</td><td style="padding-left:8px">{_fmt(row.get('target_hectares'), 1)} ha</td></tr>
            <tr><td>Prototype MSRS</td><td style="padding-left:8px">{_fmt(row.get('msrs'), 1)}</td></tr>
            <tr><td>Prototype DCVS</td><td style="padding-left:8px">{_fmt(row.get('dcvs'), 1)}</td></tr>
            <tr><td>Prototype DRI</td><td style="padding-left:8px">{_fmt(row.get('dri'), 1)}</td></tr>
            <tr><td>Latest price</td><td style="padding-left:8px">{_fmt(row.get('latest_price'), 2)}</td></tr>
          </table>
          <div style="margin-top:6px;color:#6B7280;font-size:11px">Centroid is approximate</div>
        </div>
        """

        folium.CircleMarker(
            location=list(centroid),
            radius=radius,
            color="#111827" if is_selected else color,
            weight=3 if is_selected else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.88 if is_selected else 0.72,
            tooltip=district,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(fmap)

    return fmap
