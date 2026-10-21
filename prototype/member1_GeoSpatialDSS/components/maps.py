"""
Interactive Sri Lanka Geo-Spatial Visualizations using API-Key-Free Mapping.

Renders district-level maps focused strictly on Sri Lanka (5.8N to 9.9N, 79.5E to 82.0E)
using free OpenStreetMap tiles (requires zero Mapbox/Stadia API tokens, eliminating
'API KEY REQUIRED' watermarks and broken grey backgrounds).
Displays real-time Market Saturation Risk (MSRS, 0-100), Data Reliability Index (DRI, 0-100),
and Dynamic Crop Viability (DCVS, 0-100). Markers are sized by cultivation density where available.
"""

from typing import Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _build_scatter_map_figure(
    df_plot: pd.DataFrame,
    color_col: str,
    color_scale: list,
    title_text: str,
    colorbar_title: str,
    tick_vals: list,
    tick_labels: list,
    height: int = 520,
) -> go.Figure:
    """
    Internal helper to construct a version-agnostic, token-free Sri Lanka map.
    Automatically handles Plotly 5 (scatter_mapbox) and Plotly 6/7 (scatter_map).
    """
    # Build clean tooltip data dictionary based on available columns
    hover_dict: Dict[str, Any] = {
        "latitude": False,
        "longitude": False,
        "marker_size": False,
    }
    if "province" in df_plot.columns:
        hover_dict["province"] = True
    if "msrs" in df_plot.columns:
        hover_dict["msrs"] = ":.1f"
    if "dri_score" in df_plot.columns:
        hover_dict["dri_score"] = ":.1f"
    if "committed_hectares" in df_plot.columns:
        hover_dict["committed_hectares"] = ":.1f ha"
    if "target_hectares" in df_plot.columns:
        hover_dict["target_hectares"] = ":.1f ha"

    # Use Plotly 6/7 scatter_map if available; otherwise fallback to scatter_mapbox (Plotly <6)
    use_new_map = hasattr(px, "scatter_map")
    map_fn = getattr(px, "scatter_map", getattr(px, "scatter_mapbox", None))

    kwargs: Dict[str, Any] = {
        "data_frame": df_plot,
        "lat": "latitude",
        "lon": "longitude",
        "text": "district",
        "size": "marker_size",
        "color": color_col,
        "color_continuous_scale": color_scale,
        "range_color": [0.0, 100.0],
        "hover_name": "district",
        "hover_data": hover_dict,
        "zoom": 6.6,
        "center": dict(lat=7.8731, lon=80.7718),
        "title": title_text,
    }

    if use_new_map:
        kwargs["map_style"] = "carto-positron"
    else:
        kwargs["mapbox_style"] = "carto-positron"

    fig = map_fn(**kwargs)

    # Style marker labels and transparency
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=10, color="#FFFFFF", family="Inter, sans-serif"),
        marker=dict(opacity=0.90),
    )

    # Configure dark paper container and accessible colorbar
    fig.update_layout(
        height=height,
        paper_bgcolor="#0A140D",
        plot_bgcolor="#0A140D",
        title_font=dict(color="#FFFFFF", size=12),
        margin=dict(l=10, r=10, t=55, b=15),
        coloraxis_colorbar=dict(
            title=dict(text=colorbar_title, font=dict(color="#FFFFFF", size=10)),
            tickfont=dict(color="#C8D8CC", size=9),
            tickvals=tick_vals,
            ticktext=tick_labels,
            lenmode="fraction",
            len=0.75,
            thickness=13,
            bgcolor="rgba(10, 20, 13, 0.85)",
            bordercolor="#1E3D27",
            borderwidth=1,
        ),
    )

    return fig


def render_sri_lanka_map(
    df_district_metrics: pd.DataFrame,
    crop_name: str,
    season_label: str,
    layer: str = "msrs",
    height: int = 520,
) -> go.Figure:
    """
    Render Sri Lanka district map using token-free OpenStreetMap tiles.
    Eliminates 'API KEY REQUIRED' watermarks and grey backgrounds.
    District markers are sized by cultivation density and colored by MSRS or DRI.

    Args:
        df_district_metrics: Output from compute_district_msrs_table or joined metrics.
        crop_name: Selected vegetable crop.
        season_label: Season string (e.g. "Yala 2024").
        layer: 'msrs' for Market Saturation or 'dri' for Data Reliability.
        height: Pixel height.

    Returns:
        Plotly Figure object.
    """
    df_plot = df_district_metrics.copy()

    # Dynamic marker sizing based on cultivation density (committed hectares)
    if "committed_hectares" in df_plot.columns and df_plot["committed_hectares"].max() > 0:
        max_ha = float(df_plot["committed_hectares"].max())
        df_plot["marker_size"] = 12.0 + (df_plot["committed_hectares"] / max_ha) * 18.0
    else:
        df_plot["marker_size"] = 14.0

    # Ensure necessary metric columns are present for hovertooltips
    if "dri_score" not in df_plot.columns:
        df_plot["dri_score"] = 85.0
    if "target_hectares" not in df_plot.columns:
        df_plot["target_hectares"] = 100.0
    if "committed_hectares" not in df_plot.columns:
        df_plot["committed_hectares"] = 0.0

    if layer.lower() == "dri":
        return _build_scatter_map_figure(
            df_plot=df_plot,
            color_col="dri_score",
            color_scale=[
                [0.0, "#C53030"],   # Low Reliability (< 60)
                [0.60, "#D97706"],  # Moderate Reliability (60 - 80)
                [0.80, "#2E7D32"],  # High Reliability (80 - 100)
                [1.0, "#16A34A"],
            ],
            title_text=f"<b>Regional Data Reliability Index (DRI)</b><br><span style='font-size:10px;color:#A3B899'>{crop_name} ({season_label})</span>",
            colorbar_title="DRI Score",
            tick_vals=[0, 60, 80, 100],
            tick_labels=["0 (Low)", "60 (Moderate)", "80 (High)", "100 (Optimal)"],
            height=height,
        )
    else:
        # Default MSRS Layer (0 - 100 scale)
        return _build_scatter_map_figure(
            df_plot=df_plot,
            color_col="msrs",
            color_scale=[
                [0.0, "#2E7D32"],   # Low Saturation (0 - 20)
                [0.20, "#2E7D32"],
                [0.60, "#D97706"],  # Moderate (20 - 60)
                [0.80, "#EA580C"],  # High (60 - 80)
                [1.0, "#C53030"],   # Critical Glut (80 - 100)
            ],
            title_text=f"<b>Regional Market Saturation Risk (MSRS)</b><br><span style='font-size:10px;color:#A3B899'>{crop_name} ({season_label})</span>",
            colorbar_title="MSRS Risk",
            tick_vals=[0, 20, 60, 80, 100],
            tick_labels=["0 (Safe)", "20 (Low)", "60 (Moderate)", "80 (High)", "100 (Critical)"],
            height=height,
        )


def render_sri_lanka_saturation_map(
    df_district_msrs: pd.DataFrame,
    crop_name: str,
    season_label: str,
    height: int = 520,
) -> go.Figure:
    """Wrapper calling render_sri_lanka_map with MSRS layer for backward compatibility."""
    return render_sri_lanka_map(
        df_district_metrics=df_district_msrs,
        crop_name=crop_name,
        season_label=season_label,
        layer="msrs",
        height=height,
    )


def render_district_viability_map(
    df_district_dcvs: pd.DataFrame,
    crop_name: str,
    season_label: str,
    height: int = 520,
) -> go.Figure:
    """
    Render Sri Lanka map displaying Dynamic Crop Viability Score (DCVS) on token-free tiles.
    """
    df_plot = df_district_dcvs.copy()
    df_plot["marker_size"] = 15.0

    return _build_scatter_map_figure(
        df_plot=df_plot,
        color_col="dcvs",
        color_scale=[
            [0.0, "#C53030"],   # Low viability (< 45)
            [0.45, "#D97706"],  # Moderate (45 - 70)
            [0.70, "#2E7D32"],  # High (70 - 100)
            [1.0, "#16A34A"],
        ],
        title_text=f"<b>Dynamic Crop Viability Score (DCVS)</b> — {crop_name} ({season_label})",
        colorbar_title="DCVS Viability",
        tick_vals=[0, 45, 70, 100],
        tick_labels=["0 (Unviable)", "45 (Caution)", "70 (Favorable)", "100 (Prime)"],
        height=height,
    )
