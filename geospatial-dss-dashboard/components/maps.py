"""
Interactive Sri Lanka Geo-Spatial Visualizations using Plotly.

Renders district-level maps focused strictly on Sri Lanka (5.8N to 9.9N, 79.5E to 82.0E)
displaying real-time Market Saturation Risk (MSRS), Dynamic Crop Viability (DCVS),
and spatial price spreads. Coordinated with dark forest-green & charcoal theme.
"""

from typing import Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_sri_lanka_saturation_map(
    df_district_msrs: pd.DataFrame,
    crop_name: str,
    season_label: str,
    height: int = 500,
) -> go.Figure:
    """
    Render Sri Lanka district scatter geo map displaying Market Saturation Risk (MSRS).
    Markers represent approximate district centroids for analytical visualization.

    Args:
        df_district_msrs: Output from compute_district_msrs_table.
        crop_name: Vegetable crop name.
        season_label: Season string (e.g. "Yala 2024").
        height: Height in pixels (optimized for laptop displays).

    Returns:
        Plotly Figure object.
    """
    df_plot = df_district_msrs.copy()

    # Dynamic marker sizing based on committed hectares
    max_ha = df_plot["committed_hectares"].max()
    if max_ha > 0:
        df_plot["marker_size"] = 15 + (df_plot["committed_hectares"] / max_ha) * 25
    else:
        df_plot["marker_size"] = 17

    fig = px.scatter_geo(
        df_plot,
        lat="latitude",
        lon="longitude",
        text="district",
        size="marker_size",
        color="msrs",
        color_continuous_scale=[
            [0.0, "#2E7D32"],   # Safe (< 0.3)
            [0.3, "#2E7D32"],
            [0.5, "#D97706"],   # Moderate (0.3 - 0.7)
            [0.7, "#C53030"],   # Critical (> 0.7)
            [1.0, "#991B1B"],
        ],
        range_color=[0.0, 1.0],
        hover_name="district",
        hover_data={
            "latitude": False,
            "longitude": False,
            "marker_size": False,
            "province": True,
            "msrs": ":.3f",
            "risk_label": True,
            "committed_hectares": ":.1f ha",
            "target_hectares": ":.1f ha",
            "overshoot_pct": ":+.1f%",
        },
        title=f"<b>Regional Market Saturation Risk (MSRS)</b> — {crop_name} ({season_label})",
    )

    fig.update_traces(
        textposition="top center",
        textfont=dict(size=9, color="#FFFFFF", family="Inter, sans-serif"),
        marker=dict(
            opacity=0.92,
            line=dict(width=1.5, color="#FFFFFF"),
        ),
    )

    # Dark Theme Geospatial Layout
    fig.update_geos(
        fitbounds=False,
        visible=True,
        resolution=50,
        showcountries=False,
        showcoastlines=True,
        coastlinecolor="#2E5A38",
        showland=True,
        landcolor="#152A1C",
        showocean=True,
        oceancolor="#0A140D",
        showsubunits=True,
        subunitcolor="#204229",
        center=dict(lat=7.8731, lon=80.7718),
        lataxis_range=[5.7, 10.0],
        lonaxis_range=[79.3, 82.2],
        projection_type="mercator",
    )

    fig.update_layout(
        height=height,
        paper_bgcolor="#0F1E14",
        plot_bgcolor="#0A140D",
        title_font=dict(color="#FFFFFF", size=13),
        margin=dict(l=5, r=5, t=45, b=25),
        coloraxis_colorbar=dict(
            title="MSRS Risk",
            title_font=dict(color="#FFFFFF", size=10),
            tickfont=dict(color="#C8D8CC", size=9),
            tickvals=[0.0, 0.3, 0.7, 1.0],
            ticktext=["0.0 (Safe)", "0.3 (Threshold)", "0.7 (Critical)", "1.0 (Glut Crash)"],
            lenmode="fraction",
            len=0.75,
            thickness=13,
        ),
    )

    # Centroid analytical disclaimer
    fig.add_annotation(
        text="* Markers indicate district centroids for analytical visualization (not administrative polygon boundaries).",
        xref="paper", yref="paper",
        x=0.0, y=-0.04,
        showarrow=False,
        font=dict(size=9, color="#A3B899"),
        xanchor="left",
    )

    return fig


def render_district_viability_map(
    df_district_dcvs: pd.DataFrame,
    crop_name: str,
    season_label: str,
    height: int = 500,
) -> go.Figure:
    """
    Render Sri Lanka map displaying Dynamic Crop Viability Score (DCVS).
    Markers represent approximate district centroids for analytical visualization.

    Args:
        df_district_dcvs: Output from compute_district_dcvs_table.
        crop_name: Vegetable crop name.
        season_label: Season string.
        height: Height in pixels (optimized for laptop displays).

    Returns:
        Plotly Figure object.
    """
    df_plot = df_district_dcvs.copy()
    df_plot["marker_size"] = 18

    fig = px.scatter_geo(
        df_plot,
        lat="latitude",
        lon="longitude",
        text="district",
        size="marker_size",
        color="dcvs",
        color_continuous_scale=[
            [0.0, "#C53030"],   # Low viability
            [0.45, "#D97706"],  # Moderate
            [0.70, "#2E7D32"],  # High
            [1.0, "#1E5631"],   # Prime
        ],
        range_color=[0.0, 100.0],
        hover_name="district",
        hover_data={
            "latitude": False,
            "longitude": False,
            "marker_size": False,
            "province": True,
            "dcvs": ":.1f%",
            "viability_label": True,
            "current_price": "Rs. :.1f",
            "rainfall_avg_mm": ":.1f mm",
            "temp_avg_c": ":.1f °C",
        },
        title=f"<b>Dynamic Crop Viability Score (DCVS)</b> — {crop_name} ({season_label})",
    )

    fig.update_traces(
        textposition="top center",
        textfont=dict(size=9, color="#FFFFFF", family="Inter, sans-serif"),
        marker=dict(
            opacity=0.92,
            line=dict(width=1.5, color="#FFFFFF"),
        ),
    )

    fig.update_geos(
        fitbounds=False,
        visible=True,
        resolution=50,
        showcoastlines=True,
        coastlinecolor="#2E5A38",
        showland=True,
        landcolor="#152A1C",
        showocean=True,
        oceancolor="#0A140D",
        center=dict(lat=7.8731, lon=80.7718),
        lataxis_range=[5.7, 10.0],
        lonaxis_range=[79.3, 82.2],
        projection_type="mercator",
    )

    fig.update_layout(
        height=height,
        paper_bgcolor="#0F1E14",
        plot_bgcolor="#0A140D",
        title_font=dict(color="#FFFFFF", size=13),
        margin=dict(l=5, r=5, t=45, b=25),
        coloraxis_colorbar=dict(
            title="DCVS (%)",
            title_font=dict(color="#FFFFFF", size=10),
            tickfont=dict(color="#C8D8CC", size=9),
            tickvals=[0, 45, 70, 100],
            ticktext=["0% (Unviable)", "45% (Caution)", "70% (Favorable)", "100% (Prime)"],
            lenmode="fraction",
            len=0.75,
            thickness=13,
        ),
    )

    fig.add_annotation(
        text="* Markers indicate district centroids for analytical visualization (not administrative polygon boundaries).",
        xref="paper", yref="paper",
        x=0.0, y=-0.04,
        showarrow=False,
        font=dict(size=9, color="#A3B899"),
        xanchor="left",
    )

    return fig
