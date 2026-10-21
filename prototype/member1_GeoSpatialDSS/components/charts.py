"""
Interactive Plotly charts for price dynamics, Cobweb phase plots, and regional balances.
Optimized for dark forest-green & charcoal theme, laptop screen fit, and non-overlapping layouts.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure dashboard root is on sys.path
_dashboard_root = str(Path(__file__).resolve().parent.parent)
if _dashboard_root not in sys.path:
    sys.path.insert(0, _dashboard_root)

from config import (
    COLOR_FOREST_GREEN,
    COLOR_SAGE_GREEN,
    COLOR_ACCENT_MINT,
    COLOR_WARNING_AMBER,
    COLOR_CRITICAL_RED,
    COLOR_BG_DARK,
    COLOR_SURFACE_DARK,
    COLOR_BORDER_DARK,
    COLOR_TEXT_WHITE,
    COLOR_TEXT_LIGHT_GREY,
    COLOR_TEXT_MUTED,
)

# Shared Dark Forest-Green Layout Configuration
DARK_PLOT_THEME: Dict[str, Any] = dict(
    paper_bgcolor=COLOR_SURFACE_DARK,
    plot_bgcolor=COLOR_BG_DARK,
    font=dict(family="Inter, sans-serif", color=COLOR_TEXT_LIGHT_GREY, size=11),
)

# Shared Dark Horizontal Legend Configuration (Non-overlapping above plot)
DARK_LEGEND: Dict[str, Any] = dict(
    orientation="h",
    yanchor="bottom",
    y=1.03,
    xanchor="right",
    x=1.0,
    bgcolor="rgba(15, 30, 20, 0.85)",
    bordercolor=COLOR_BORDER_DARK,
    borderwidth=1,
    font=dict(color=COLOR_TEXT_LIGHT_GREY, size=10),
)


def apply_dark_axes(fig: go.Figure) -> None:
    """Apply muted green gridlines and light labels to axes."""
    fig.update_xaxes(
        gridcolor="rgba(74, 124, 89, 0.18)",
        zerolinecolor="rgba(74, 124, 89, 0.30)",
        color=COLOR_TEXT_MUTED,
        title_font=dict(color=COLOR_TEXT_LIGHT_GREY, size=11),
        tickfont=dict(color=COLOR_TEXT_MUTED, size=10),
    )
    fig.update_yaxes(
        gridcolor="rgba(74, 124, 89, 0.18)",
        zerolinecolor="rgba(74, 124, 89, 0.30)",
        color=COLOR_TEXT_MUTED,
        title_font=dict(color=COLOR_TEXT_LIGHT_GREY, size=11),
        tickfont=dict(color=COLOR_TEXT_MUTED, size=10),
    )


def render_price_history_chart(
    df_price: pd.DataFrame,
    crop_name: str,
    district_filter: Optional[str] = None,
    height: int = 370,
) -> go.Figure:
    """
    Render weekly price historical time series with rolling moving average.
    Accurately constrained to observed dates (no artificial future extrapolation).
    """
    if district_filter and district_filter != "All Island (National Average)":
        sub = df_price[(df_price["crop"] == crop_name) & (df_price["district"] == district_filter)]
    else:
        sub = df_price[df_price["crop"] == crop_name]

    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No price records available for {crop_name} in {district_filter}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color=COLOR_TEXT_MUTED)
        )
        fig.update_layout(**DARK_PLOT_THEME, height=height)
        return fig

    if not pd.api.types.is_datetime64_any_dtype(sub["price_date"]):
        sub = sub.copy()
        sub["price_date"] = pd.to_datetime(sub["price_date"])

    sub = sub.sort_values(by="price_date")
    piv = sub.groupby(["price_date", "price_type"])["price"].mean().reset_index()
    piv = piv.sort_values(by="price_date")

    fig = go.Figure()

    retail = piv[piv["price_type"] == "retail"].sort_values("price_date").reset_index(drop=True)
    wholesale = piv[piv["price_type"] == "wholesale"].sort_values("price_date").reset_index(drop=True)

    # Accurate Observed Date Range
    all_dates = piv["price_date"]
    min_date = all_dates.min()
    max_date = all_dates.max()

    # 1. Retail Series (Mint / Sage Green)
    if not retail.empty:
        fig.add_trace(go.Scatter(
            x=retail["price_date"],
            y=retail["price"],
            mode="lines",
            name="Retail (Weekly)",
            line=dict(color="#4ADE80", width=1.5),
            opacity=0.65,
            hovertemplate="<b>Retail Observed</b>: Rs. %{y:.1f}/kg<extra></extra>",
        ))
        # Add 8-week moving average only where at least 8 observations exist
        if len(retail) >= 8:
            retail["rolling_ma"] = retail["price"].rolling(window=8, min_periods=8).mean()
            ma_valid = retail.dropna(subset=["rolling_ma"])
            if not ma_valid.empty:
                fig.add_trace(go.Scatter(
                    x=ma_valid["price_date"],
                    y=ma_valid["rolling_ma"],
                    mode="lines",
                    name="Retail 8-Wk Trend (MA)",
                    line=dict(color="#22C55E", width=2.8),
                    hovertemplate="<b>Retail 8W MA</b>: Rs. %{y:.1f}/kg<extra></extra>",
                ))

    # 2. Wholesale Series (Warm Amber / Gold)
    if not wholesale.empty:
        fig.add_trace(go.Scatter(
            x=wholesale["price_date"],
            y=wholesale["price"],
            mode="lines",
            name="Wholesale (Weekly)",
            line=dict(color="#FBBF24", width=1.5),
            opacity=0.60,
            hovertemplate="<b>Wholesale Observed</b>: Rs. %{y:.1f}/kg<extra></extra>",
        ))
        # Add 8-week moving average only where at least 8 observations exist
        if len(wholesale) >= 8:
            wholesale["rolling_ma"] = wholesale["price"].rolling(window=8, min_periods=8).mean()
            ma_valid_ws = wholesale.dropna(subset=["rolling_ma"])
            if not ma_valid_ws.empty:
                fig.add_trace(go.Scatter(
                    x=ma_valid_ws["price_date"],
                    y=ma_valid_ws["rolling_ma"],
                    mode="lines",
                    name="Wholesale 8-Wk Trend (MA)",
                    line=dict(color="#D97706", width=2.5, dash="dash"),
                    hovertemplate="<b>Wholesale 8W MA</b>: Rs. %{y:.1f}/kg<extra></extra>",
                ))

    loc_str = district_filter if district_filter else "National Average"
    fig.update_layout(
        **DARK_PLOT_THEME,
        title=dict(
            text=f"<b>Weekly Price Trajectory & Trend</b><br><span style='font-size:10px;color:{COLOR_TEXT_MUTED}'>{crop_name} ({loc_str})</span>",
            font=dict(color=COLOR_TEXT_WHITE, size=12),
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(
            title="Timeline (Weekly ISO Monday)",
            range=[min_date, max_date],
        ),
        yaxis=dict(title="Price (Rs./kg)"),
        height=height,
        hovermode="x unified",
        legend=DARK_LEGEND,
        margin=dict(l=60, r=25, t=85, b=45),
    )

    apply_dark_axes(fig)
    return fig


def render_cobweb_phase_plot(
    df_price: pd.DataFrame,
    df_comm: pd.DataFrame,
    crop_name: str,
    height: int = 370,
) -> go.Figure:
    """
    Render empirical Cobweb phase plot (Price vs. Committed Supply Extent).
    Non-overlapping layout, selective annotations, and accurate axis boundaries.
    """
    comm_agg = (
        df_comm[df_comm["crop_name"] == crop_name]
        .groupby(["season_year", "season_type"])
        .agg({"committed_hectares": "sum", "target_hectares": "sum"})
        .reset_index()
    )

    p_df = df_price[df_price["crop"] == crop_name].copy()
    p_df["price_date"] = pd.to_datetime(p_df["price_date"])
    p_df["month"] = p_df["price_date"].dt.month
    p_df["year"] = p_df["price_date"].dt.year
    p_df["season_type"] = np.where(p_df["month"].between(4, 9), "Yala", "Maha")
    p_df["season_year"] = np.where(p_df["month"] >= 10, p_df["year"], np.where(p_df["month"] <= 3, p_df["year"] - 1, p_df["year"]))

    p_agg = p_df.groupby(["season_year", "season_type"])["price"].mean().reset_index()

    merged = pd.merge(comm_agg, p_agg, on=["season_year", "season_type"]).sort_values(
        by=["season_year", "season_type"]
    )

    if merged.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient multi-season series for Cobweb phase plot",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color=COLOR_TEXT_MUTED)
        )
        fig.update_layout(**DARK_PLOT_THEME, height=height)
        return fig

    merged["season_label"] = merged["season_type"] + " " + merged["season_year"].astype(str)

    fig = go.Figure()

    # Dynamic line trajectory with hovertooltips (no text label collisions)
    fig.add_trace(go.Scatter(
        x=merged["committed_hectares"],
        y=merged["price"],
        mode="lines+markers",
        text=merged["season_label"],
        hovertemplate="<b>%{text}</b><br>Committed Extent: %{x:,.0f} ha<br>Average Price: Rs. %{y:.1f}/kg<extra></extra>",
        line=dict(color="#52B788", width=2.5, dash="solid"),
        marker=dict(size=8, color="#22C55E", line=dict(width=1.5, color="#FFFFFF")),
        name="Cobweb Spiral Trajectory",
    ))

    # Add selective annotations only for start and finish (prevent collisions)
    if len(merged) > 1:
        first_pt = merged.iloc[0]
        fig.add_annotation(
            x=first_pt["committed_hectares"],
            y=first_pt["price"],
            text=f"Start: {first_pt['season_label']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="#A3B899",
            font=dict(size=9, color="#E8EFEA", family="Inter, sans-serif"),
            bgcolor="rgba(15, 30, 20, 0.90)",
            bordercolor="#2B5237",
            borderwidth=1,
            borderpad=4,
            ax=-40,
            ay=-30,
        )
        last_pt = merged.iloc[-1]
        fig.add_annotation(
            x=last_pt["committed_hectares"],
            y=last_pt["price"],
            text=f"Latest: {last_pt['season_label']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="#4ADE80",
            font=dict(size=10, color="#FFFFFF", family="Inter, sans-serif"),
            bgcolor="rgba(10, 20, 13, 0.95)",
            bordercolor="#4ADE80",
            borderwidth=1.5,
            borderpad=5,
            ax=45,
            ay=-40,
        )

    mean_target = merged["target_hectares"].mean()
    fig.add_vline(
        x=mean_target,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Quota Threshold",
        annotation_position="top left",
        annotation_font=dict(color="#F87171", size=10),
    )

    # Meaningful axis ranges strictly based on empirical observations
    x_min = merged["committed_hectares"].min() * 0.90
    x_max = max(merged["committed_hectares"].max(), mean_target) * 1.08
    y_min = merged["price"].min() * 0.85
    y_max = merged["price"].max() * 1.12

    fig.update_layout(
        **DARK_PLOT_THEME,
        title=dict(
            text=f"<b>Empirical Cobweb Dynamics</b><br><span style='font-size:10px;color:{COLOR_TEXT_MUTED}'>{crop_name} Phase-Space Trajectory</span>",
            font=dict(color=COLOR_TEXT_WHITE, size=12),
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(
            title="National Committed Extent (Hectares)",
            range=[x_min, x_max],
        ),
        yaxis=dict(
            title="Average Realized Price (Rs./kg)",
            range=[y_min, y_max],
        ),
        height=height,
        legend=DARK_LEGEND,
        margin=dict(l=60, r=25, t=85, b=45),
    )

    apply_dark_axes(fig)
    return fig


def render_district_commitment_barchart(
    df_district_msrs: pd.DataFrame,
    crop_name: str,
    season_label: str,
    height: int = 370,
) -> go.Figure:
    """
    Render grouped bar chart comparing Committed vs. Target Hectares across districts.
    Dark-theme coordinated with distinct capacity bars and unclipped rotated district labels.
    """
    top_d = df_district_msrs.sort_values(by="committed_hectares", ascending=False).head(12)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_d["district"],
        y=top_d["committed_hectares"],
        name="Committed Area (ha)",
        marker_color=top_d["color"],
        hovertemplate="<b>%{x}</b><br>Committed: %{y:.1f} ha<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=top_d["district"],
        y=top_d["target_hectares"],
        name="Target Capacity (ha)",
        marker_color="#6B8E73",
        hovertemplate="<b>%{x}</b><br>Target Capacity: %{y:.1f} ha<extra></extra>",
    ))

    fig.update_layout(
        **DARK_PLOT_THEME,
        barmode="group",
        title=dict(
            text=f"<b>Commitment vs. Sustainable Quota</b><br><span style='font-size:10px;color:{COLOR_TEXT_MUTED}'>{crop_name} ({season_label})</span>",
            font=dict(color=COLOR_TEXT_WHITE, size=12),
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(
            title="Administrative District",
            tickangle=-25,
        ),
        yaxis=dict(title="Area (Hectares)"),
        height=height,
        legend=DARK_LEGEND,
        margin=dict(l=60, r=25, t=85, b=45),
    )

    apply_dark_axes(fig)
    return fig


def render_weather_trend_chart(
    df_weather: pd.DataFrame,
    district: str,
    height: int = 350,
) -> go.Figure:
    """
    Render dual-axis chart of weekly precipitation accumulation and mean temperature.
    Coordinates with dark forest theme and constraints to verified 52-week data.
    """
    sub = df_weather[df_weather["district"] == district]
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No weather records available for {district}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color=COLOR_TEXT_MUTED)
        )
        fig.update_layout(**DARK_PLOT_THEME, height=height)
        return fig

    if not pd.api.types.is_datetime64_any_dtype(sub["obs_date"]):
        sub = sub.copy()
        sub["obs_date"] = pd.to_datetime(sub["obs_date"])

    sub = sub.copy()
    sub["week"] = sub["obs_date"].dt.to_period("W").dt.start_time
    weekly_w = sub.groupby("week").agg({"rainfall_mm": "sum", "temp_c": "mean"}).reset_index()

    weekly_w = weekly_w.tail(52)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=weekly_w["week"],
            y=weekly_w["rainfall_mm"],
            name="Weekly Rainfall (mm)",
            marker_color="#4ADE80",
            opacity=0.80,
            hovertemplate="<b>Week</b>: %{x|%Y-%m-%d}<br>Rainfall: %{y:.1f} mm<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=weekly_w["week"],
            y=weekly_w["temp_c"],
            name="Mean Temp (°C)",
            line=dict(color="#F59E0B", width=2.5),
            hovertemplate="<b>Week</b>: %{x|%Y-%m-%d}<br>Temperature: %{y:.1f} °C<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        **DARK_PLOT_THEME,
        title=dict(
            text=f"<b>Agro-Meteorological Environment</b><br><span style='font-size:10px;color:{COLOR_TEXT_MUTED}'>{district} (NASA POWER Last 52 Weeks)</span>",
            font=dict(color=COLOR_TEXT_WHITE, size=12),
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(title="Timeline (Weekly)"),
        height=height,
        legend=DARK_LEGEND,
        margin=dict(l=60, r=50, t=85, b=45),
    )

    fig.update_yaxes(
        title_text="Rainfall (mm/week)",
        secondary_y=False,
        gridcolor="rgba(74, 124, 89, 0.18)",
        color=COLOR_TEXT_MUTED,
        title_font=dict(color=COLOR_TEXT_LIGHT_GREY, size=11),
    )
    fig.update_yaxes(
        title_text="Temperature (°C)",
        secondary_y=True,
        gridcolor="rgba(74, 124, 89, 0.18)",
        color=COLOR_TEXT_MUTED,
        title_font=dict(color=COLOR_TEXT_LIGHT_GREY, size=11),
    )

    return fig


def render_forecast_trajectory_chart(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    crop_name: str,
    district: str,
    height: int = 380,
) -> go.Figure:
    """
    Render historical observed price trajectory seamlessly connected to forward
    forecast uncertainty corridors. Handles both active demo simulation and standby baseline.
    """
    fig = go.Figure()

    # 0. Empty State Check
    if historical_df.empty and (forecast_df.empty or "predicted_price" not in forecast_df.columns):
        fig.add_annotation(
            text=f"No verified historical price observations available for {crop_name} in {district}.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=13, color=COLOR_TEXT_LIGHT_GREY),
            bgcolor="rgba(19, 38, 25, 0.90)",
            bordercolor=COLOR_BORDER_DARK,
            borderwidth=1,
            borderpad=10,
        )
        fig.update_layout(
            **DARK_PLOT_THEME,
            title=dict(
                text=f"<b>Observed Historical Baseline</b><br><span style='font-size:10px;color:{COLOR_TEXT_MUTED}'>{crop_name} ({district})</span>",
                font=dict(color=COLOR_TEXT_WHITE, size=12),
                x=0.01, y=0.98, xanchor="left", yanchor="top",
            ),
            xaxis=dict(title="Timeline (Weekly ISO Monday)"),
            yaxis=dict(title="Price (Rs./kg)"),
            height=height,
            margin=dict(l=60, r=25, t=85, b=45),
        )
        apply_dark_axes(fig)
        return fig

    # 1. Historical Observed Series (Genuine observations with gap preservation)
    if not historical_df.empty:
        hist_sorted = historical_df.sort_values(by="price_date").copy()
        hist_sorted["price_date"] = pd.to_datetime(hist_sorted["price_date"])

        # Detect intermediate missing weeks (>14 days) and insert NaN to leave gaps rather than drawing misleading continuous lines
        records_with_gaps = []
        for i in range(len(hist_sorted)):
            curr_row = hist_sorted.iloc[i]
            if i > 0:
                prev_row = hist_sorted.iloc[i - 1]
                day_gap = (curr_row["price_date"] - prev_row["price_date"]).days
                if day_gap > 14:
                    # Intermediate gap detected: insert NaN marker so Plotly leaves a gap
                    records_with_gaps.append({
                        "price_date": prev_row["price_date"] + pd.Timedelta(days=7),
                        "price": np.nan,
                    })
            records_with_gaps.append({
                "price_date": curr_row["price_date"],
                "price": curr_row["price"],
            })
        plot_hist_df = pd.DataFrame(records_with_gaps)

        # Plot with markers and no gap bridging
        fig.add_trace(go.Scatter(
            x=plot_hist_df["price_date"],
            y=plot_hist_df["price"],
            mode="lines+markers",
            name="Verified Historical Observations (HARTI)",
            line=dict(color="#4ADE80", width=2.5),
            marker=dict(size=5, color="#4ADE80"),
            connectgaps=False,  # Never interpolate or connect unobserved missing weeks
            hovertemplate="<b>Observed Date</b>: %{x|%Y-%m-%d}<br><b>Verified Price</b>: Rs. %{y:.1f}/kg<extra></extra>",
        ))

        # If only 1 or 2 observations exist, provide transparent limited coverage disclosure
        if len(hist_sorted) <= 2:
            fig.add_annotation(
                text=f"Limited historical coverage: {len(hist_sorted)} observation(s) recorded in dataset.",
                xref="paper", yref="paper",
                x=0.02, y=0.08,
                showarrow=False,
                font=dict(size=11, color="#F59E0B"),
                bgcolor="rgba(20, 30, 22, 0.90)",
                bordercolor="#F59E0B",
                borderwidth=1,
                borderpad=5,
            )

    # 2. Forward Forecast Projection with Confidence Interval
    if not forecast_df.empty and "predicted_price" in forecast_df.columns:
        fc_sorted = forecast_df.sort_values(by="forecast_date")
        fc_sorted["forecast_date"] = pd.to_datetime(fc_sorted["forecast_date"])

        # Connect the last historical observation to the first forecast point
        if not historical_df.empty:
            last_hist_date = hist_sorted["price_date"].iloc[-1]
            last_hist_price = hist_sorted["price"].iloc[-1]
            bridge_x = [last_hist_date] + list(fc_sorted["forecast_date"])
            bridge_y = [last_hist_price] + list(fc_sorted["predicted_price"])
            bridge_upper = [last_hist_price] + list(fc_sorted["upper_bound_95"])
            bridge_lower = [last_hist_price] + list(fc_sorted["lower_bound_95"])
        else:
            bridge_x = list(fc_sorted["forecast_date"])
            bridge_y = list(fc_sorted["predicted_price"])
            bridge_upper = list(fc_sorted["upper_bound_95"])
            bridge_lower = list(fc_sorted["lower_bound_95"])

        # 95% Calibrated Confidence Band (Genuine models only)
        fig.add_trace(go.Scatter(
            x=bridge_x + bridge_x[::-1],
            y=bridge_upper + bridge_lower[::-1],
            fill="toself",
            fillcolor="rgba(34, 197, 94, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Confidence Interval",
            showlegend=True,
        ))

        # Projected Mean Trajectory
        fig.add_trace(go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name="Forecast Trajectory",
            line=dict(color="#22C55E", width=2.5, dash="dash"),
            marker=dict(size=6, color="#22C55E"),
            hovertemplate="<b>Forecast Date</b>: %{x|%Y-%m-%d}<br><b>Projected Price</b>: Rs. %{y:.1f}/kg<extra></extra>",
        ))

        title_text = f"<b>Price Trajectory & Forecast Projection</b><br><span style='font-size:10px;color:{COLOR_TEXT_MUTED}'>{crop_name} ({district})</span>"
    else:
        title_text = f"<b>Observed Historical Baseline</b><br><span style='font-size:10px;color:{COLOR_TEXT_MUTED}'>{crop_name} ({district})</span>"
        # Badge placed at top-right corner to avoid colliding with plot lines or axes
        fig.add_annotation(
            text="Forecast service not connected — Verified historical baseline",
            xref="paper", yref="paper",
            x=0.98, y=0.95,
            xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(size=11, color=COLOR_TEXT_LIGHT_GREY),
            bgcolor="rgba(19, 38, 25, 0.88)",
            bordercolor=COLOR_BORDER_DARK,
            borderwidth=1,
            borderpad=6,
        )

    fig.update_layout(
        **DARK_PLOT_THEME,
        title=dict(
            text=title_text,
            font=dict(color=COLOR_TEXT_WHITE, size=12),
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        xaxis=dict(title="Timeline (Weekly ISO Monday)"),
        yaxis=dict(title="Price (Rs./kg)"),
        height=height,
        hovermode="x unified",
        legend=DARK_LEGEND,
        margin=dict(l=60, r=25, t=85, b=45),
    )

    apply_dark_axes(fig)
    return fig
