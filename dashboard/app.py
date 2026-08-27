"""Geo-Spatial DSS prototype for Sri Lankan vegetable markets.

Run from the repository root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from data_loader import (
    filter_commitment,
    load_commitment,
    load_price,
    load_weather_monthly,
    price_latest_table,
    price_observation_counts,
    price_seasonal_table,
    seasonal_weather,
)
from geo import build_district_map
from recommend import recommend_crops
from scores import compute_score_table, dri_reasons, get_dcvs, get_forecast, get_msrs

st.set_page_config(
    page_title="Geo-Spatial DSS | Vegetable Markets",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Segoe UI, sans-serif", color="#1F2937", size=13),
    margin=dict(l=20, r=20, t=40, b=20),
    hoverlabel=dict(bgcolor="white"),
)

METRIC_OPTIONS = {
    "Cultivation Density": "cultivation_density",
    "Prototype MSRS": "msrs",
    "Prototype DCVS": "dcvs",
    "Prototype DRI": "dri",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: "Source Sans 3", "Segoe UI", sans-serif; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1400px; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        .hero {
            background: linear-gradient(120deg, #0F3D2E 0%, #1B5E4B 55%, #2D6A4F 100%);
            color: #F8FAF9; padding: 1.35rem 1.6rem; border-radius: 16px;
            margin-bottom: 1.1rem;
        }
        .hero h1 { font-size: 1.55rem; margin: 0 0 0.25rem 0; letter-spacing: -0.02em; }
        .hero p { margin: 0; color: #D8EDE4; font-size: 0.98rem; }
        .hero .tag {
            display: inline-block; margin-top: 0.7rem; background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.2); padding: 0.2rem 0.65rem;
            border-radius: 999px; font-size: 0.78rem; letter-spacing: 0.03em;
        }
        .kpi {
            background: #ffffff; border: 1px solid #E5E7EB; border-radius: 14px;
            padding: 0.95rem 1.05rem; box-shadow: 0 1px 2px rgba(15,23,42,0.05);
            min-height: 118px;
        }
        .kpi .label { color: #6B7280; font-size: 0.78rem; text-transform: uppercase;
            letter-spacing: 0.06em; font-weight: 600; }
        .kpi .value { font-size: 1.55rem; font-weight: 700; color: #111827; margin: 0.2rem 0; }
        .kpi .sub { color: #6B7280; font-size: 0.82rem; }
        .section-title { font-size: 1.12rem; font-weight: 700; color: #111827; margin: 0.2rem 0 0.15rem; }
        .section-sub { color: #6B7280; font-size: 0.9rem; margin-bottom: 0.6rem; }
        .map-caption { color: #6B7280; font-size: 0.85rem; margin-top: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading commitment and price datasets...")
def load_core_tables():
    commitment = load_commitment()
    price = load_price()
    return (
        commitment,
        price,
        price_seasonal_table(price),
        price_latest_table(price),
        price_observation_counts(price),
    )


@st.cache_data(show_spinner="Aggregating weather (first run caches a monthly summary)...")
def load_weather_tables():
    monthly = load_weather_monthly()
    return monthly, seasonal_weather(monthly)


def kpi_card(label: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_num(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):,.{digits}f}{suffix}"


def selected_row(score_df: pd.DataFrame, district: str) -> pd.Series | None:
    match = score_df[score_df["district"] == district]
    if match.empty:
        return None
    return match.iloc[0]


inject_css()

commitment, price, price_seasonal, price_latest, price_counts = load_core_tables()
weather_monthly, weather_seasonal_df = load_weather_tables()

districts = sorted(commitment["district"].dropna().unique().tolist())
crops = sorted(commitment["crop_name"].dropna().unique().tolist())
seasons = ["Yala", "Maha"]
years = sorted(commitment["season_year"].dropna().astype(int).unique().tolist(), reverse=True)
price_types = sorted(price["price_type"].dropna().unique().tolist())

st.markdown(
    """
    <div class="hero">
        <h1>Geo-Spatial Decision Support System</h1>
        <p>Data-driven prototype to mitigate the cobweb effect in Sri Lankan vegetable markets</p>
        <span class="tag">RESEARCH PROPOSAL PROTOTYPE &nbsp;·&nbsp; DISTRICT-LEVEL DSS</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Filters")
    st.caption("Views update from the three prototype datasets in `data/v1/`.")

    if "district" not in st.session_state:
        st.session_state.district = "Nuwara Eliya" if "Nuwara Eliya" in districts else districts[0]

    district = st.selectbox("District", districts, key="district")
    crop = st.selectbox("Crop", crops, index=crops.index("Carrot") if "Carrot" in crops else 0)
    season = st.selectbox("Season", seasons, index=0)
    year = st.selectbox("Year", years, index=years.index(2024) if 2024 in years else 0)
    price_type = st.selectbox(
        "Price type",
        price_types,
        index=price_types.index("retail") if "retail" in price_types else 0,
    )
    map_metric_label = st.selectbox("Map metric", list(METRIC_OPTIONS.keys()), index=1)
    st.markdown("---")
    st.caption(
        "Commitment: `synthetic_proxy`  \n"
        "Price: `harti`  \n"
        "Weather: `nasa_power`"
    )

map_metric = METRIC_OPTIONS[map_metric_label]

crop_scores = compute_score_table(
    commitment,
    price_seasonal,
    price_latest,
    price_counts,
    weather_seasonal_df,
    year=year,
    season=season,
    price_type=price_type,
    crop=crop,
)
all_crop_scores = compute_score_table(
    commitment,
    price_seasonal,
    price_latest,
    price_counts,
    weather_seasonal_df,
    year=year,
    season=season,
    price_type=price_type,
    crop=None,
)

if crop_scores.empty:
    st.warning(
        f"No commitment records for **{crop}** in **{season} {year}**. "
        "Yala 2014 is absent from the prototype commitment file. Choose another season or year."
    )
    st.stop()

row = selected_row(crop_scores, district)
if row is None:
    st.warning(f"No commitment row for **{district}** / **{crop}** in {season} {year}.")
    st.stop()

msrs_val = get_msrs(district, crop, season, year, crop_scores)
dcvs_val = get_dcvs(district, crop, season, year, crop_scores)
forecast = get_forecast(district, crop, season, year)

# --- KPIs ---
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card(
        "Committed hectares",
        fmt_num(row.get("committed_hectares"), 1, " ha"),
        f"Target {fmt_num(row.get('target_hectares'), 1, ' ha')}",
    )
with k2:
    if pd.isna(row.get("latest_price")):
        kpi_card("Latest available price", "No price data available", f"{price_type.title()} · HARTI")
    else:
        date_txt = (
            pd.to_datetime(row["price_date"]).strftime("%Y-%m-%d")
            if pd.notna(row.get("price_date"))
            else "date unknown"
        )
        kpi_card(
            "Latest available price",
            f"Rs. {fmt_num(row.get('latest_price'), 2)}",
            f"{price_type.title()} · {date_txt}",
        )
with k3:
    kpi_card("Prototype MSRS", fmt_num(msrs_val, 1), "Saturation risk · 0–100")
with k4:
    kpi_card("Prototype DCVS", fmt_num(dcvs_val, 1), "Crop viability · 0–100")
with k5:
    kpi_card("Prototype DRI", fmt_num(row.get("dri"), 1), "Data reliability · 0–100")

st.markdown("")

# --- Map ---
st.markdown('<div class="section-title">Sri Lankan district map</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-sub">Showing <b>{map_metric_label}</b> for <b>{crop}</b> · '
    f"{season} {year}. Circle size follows committed hectares. "
    "Click a district for details.</div>",
    unsafe_allow_html=True,
)

fmap = build_district_map(crop_scores, map_metric, map_metric_label, district)
map_state = st_folium(
    fmap,
    height=560,
    width=1200,
    returned_objects=["last_object_clicked_tooltip"],
    key="district_map",
)
st.markdown(
    '<div class="map-caption">Prototype district visualization — approximate centroids. '
    "Official GeoJSON boundaries are not included in this prototype.</div>",
    unsafe_allow_html=True,
)

clicked = None
if map_state:
    clicked = map_state.get("last_object_clicked_tooltip")
if clicked and clicked in districts and clicked != st.session_state.district:
    st.session_state.district = clicked
    st.rerun()

# --- District / crop details ---
st.markdown("---")
st.markdown('<div class="section-title">District and crop details</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-sub">{district} · {crop} · {season} {year}</div>',
    unsafe_allow_html=True,
)

c_left, c_right = st.columns([1.15, 1])
with c_left:
    intensity = row.get("cultivation_intensity")
    crowding = row.get("crowding")
    overshoot = row.get("overshoot")
    st.markdown(
        f"""
        **Cultivation**
        - Committed area: **{fmt_num(row.get("committed_hectares"), 1)} ha**
        - Target area: **{fmt_num(row.get("target_hectares"), 1)} ha**
        - Cultivation intensity (committed / target): **{fmt_num(intensity, 2)}**
        - Crowding vs crop median: **{fmt_num(crowding, 2)}×**
        - Overshoot vs target: **{fmt_num(overshoot, 2)}**
        - Commitment source: `{row.get("source")}`
        """
    )
    rain = row.get("season_rainfall_mm")
    temp = row.get("season_temp_c")
    if pd.isna(rain):
        weather_line = "No seasonal weather summary for this district / season / year."
    else:
        weather_line = (
            f"Season rainfall **{fmt_num(rain, 1)} mm**, mean temperature "
            f"**{fmt_num(temp, 1)} °C** (NASA POWER monthly aggregate)."
        )
    st.markdown(f"**Weather**  \n{weather_line}")

with c_right:
    st.markdown("**Why these prototype scores?**")
    st.markdown(
        f"- Prototype MSRS uses crowding `{fmt_num(crowding, 2)}` and overshoot `{fmt_num(overshoot, 2)}`."
    )
    if pd.isna(row.get("price_score")):
        st.markdown("- Prototype DCVS has **no seasonal price component** for this filter.")
    else:
        st.markdown(f"- Prototype price score: **{fmt_num(row.get('price_score'), 1)}**.")
    if pd.isna(row.get("weather_score")):
        st.markdown("- Prototype DCVS has **no weather typicality component** for this filter.")
    else:
        st.markdown(f"- Prototype weather typicality: **{fmt_num(row.get('weather_score'), 1)}**.")
    st.markdown("**Prototype DRI breakdown**")
    for reason in dri_reasons(row):
        st.caption(f"• {reason}")

st.info(forecast["message"])

# --- Recommendations ---
st.markdown("---")
st.markdown('<div class="section-title">Crop recommendations</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-sub">Ranked by prototype DCVS for <b>{district}</b> in {season} {year}. '
    "Status rules are a demo heuristic, not an agronomic prescription.</div>",
    unsafe_allow_html=True,
)

district_all = all_crop_scores[all_crop_scores["district"] == district]
rec_table = recommend_crops(district_all)
show_cols = ["Crop", "DCVS", "MSRS", "DRI", "Status", "Reason"]
if rec_table.empty:
    st.warning("No crop scores could be produced for this district and season.")
else:
    def _color_status(status: str) -> str:
        if status == "Recommended":
            return "background-color: #D1FAE5; color: #065F46"
        if status == "High Risk / Avoid":
            return "background-color: #FEE2E2; color: #991B1B"
        if status == "Insufficient data":
            return "background-color: #F3F4F6; color: #4B5563"
        return "background-color: #FEF3C7; color: #92400E"

    styled = rec_table[show_cols].style.map(_color_status, subset=["Status"]).format(
        {"DCVS": lambda v: "n/a" if pd.isna(v) else f"{v:.1f}",
         "MSRS": lambda v: "n/a" if pd.isna(v) else f"{v:.1f}",
         "DRI": lambda v: "n/a" if pd.isna(v) else f"{v:.1f}"}
    )
    st.dataframe(styled, width="stretch", hide_index=True)

# --- Charts ---
st.markdown("---")
st.markdown('<div class="section-title">Price, cultivation and weather</div>', unsafe_allow_html=True)

tab_price, tab_cult, tab_weather = st.tabs(
    ["Price analysis", "Cultivation analysis", "Weather analysis"]
)

with tab_price:
    price_sub = price[
        (price["district"] == district)
        & (price["crop"] == crop)
        & (price["price_type"] == price_type)
    ].sort_values("price_date")
    if price_sub.empty:
        st.warning(
            f"No price data available for **{district}** / **{crop}** / **{price_type}** "
            "in the HARTI prototype file."
        )
    else:
        weekly = price_sub[price_sub["granularity"].str.endswith("W", na=False)]
        plot_df = weekly if not weekly.empty else price_sub
        fig = px.line(
            plot_df,
            x="price_date",
            y="price",
            title=f"{crop} {price_type} price — {district}",
        )
        fig.update_traces(line_color="#1B5E4B", line_width=2)
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Price (Rs.)", xaxis_title="")
        st.plotly_chart(fig, width="stretch")
        latest = plot_df.sort_values("price_date").iloc[-1]
        st.caption(
            f"Latest {price_type} observation: Rs. {latest['price']:.2f} on "
            f"{pd.to_datetime(latest['price_date']).date()} "
            f"({latest['granularity']}, source `{latest['source']}`). "
            "Missing district–crop combinations are left blank, not imputed."
        )
        type_counts = (
            price[(price["district"] == district) & (price["crop"] == crop)]
            .groupby("price_type")
            .size()
            .rename("rows")
            .reset_index()
        )
        st.dataframe(type_counts, hide_index=True, width="stretch")

with tab_cult:
    cult_year = filter_commitment(commitment, year, season, crop=crop)
    if cult_year.empty:
        st.warning("No cultivation records for this filter.")
    else:
        cult_plot = cult_year.sort_values("committed_hectares", ascending=False)
        fig = go.Figure()
        fig.add_bar(
            x=cult_plot["district"],
            y=cult_plot["committed_hectares"],
            name="Committed ha",
            marker_color="#1B5E4B",
        )
        fig.add_bar(
            x=cult_plot["district"],
            y=cult_plot["target_hectares"],
            name="Target ha",
            marker_color="#A7C4B5",
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            title=f"{crop} area by district — {season} {year}",
            yaxis_title="Hectares",
            xaxis_title="",
            xaxis_tickangle=-40,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width="stretch")

        hist = commitment[
            (commitment["district"] == district) & (commitment["crop_name"] == crop)
        ].sort_values(["season_year", "season_type"])
        hist["season_label"] = hist["season_year"].astype(str) + " " + hist["season_type"]
        fig2 = px.line(
            hist,
            x="season_label",
            y=["committed_hectares", "target_hectares"],
            title=f"{crop} committed vs target — {district}",
        )
        fig2.update_layout(**PLOTLY_LAYOUT, yaxis_title="Hectares", xaxis_title="", legend_title="")
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "Cultivation intensity = committed hectares / target hectares. "
            "All commitment rows are labelled `synthetic_proxy` in the source file."
        )

with tab_weather:
    wx = weather_monthly[weather_monthly["district"] == district].sort_values(["year", "month"])
    if wx.empty:
        st.warning(f"No weather data available for **{district}**.")
    else:
        wx = wx.copy()
        wx["month_date"] = pd.to_datetime(
            dict(year=wx["year"].astype(int), month=wx["month"].astype(int), day=1)
        )
        fig = go.Figure()
        fig.add_bar(
            x=wx["month_date"],
            y=wx["rainfall_mm"],
            name="Monthly rainfall (mm)",
            marker_color="#4C8D6E",
            opacity=0.75,
        )
        fig.add_trace(
            go.Scatter(
                x=wx["month_date"],
                y=wx["temp_c"],
                name="Mean temperature (°C)",
                yaxis="y2",
                line=dict(color="#B45309", width=2),
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Monthly weather — {district}",
            yaxis=dict(title="Rainfall (mm)"),
            yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width="stretch")

        season_wx = weather_seasonal_df[weather_seasonal_df["district"] == district].sort_values(
            ["season_year", "season_type"]
        )
        if not season_wx.empty:
            season_wx = season_wx.copy()
            season_wx["label"] = (
                season_wx["season_year"].astype(str) + " " + season_wx["season_type"]
            )
            fig_s = px.bar(
                season_wx,
                x="label",
                y="rainfall_mm",
                color="season_type",
                title=f"Seasonal rainfall — {district}",
                color_discrete_map={"Yala": "#2D6A4F", "Maha": "#95B8A8"},
            )
            fig_s.update_layout(**PLOTLY_LAYOUT, yaxis_title="Rainfall (mm)", xaxis_title="", legend_title="")
            st.plotly_chart(fig_s, width="stretch")
        st.caption(
            "Weather is aggregated from `weather_data.json` (source `nasa_power`) to monthly "
            "and seasonal summaries and cached. The raw JSON is not re-parsed on every click. "
            "Coverage starts 2015-01-01, so 2014 seasons may have no weather component."
        )

# --- Methodology ---
st.markdown("---")
with st.expander("Prototype Assumptions & Methodology", expanded=False):
    st.markdown(
        """
        This dashboard is a **visual prototype for the research proposal presentation**.
        It is not the completed Geo-Spatial DSS and it does not claim validated agronomic advice.

        **Datasets (supplied by the project team, used as-is)**
        - `data/v1/analysis_ready_commitment.csv` — source labelled **`synthetic_proxy`**
        - `data/v1/analysis_ready_price.csv` — source labelled **`harti`**
        - `data/v1/weather_data.json` — source labelled **`nasa_power`**

        No new HARTI series, NASA POWER extracts, or other agricultural observations
        were downloaded for this prototype. Original files are not modified.

        **Map**
        - Official district GeoJSON is not in the repository.
        - Districts are drawn as circles at **approximate centroids**.
        - Label: *Prototype district visualization — approximate centroids*.

        **Prototype MSRS** (Market Saturation Risk Score)
        - `crowding = committed_hectares / median committed hectares of the same crop`
        - `overshoot = committed_hectares / target_hectares`
        - Each ratio is scaled so that 0.5 → 0 and 2.0 → 100, then averaged and clipped to 0–100.
        - This is **not** the final crowding / MSRS research model.

        **Prototype DCVS** (Dynamic Crop Viability Score)
        - Price component: seasonal mean price vs that district–crop–season historical median.
        - Weather component: how typical seasonal rainfall and temperature are versus the
          district's own historical distribution (not a crop suitability model).
        - Crowding component: `100 − MSRS`.
        - Default mix 0.45 / 0.35 / 0.20; missing components are dropped and weights are
          renormalised. Missing values are not invented.

        **Prototype DRI** (Data Reliability Index)
        - Uses the actual `source` fields and completeness of overlapping records.
        - Placeholder source weights: `synthetic_proxy` 0.35, `harti` 0.90, `nasa_power` 0.85.
        - These are **not** final reliability weights.

        **Seasons (prototype calendar mapping)**
        - Yala: April–September of the season year.
        - Maha: October of the season year through March of the following year.

        **Future integration**
        - `get_msrs()`, `get_dcvs()`, and `get_forecast()` in `scores.py` are hooks.
        - Later they can call the other members' MSRS, DCVS and forecasting APIs.
        - Those APIs are not built in this prototype.
        """
    )
