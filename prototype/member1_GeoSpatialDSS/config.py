"""
Configuration, constants, district geographic coordinates, crop vocabularies,
and visual theme tokens for the Member 1 Geo-Spatial DSS Dashboard.
"""

from typing import Dict, List, Any, Tuple, Optional

# Dashboard Metadata & Research Identity
RESEARCH_TOPIC = "Data-Driven Decision Support System to Mitigate the “Cobweb Effect” in Sri Lankan Vegetable Markets"
SIDEBAR_BRAND_TITLE = "Geo-Spatial Decision Support System"
SIDEBAR_BRAND_SUBTITLE = "Sri Lankan Vegetable Market Intelligence"
SIDEBAR_BRAND_DESC = "Data-driven insights to understand and mitigate the Cobweb Effect in Sri Lankan vegetable markets."
RESEARCH_MEMBER = "Member 1 — Data Engineering & Geo-Spatial Decision Support System"
SLIIT_PROJECT_ID = "SLIIT-Research-Project-2026-27-DS-322"  # Retained in config for internal reference only

# Required Collaboration Footer & Truthful Provenance Note
FOOTER_TEXT = "Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI) collaboration | SLIIT | J26-DS-322"
COMMITMENT_PROVENANCE_NOTE = "Cultivation commitment values for the prototype are derived from historical HARTI district records and refreshed per season."

# Standardized 6 Research Tabs for Member 1 Geo-Spatial DSS (No Emojis)
DASHBOARD_TABS = [
    "Overview",
    "Market Saturation",
    "Price Dynamics",
    "Agro-Meteorology",
    "Recommendations",
    "Data Reliability",
]

# Dashboard Primary Display Titles
APP_TITLE = RESEARCH_TOPIC
APP_SUBTITLE = "Sri Lankan Vegetable Market Intelligence"

# 25 Official Administrative Districts of Sri Lanka with High-Precision Centroids
DISTRICT_COORDS: Dict[str, Dict[str, Any]] = {
    "Ampara": {"lat": 7.2912, "lon": 81.6724, "province": "Eastern"},
    "Anuradhapura": {"lat": 8.3114, "lon": 80.4037, "province": "North Central"},
    "Badulla": {"lat": 6.9934, "lon": 81.0550, "province": "Uva"},
    "Batticaloa": {"lat": 7.7310, "lon": 81.6747, "province": "Eastern"},
    "Colombo": {"lat": 6.9271, "lon": 79.8612, "province": "Western"},
    "Galle": {"lat": 6.0535, "lon": 80.2210, "province": "Southern"},
    "Gampaha": {"lat": 7.0840, "lon": 79.9939, "province": "Western"},
    "Hambantota": {"lat": 6.1429, "lon": 81.1212, "province": "Southern"},
    "Jaffna": {"lat": 9.6615, "lon": 80.0255, "province": "Northern"},
    "Kalutara": {"lat": 6.5854, "lon": 79.9607, "province": "Western"},
    "Kandy": {"lat": 7.2906, "lon": 80.6337, "province": "Central"},
    "Kegalle": {"lat": 7.2513, "lon": 80.3464, "province": "Sabaragamuwa"},
    "Kilinochchi": {"lat": 9.3803, "lon": 80.3770, "province": "Northern"},
    "Kurunegala": {"lat": 7.4863, "lon": 80.3623, "province": "North Western"},
    "Mannar": {"lat": 8.9810, "lon": 79.9044, "province": "Northern"},
    "Matale": {"lat": 7.4675, "lon": 80.6234, "province": "Central"},
    "Matara": {"lat": 5.9549, "lon": 80.5550, "province": "Southern"},
    "Monaragala": {"lat": 6.8728, "lon": 81.3507, "province": "Uva"},
    "Mullaitivu": {"lat": 9.2671, "lon": 80.8142, "province": "Northern"},
    "Nuwara Eliya": {"lat": 6.9497, "lon": 80.7891, "province": "Central"},
    "Polonnaruwa": {"lat": 7.9403, "lon": 81.0188, "province": "North Central"},
    "Puttalam": {"lat": 8.0362, "lon": 79.8283, "province": "North Western"},
    "Ratnapura": {"lat": 6.6828, "lon": 80.4034, "province": "Sabaragamuwa"},
    "Trincomalee": {"lat": 8.5874, "lon": 81.2152, "province": "Eastern"},
    "Vavuniya": {"lat": 8.7514, "lon": 80.4971, "province": "Northern"},
}

DISTRICT_NAMES: List[str] = sorted(list(DISTRICT_COORDS.keys()))

# The Five Official Research Target Crops
TARGET_CROPS: List[str] = [
    "Carrot",
    "Leek",
    "Cabbage",
    "Tomato",
    "Big Onion (Local)"
]

# Curated Visual Palette for Crop Visualizations
CROP_COLORS: Dict[str, str] = {
    "Carrot": "#ea580c",            # Vibrant Deep Orange
    "Leek": "#16a34a",              # Lush Green
    "Cabbage": "#65a30d",           # Light Lime Green
    "Tomato": "#dc2626",            # Bold Crimson Red
    "Big Onion (Local)": "#9333ea",  # Rich Purple
}

# Baseline Production and Market Capacity Estimates (Hectares National Base)
NATIONAL_BASE_CAPACITY: Dict[str, float] = {
    "Carrot": 2300.0,
    "Leek": 1350.0,
    "Cabbage": 2200.0,
    "Tomato": 2800.0,
    "Big Onion (Local)": 4000.0,
}

# Baseline Farmgate Reference Price (Rs./kg)
BASE_REFERENCE_PRICES: Dict[str, float] = {
    "Carrot": 120.0,
    "Leek": 150.0,
    "Cabbage": 70.0,
    "Tomato": 90.0,
    "Big Onion (Local)": 180.0,
}

# Theme Color Tokens (Dark Forest-Green & Charcoal Architecture)
COLOR_BG_DARK = "#0A140D"          # Main dark dashboard background
COLOR_SURFACE_DARK = "#0F1E14"     # Dark forest surface
COLOR_SIDEBAR_DARK = "#0F1E14"     # Sidebar dark surface
COLOR_INPUT_DARK = "#172D1E"       # Input field dark background
COLOR_BORDER_DARK = "#1E3A26"      # Muted green border
COLOR_FOREST_GREEN = "#1E5631"     # Forest green brand token
COLOR_SAGE_GREEN = "#4A7C59"       # Sage green accent token
COLOR_ACCENT_MINT = "#4ADE80"      # Active indicator / bright mint green
COLOR_TEXT_WHITE = "#FFFFFF"       # Headings on dark background
COLOR_TEXT_LIGHT_GREY = "#C8D8CC"  # Secondary text on dark background
COLOR_TEXT_MUTED = "#A3B899"       # Muted / caption text on dark background

# Card Tokens (Light Surfaces on Dark Dashboard)
COLOR_CARD_SURFACE = "#FFFFFF"     # Executive summary card background
COLOR_CARD_TEXT = "#0E1C11"        # Dark charcoal card value text
COLOR_CARD_TITLE = "#2D4A33"       # Dark green card label text
COLOR_CARD_SUB = "#3D5743"         # Dark muted card subtext
COLOR_CARD_BORDER = "#C8DEC9"      # Crisp card border

# Standardized Risk Bands (0–100 Scale)
# Low: 0–20, Moderate: 20–60, High: 60–80, Critical: 80–100
RISK_BAND_LOW_MAX = 20.0
RISK_BAND_MODERATE_MAX = 60.0
RISK_BAND_HIGH_MAX = 80.0
RISK_BAND_CRITICAL_MAX = 100.0

# Backward compatibility threshold aliases
MSRS_SAFE_MAX = 20.0
MSRS_WARNING_MAX = 60.0

# Consistent Risk & Status Palette
COLOR_SAFE_GREEN = "#2E7D32"       # Low Risk (0–20) / High Viability (Safe)
COLOR_WARNING_AMBER = "#D97706"    # Moderate Risk (20–60) (Warning)
COLOR_HIGH_ORANGE = "#EA580C"      # High Risk (60–80)
COLOR_CRITICAL_RED = "#C53030"     # Critical Risk (80–100) / Glut Crash

# Configurable DRI Component Weights (Sum = 1.0)
DRI_WEIGHT_COMPLETENESS = 0.4
DRI_WEIGHT_RECENCY = 0.3
DRI_WEIGHT_CONSISTENCY = 0.3

# Plain-Language Confidence Thresholds
CONFIDENCE_HIGH_MIN = 80.0
CONFIDENCE_MODERATE_MIN = 60.0


def get_risk_band(score: float) -> Tuple[str, str, str]:
    """
    Evaluate standardized 0–100 risk score into unambiguous risk band.
    Bands:
      Low: 0.0 <= score <= 20.0
      Moderate: 20.0 < score <= 60.0
      High: 60.0 < score <= 80.0
      Critical: 80.0 < score <= 100.0
    Returns:
      (band_key, band_label, hex_color)
    """
    s = min(max(float(score), 0.0), 100.0)
    if s <= RISK_BAND_LOW_MAX:
        return "LOW", "Low Risk", COLOR_SAFE_GREEN
    elif s <= RISK_BAND_MODERATE_MAX:
        return "MODERATE", "Moderate Risk", COLOR_WARNING_AMBER
    elif s <= RISK_BAND_HIGH_MAX:
        return "HIGH", "High Risk", COLOR_HIGH_ORANGE
    else:
        return "CRITICAL", "Critical Risk", COLOR_CRITICAL_RED


def get_plain_language_confidence(score: float) -> Tuple[str, str]:
    """
    Standard plain-language confidence based on DRI score (0-100):
      80 or above: High confidence
      60 to below 80: Moderate confidence
      below 60: Low confidence — verify with local officer
    Returns:
      (plain_text_label, hex_color)
    """
    s = min(max(float(score), 0.0), 100.0)
    if s >= CONFIDENCE_HIGH_MIN:
        return "High confidence", COLOR_SAFE_GREEN
    elif s >= CONFIDENCE_MODERATE_MIN:
        return "Moderate confidence", COLOR_WARNING_AMBER
    else:
        return "Low confidence — verify with local officer", COLOR_CRITICAL_RED


# Custom CSS for Dark Forest-Green / Charcoal Professional Agriculture Theme
CUSTOM_CSS = """
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Dark Theme Base */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0A140D !important;
        color: #E8EFEA !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(10, 20, 13, 0.96) !important;
        height: 2.75rem !important;
        z-index: 100 !important;
    }

    /* Main Container Padding - Optimized for 1366x768 and 1920x1080 Laptops */
    .block-container {
        padding-top: 3.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1380px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Typography on Dark Background */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    p, span, div {
        color: #E8EFEA;
    }
    .stMarkdown p {
        color: #D1DDD4 !important;
        font-size: 0.90rem !important;
        line-height: 1.5;
    }
    .stCaption, small {
        color: #A3B899 !important;
        font-size: 0.80rem !important;
    }

    /* Header & Subtitle - High specificity to guarantee white title on dark background */
    .dashboard-header,
    h1.dashboard-header,
    [data-testid="stMarkdownContainer"] .dashboard-header,
    [data-testid="stMarkdownContainer"] h1.dashboard-header {
        font-size: 1.48rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin-top: 0.10rem !important;
        margin-bottom: 0.35rem !important;
        padding: 0 !important;
        letter-spacing: -0.015em !important;
        line-height: 1.30 !important;
    }
    .dashboard-header-sub {
        font-weight: 700 !important;
        color: #E2ECE5 !important;
        font-size: 0.95em !important;
    }
    .dashboard-caption {
        font-size: 0.84rem !important;
        color: #C8D8CC !important;
        margin-bottom: 0.85rem !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        gap: 0.50rem !important;
    }

    /* Suppress Streamlit default cache-miss function execution spinners (e.g. 'Running func(...)') */
    [data-testid="stSpinner"]:has(code),
    .stSpinner:has(code) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* =========================================================================
       SIDEBAR STYLING (Dark Forest Green with Crisp Light Labels)
       ========================================================================= */
    section[data-testid="stSidebar"] {
        background-color: #0F1E14 !important;
        border-right: 1px solid #1E3A26 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-size: 1.05rem !important;
    }
    section[data-testid="stSidebar"] label {
        color: #E0EAE2 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] p {
        color: #C8D8CC !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #A3B899 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1E3A26 !important;
        margin: 0.6rem 0 !important;
    }

    /* Sidebar Project Branding Card (Clean Research Identity with Forest-Green Accent) */
    .sidebar-brand-box {
        background: #14281B !important;
        border: 1px solid #244B30 !important;
        border-left: 4px solid #4ADE80 !important;
        border-radius: 8px !important;
        padding: 0.85rem 0.95rem !important;
        margin-bottom: 0.90rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25) !important;
    }
    .sidebar-brand-title {
        color: #FFFFFF !important;
        font-size: 1.02rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em !important;
        line-height: 1.25 !important;
        margin-bottom: 0.25rem !important;
    }
    .sidebar-brand-subtitle {
        color: #4ADE80 !important;
        font-size: 0.84rem !important;
        font-style: italic !important;
        font-weight: 500 !important;
        line-height: 1.35 !important;
        margin-bottom: 0.45rem !important;
    }
    .sidebar-brand-desc {
        color: #C8D8CC !important;
        font-size: 0.76rem !important;
        line-height: 1.45 !important;
        margin-bottom: 0 !important;
    }

    /* Input controls in sidebar */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #172D1E !important;
        border: 1px solid #2B5237 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="radio"] label {
        color: #E0EAE2 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {
        fill: #A3B899 !important;
    }

    /* Selectbox Dropdown Options */
    [data-baseweb="popover"] ul {
        background-color: #132619 !important;
        border: 1px solid #2B5237 !important;
    }
    [data-baseweb="popover"] li {
        color: #E8EFEA !important;
        background-color: transparent !important;
        font-size: 0.88rem !important;
    }
    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] li[aria-selected="true"] {
        background-color: #1E3D27 !important;
        color: #4ADE80 !important;
    }

    /* =========================================================================
       TABS THEMING (Optimized for Laptop Viewports)
       ========================================================================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        border-bottom: 2px solid #1E3A26 !important;
        padding-bottom: 0px !important;
        background-color: transparent !important;
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        scrollbar-width: thin;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        white-space: nowrap !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #A3B899 !important;
        border-radius: 6px 6px 0px 0px !important;
        padding: 0 10px !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
        background-color: #132619 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4ADE80 !important;
        border-bottom: 3px solid #4ADE80 !important;
        background-color: #132619 !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4ADE80 !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: #1E3A26 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.25rem !important;
        padding-bottom: 1.5rem !important;
        box-sizing: border-box !important;
        overflow: visible !important;
    }

    /* =========================================================================
       EXECUTIVE KPI CARDS (Dark Forest-Green Theme with High Contrast)
       ========================================================================= */
    .metric-card {
        background: #112417 !important;
        border: 1px solid #1E3D27 !important;
        border-radius: 8px !important;
        padding: 0.75rem 0.95rem !important;
        border-left: 5px solid #1E5631 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35) !important;
        margin-bottom: 0.6rem !important;
        transition: transform 0.15s ease-in-out;
        min-height: 118px !important;
        height: 118px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.45) !important;
        border-color: #2E5A38 !important;
    }
    .metric-title {
        font-size: 0.73rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: #A3B899 !important;
        font-weight: 700 !important;
        margin-bottom: 0.15rem !important;
        line-height: 1.25 !important;
    }
    .metric-value {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        line-height: 1.15 !important;
    }
    .metric-sub {
        font-size: 0.76rem !important;
        color: #C8D8CC !important;
        margin-top: 0.15rem !important;
        font-weight: 500 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Provenance Badges */
    .badge-real {
        background-color: #EAF7EC !important;
        color: #1E5631 !important;
        padding: 0.2rem 0.55rem !important;
        border-radius: 9999px !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        display: inline-block !important;
        border: 1px solid #B8DCBD !important;
    }
    .badge-proxy {
        background-color: #EBF3F8 !important;
        color: #1C5373 !important;
        padding: 0.2rem 0.55rem !important;
        border-radius: 9999px !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        display: inline-block !important;
        border: 1px solid #BEDAE8 !important;
    }
    .badge-demo {
        background-color: #FEF7EA !important;
        color: #8C530A !important;
        padding: 0.2rem 0.55rem !important;
        border-radius: 9999px !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        display: inline-block !important;
        border: 1px solid #F5DAB3 !important;
    }

    /* Filter Context Confirmation Banner */
    .filter-context-banner {
        background: #112316 !important;
        border: 1px solid #1E3D27 !important;
        border-radius: 6px !important;
        padding: 0.60rem 0.90rem !important;
        margin-bottom: 0.85rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        font-size: 0.88rem !important;
        color: #E8EFEA !important;
        font-weight: 500 !important;
    }
    .filter-context-banner strong {
        color: #FFFFFF !important;
    }

    /* Advisory Banners on Dark Background */
    .banner-safe {
        background: #0E2916 !important;
        border: 1px solid #1A4A28 !important;
        border-left: 5px solid #2E7D32 !important;
        padding: 0.85rem 1.05rem !important;
        border-radius: 6px !important;
        margin: 0.6rem 0 !important;
        color: #E8F5EB !important;
    }
    .banner-warning {
        background: #2A1E08 !important;
        border: 1px solid #543C12 !important;
        border-left: 5px solid #D97706 !important;
        padding: 0.85rem 1.05rem !important;
        border-radius: 6px !important;
        margin: 0.6rem 0 !important;
        color: #FEF3C7 !important;
    }
    .banner-critical {
        background: #2B0F11 !important;
        border: 1px solid #5A1C20 !important;
        border-left: 5px solid #C53030 !important;
        padding: 0.85rem 1.05rem !important;
        border-radius: 6px !important;
        margin: 0.6rem 0 !important;
        color: #FEE2E2 !important;
    }

    /* Expanders & Dataframe */
    .streamlit-expanderHeader {
        background-color: #132619 !important;
        color: #E8EFEA !important;
        border: 1px solid #1E3A26 !important;
        border-radius: 6px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background-color: #0F1E14 !important;
        border: 1px solid #1E3A26 !important;
        border-top: none !important;
    }
    .stAlert {
        background-color: #132619 !important;
        color: #E8EFEA !important;
        border: 1px solid #1E3A26 !important;
    }

    /* Dashboard Research Collaboration Footer */
    .dashboard-footer {
        border-top: 1px solid #1E3A26 !important;
        padding: 1.5rem 0.5rem !important;
        margin-top: 3rem !important;
        margin-bottom: 1.5rem !important;
        text-align: center !important;
        color: #A3B899 !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        background-color: transparent !important;
    }
    /* Plotly Modebar & Chart Header Positioning */
    .stPlotlyChart .modebar-container {
        top: 2px !important;
        right: 4px !important;
    }
    .stPlotlyChart .modebar-btn {
        padding: 3px !important;
    }

    /* Suppress Plotly snapshot toast notifications completely */
    .notifier-note,
    .notifier-close,
    .notifier,
    div[class*="notifier"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
</style>
"""

# Plotly Modebar Configurations
# For maps: High-resolution PNG export with CORS-friendly tiles, removing lasso/select and suppressing notifiers
MAP_PLOTLY_CONFIG: Dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "displayNotifier": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "geospatial_sri_lanka_map",
        "height": 650,
        "width": 750,
        "scale": 2,
    },
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "scrollZoom": True,
}

# For standard time-series & scatter charts: Clean modebar with reliable PNG export and no notifiers
CHART_PLOTLY_CONFIG: Dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "displayNotifier": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "chart_export",
        "height": 500,
        "width": 800,
        "scale": 2,
    },
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "scrollZoom": False,
}

# For categorical bar charts: Keep toImage & resetScale, remove 2D pan/zoom & notifiers
BAR_CHART_PLOTLY_CONFIG: Dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "displayNotifier": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "quota_commitment_barchart",
        "height": 500,
        "width": 800,
        "scale": 2,
    },
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
    "scrollZoom": False,
}
