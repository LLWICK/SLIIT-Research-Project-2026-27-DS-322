"""
Configuration, constants, district geographic coordinates, crop vocabularies,
and visual theme tokens for the Member 1 Geo-Spatial DSS Dashboard.
"""

from typing import Dict, List, Any

# Dashboard Metadata & Research Identity
RESEARCH_TOPIC = "Data-Driven Decision Support System to Mitigate the “Cobweb Effect” in Sri Lankan Vegetable Markets"
SIDEBAR_BRAND_TITLE = "Geo-Spatial Decision Support System"
SIDEBAR_BRAND_SUBTITLE = "Sri Lankan Vegetable Market Intelligence"
SIDEBAR_BRAND_DESC = "Data-driven insights to understand and mitigate the Cobweb Effect in Sri Lankan vegetable markets."
RESEARCH_MEMBER = "Member 1 — Data Engineering & Geo-Spatial Decision Support System"
SLIIT_PROJECT_ID = "SLIIT-Research-Project-2026-27-DS-322"  # Retained in config for internal reference only

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

# Restrained Status & Risk Palette
COLOR_SAFE_GREEN = "#2E7D32"       # Low saturation / High viability (Safe)
COLOR_WARNING_AMBER = "#D97706"    # Moderate saturation / Balanced (Warning)
COLOR_CRITICAL_RED = "#C53030"     # High saturation / Oversupply glut (Critical)

# Saturation Risk Thresholds
MSRS_SAFE_MAX = 0.30
MSRS_WARNING_MAX = 0.70

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
        padding-top: 2.4rem !important;
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

    /* =========================================================================
       EXECUTIVE KPI CARDS (Light Surfaces on Dark Background - Strictly Consistent)
       ========================================================================= */
    .metric-card {
        background: #FFFFFF !important;
        border: 1px solid #C8DEC9 !important;
        border-radius: 8px !important;
        padding: 0.75rem 0.95rem !important;
        border-left: 5px solid #1E5631 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
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
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.35) !important;
    }
    .metric-title {
        font-size: 0.73rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: #2D4A33 !important;
        font-weight: 700 !important;
        margin-bottom: 0.15rem !important;
        line-height: 1.25 !important;
    }
    .metric-value {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: #0E1C11 !important;
        line-height: 1.15 !important;
    }
    .metric-sub {
        font-size: 0.76rem !important;
        color: #3D5743 !important;
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
</style>
"""
