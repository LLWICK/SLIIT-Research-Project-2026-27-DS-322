"""Configuration and constant registry for Member 1 Data Engineering Pipeline.

Defines target crops, verified district coordinates, market centers,
API/URL endpoints, and file path conventions.
"""

from pathlib import Path

# Base Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"

# Domain-specific raw and processed directories
RAW_WEATHER_DIR = RAW_DATA_DIR / "weather"
PROCESSED_WEATHER_DIR = PROCESSED_DATA_DIR / "weather"

RAW_PRICE_DIR = RAW_DATA_DIR / "prices"
PROCESSED_PRICE_DIR = PROCESSED_DATA_DIR / "prices"

RAW_CBSL_DIR = RAW_PRICE_DIR / "cbsl"
RAW_HARTI_DIR = RAW_PRICE_DIR / "harti"

# 5 Target Vegetable Crops defined in research scope
TARGET_CROPS = [
    "Carrot",
    "Cabbage",
    "Leek",
    "Big onion",
    "Big onion (Imported)",
    "Tomato",
]

# Standardized crop aliases mapping across heterogeneous reporting formats
CROP_NAME_MAPPINGS = {
    # Carrot
    "carrot": "Carrot",
    "carrots": "Carrot",
    # Cabbage
    "cabbage": "Cabbage",
    "cabbages": "Cabbage",
    # Leek
    "leek": "Leek",
    "leeks": "Leek",
    # Big onion
    "big onion": "Big onion",
    "big onions": "Big onion",
    "big onion (local)": "Big onion",
    "big onion (imp)": "Big onion (Imported)",
    "big onion (imported)": "Big onion (Imported)",
    "onion (big)": "Big onion",
    # Tomato
    "tomato": "Tomato",
    "tomatoes": "Tomato",
}

# Verified Market Centers in Sri Lankan Vegetable Supply Chains
# Preserves market-level identities (never collapses them into generic district averages)
MARKET_CENTERS = {
    "Pettah": {
        "market_name": "Pettah",
        "market_type": "Terminal Wholesale & Retail Market",
        "district": "Colombo",
        "latitude": 6.9365,
        "longitude": 79.8528,
    },
    "Dambulla": {
        "market_name": "Dambulla",
        "market_type": "Dedicated Economic Centre",
        "district": "Matale",
        "latitude": 7.8606,
        "longitude": 80.6517,
    },
    "Kandy": {
        "market_name": "Kandy",
        "market_type": "Central Municipal Market",
        "district": "Kandy",
        "latitude": 7.2906,
        "longitude": 80.6337,
    },
    "Keppetipola": {
        "market_name": "Keppetipola",
        "market_type": "Dedicated Economic Centre",
        "district": "Badulla",
        "latitude": 6.8911,
        "longitude": 80.8522,
    },
    "Narahenpita": {
        "market_name": "Narahenpita",
        "market_type": "Dedicated Economic Centre (Retail)",
        "district": "Colombo",
        "latitude": 6.8942,
        "longitude": 79.8821,
    },
    "Peliyagoda": {
        "market_name": "Peliyagoda",
        "market_type": "Special Economic Centre",
        "district": "Gampaha",
        "latitude": 6.9644,
        "longitude": 79.8972,
    },
    "Meegoda": {
        "market_name": "Meegoda",
        "market_type": "Dedicated Economic Centre",
        "district": "Colombo",
        "latitude": 6.8375,
        "longitude": 80.0542,
    },
    "Norochcholai": {
        "market_name": "Norochcholai",
        "market_type": "Dedicated Economic Centre",
        "district": "Puttalam",
        "latitude": 8.1633,
        "longitude": 79.7428,
    },
}

# Official Price Sources Endpoints & URL Patterns
CBSL_CONFIG = {
    "agency_name": "CBSL",
    "portal_url": "https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report",
    # Format: price_report_YYYYMMDD_e.pdf
    "pdf_url_template": "https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/statistics/pricerpt/price_report_{date}_e.pdf",
}

HARTI_CONFIG = {
    "agency_name": "HARTI",
    "portal_url": "https://www.harti.gov.lk/daily-price.php",
    # Format: assets/pdf/food_price/daily/eng/{year}/{month}/daily_{dd-mm-yyyy}.pdf
    "pdf_url_template": "https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/{year}/{month}/daily_{date}.pdf",
}

# Verified District Coordinates (latitude, longitude) for all 25 Administrative Districts
DISTRICT_COORDINATES: dict[str, tuple[float, float]] = {
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

# Sri Lanka Cultivation Seasons
# Maha: October to March (season year assigned to the October start year)
# Yala: April to September (same calendar year)
YALA_MONTHS = (4, 5, 6, 7, 8, 9)
MAHA_MONTHS = (10, 11, 12, 1, 2, 3)
