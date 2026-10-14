"""Configuration and constant registry for Member 1 Data Engineering Pipeline.

Defines target crops, verified district coordinates, API endpoints,
and file path conventions.
"""

from pathlib import Path

# Base Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"

RAW_WEATHER_DIR = RAW_DATA_DIR / "weather"
PROCESSED_WEATHER_DIR = PROCESSED_DATA_DIR / "weather"

# 5 Target Vegetable Crops defined in research scope
TARGET_CROPS = [
    "Carrot",
    "Cabbage",
    "Leek",
    "Big onion",
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
    "big onion (imported)": "Big onion (Imported)",
    "onion (big)": "Big onion",
    # Tomato
    "tomato": "Tomato",
    "tomatoes": "Tomato",
}

# Verified District Coordinates (latitude, longitude) for all 25 Administrative Districts
# Coordinates reflect district agricultural/administrative centers
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

# Key vegetable production and commercialization hubs for initial priority extraction
PRIORITY_VEGETABLE_DISTRICTS = [
    "Nuwara Eliya",  # Up-country core: Carrot, Leek, Cabbage
    "Badulla",       # Uva/Welimada basin: Carrot, Cabbage, Tomato, Leek
    "Kandy",         # Mid-country hub and central consumer market
    "Matale",        # Dambulla Dedicated Economic Centre & Big Onion growing zone
    "Anuradhapura",  # Major dry-zone Big Onion zone (Galnewa, Ipalogama)
    "Colombo",       # Main terminal wholesale consuming market (Pettah)
]

# NASA POWER Daily Point API Specifications
NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_COMMUNITY = "AG"  # Agroclimatology
NASA_POWER_PARAMETERS = [
    "PRECTOTCORR",  # Precipitation Corrected (mm/day)
    "T2M",          # Temperature at 2 Meters (°C)
    "T2M_MAX",      # Maximum Temperature at 2 Meters (°C)
    "T2M_MIN",      # Minimum Temperature at 2 Meters (°C)
    "RH2M",         # Relative Humidity at 2 Meters (%)
    "WS2M",         # Wind Speed at 2 Meters (m/s)
]
NASA_POWER_FILL_VALUE = -999.0

# Parameter units and descriptions
NASA_PARAMETER_METADATA = {
    "PRECTOTCORR": {
        "standard_name": "precipitation_mm",
        "units": "mm/day",
        "description": "Precipitation Corrected",
    },
    "T2M": {
        "standard_name": "temp_mean_c",
        "units": "°C",
        "description": "Mean Temperature at 2 Meters",
    },
    "T2M_MAX": {
        "standard_name": "temp_max_c",
        "units": "°C",
        "description": "Maximum Temperature at 2 Meters",
    },
    "T2M_MIN": {
        "standard_name": "temp_min_c",
        "units": "°C",
        "description": "Minimum Temperature at 2 Meters",
    },
    "RH2M": {
        "standard_name": "relative_humidity_pct",
        "units": "%",
        "description": "Relative Humidity at 2 Meters",
    },
    "WS2M": {
        "standard_name": "wind_speed_ms",
        "units": "m/s",
        "description": "Wind Speed at 2 Meters",
    },
}

# Sri Lanka Cultivation Seasons
# Maha: October to March (season year assigned to the October start year)
# Yala: April to September (same calendar year)
YALA_MONTHS = (4, 5, 6, 7, 8, 9)
MAHA_MONTHS = (10, 11, 12, 1, 2, 3)
