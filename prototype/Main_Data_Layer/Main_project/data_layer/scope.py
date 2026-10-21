"""Crops and markets in the first version of the data layer."""

# HARTI sheet spellings that mean the same market.
MARKET_ALIASES = {
    "COLOMBO": "colombo",
    "DAMBULLA": "dambulla",
    "MEEGODA": "meegoda",
    "MEEGODA(DEC)": "meegoda",
    "N'ELIYA": "nuwara_eliya",
    "NUWARA ELIYA": "nuwara_eliya",
}

MARKETS = {
    "colombo": "Colombo",
    "meegoda": "Meegoda",
    "nuwara_eliya": "Nuwara Eliya",
    "dambulla": "Dambulla",
}

# Administrative district of the market town. This is not the supply catchment.
# Colombo and Meegoda use the national cultivation extent as their supply signal.
MARKET_DISTRICT = {
    "colombo": "Colombo",
    "meegoda": "Colombo",
    "nuwara_eliya": "Nuwara Eliya",
    "dambulla": "Matale",
}

CROP_ALIASES = {
    "CARROT": "carrot",
    "LEEKS": "leeks",
    "TOMATOES": "tomato",
}

CROPS = {
    "carrot": "Carrot",
    "leeks": "Leeks",
    "tomato": "Tomato",
}

WEEKLY_SHEETS = {
    "Retail - W": "retail",
    "Wholesale - W": "wholesale",
}

# Weather is taken where the crops are grown, not at the Colombo and Meegoda markets.
# Coordinates are the town points sent to NASA POWER. The API snaps each one to a grid cell.
WEATHER_POINTS = {
    "nuwara_eliya": ("Nuwara Eliya", "Nuwara Eliya", 6.97, 80.77),
    "badulla": ("Badulla", "Badulla", 6.99, 81.06),
    "dambulla": ("Dambulla", "Matale", 7.87, 80.65),
}
