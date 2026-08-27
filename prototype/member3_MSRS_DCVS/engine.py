def calculate_msrs(committed_area: float, market_demand_capacity: float) -> float:
    """
    Calculate the Market Saturation Risk Score (MSRS) (0.0 to 1.0).
    MSRS quantifies the risk of oversupply based on real-time farmer commitments.
    
    If the committed area exceeds the estimated market demand capacity, the MSRS 
    approaches 1.0. A score above 0.7 usually indicates a high risk of price crash 
    (the Cobweb effect).
    """
    if market_demand_capacity <= 0:
        return 1.0
        
    overshoot_ratio = committed_area / market_demand_capacity
    
    # Safe zone (e.g. <= 80% capacity) -> 0 MSRS
    if overshoot_ratio <= 0.8:
        return 0.0
    
    # Critical zone (e.g. >= 1.2 capacity) -> 1.0 MSRS
    if overshoot_ratio >= 1.2:
        return 1.0
        
    # Linear scale between 80% and 120% capacity
    msrs = (overshoot_ratio - 0.8) / (1.2 - 0.8)
    return round(min(max(msrs, 0.0), 1.0), 3)

def calculate_dcvs(msrs: float, weather_index: float, price_forecast_index: float) -> float:
    """
    Calculate the Dynamic Crop Viability Score (DCVS) (0 to 100%).
    DCVS is a holistic recommendation score that proactively adjusts crop recommendations.
    
    Weights (example):
    - 40% weight on Weather Suitability Index (0.0 to 1.0)
    - 30% weight on Price Forecast Index (0.0 to 1.0)
    - 30% weight on Market Saturation Risk Score (inverted, so lower MSRS is better)
    """
    weather_weight = 0.4
    price_weight = 0.3
    saturation_weight = 0.3
    
    # Invert MSRS since high risk means low viability
    saturation_score = 1.0 - msrs
    
    dcvs_raw = (weather_index * weather_weight) + \
               (price_forecast_index * price_weight) + \
               (saturation_score * saturation_weight)
               
    return round(dcvs_raw * 100, 1)

def check_crash_risk(msrs: float) -> str:
    """Return a qualitative risk assessment."""
    if msrs >= 0.7:
        return "CRITICAL: High risk of Cobweb price crash. Oversupply imminent."
    elif msrs >= 0.3:
        return "WARNING: Moderate crowding detected. Reconsider planting."
    else:
        return "SAFE: Healthy market demand."
