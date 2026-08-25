"""
Dummy historical data for developing the ABM without waiting on the
real pipeline. Structured to LOOK like what you'll eventually get
from Thamel's data/exports/analysis_ready_price.csv, so swapping in
real data later is a drop-in replacement — same column names, same
shape — not a rewrite.

Simulates a crude but real cobweb pattern (price up -> more planting
next season -> oversupply -> price crash -> less planting -> price
recovers) so the baseline ABM actually has a real pattern to
reproduce, rather than pure noise.
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
CROPS = ["Carrot", "Leek", "Tomato", "Cabbage", "Big Onion (Local)"]
DISTRICTS = ["Nuwara Eliya", "Badulla", "Kandy", "Kurunegala", "Matale"]
N_SEASONS = 24  # 12 years of Maha+Yala


def generate_dummy_seasons(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for crop in CROPS:
        base_price = rng.uniform(80, 200)      # Rs./kg baseline
        base_extent = rng.uniform(1500, 3000)  # hectares, national
        price = base_price
        extent = base_extent

        for season_idx in range(N_SEASONS):
            season_type = "Yala" if season_idx % 2 == 0 else "Maha"
            year = 2014 + season_idx // 2

            # crude cobweb mechanic: extent this season reacts to
            # LAST season's price; price this season reacts inversely
            # to how much extent was planted (oversupply -> crash)
            price_signal = (price - base_price) / base_price
            extent = base_extent * (1 + 0.7 * price_signal) * rng.normal(1, 0.08)
            extent = max(extent, base_extent * 0.3)

            supply_pressure = (extent - base_extent) / base_extent
            price = base_price * (1 - 0.6 * supply_pressure) * rng.normal(1, 0.1)
            price = max(price, base_price * 0.2)

            for district in DISTRICTS:
                district_weight = rng.uniform(0.1, 0.3)
                rows.append({
                    "season_type": season_type,
                    "season_year": year,
                    "district": district,
                    "crop": crop,
                    "avg_retail_price": round(price * rng.normal(1, 0.05), 2),
                    "committed_hectares": round(extent * district_weight, 1),
                    "source": "dummy",
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dummy_seasons()
    df.to_csv("data/dummy_season_history.csv", index=False)
    print(f"Generated {len(df)} rows -> data/dummy_season_history.csv")
    print(df.head(10))
