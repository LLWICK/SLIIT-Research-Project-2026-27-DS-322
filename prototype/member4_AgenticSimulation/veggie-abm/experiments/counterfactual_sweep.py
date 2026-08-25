"""
The actual experiment your TAF asks for (section 9, Wickramaarachchi's
sub-objective): does giving farmers access to MSRS reduce price-crash
frequency, and at what adoption rate does that effect kick in?

Runs BOTH conditions (baseline vs MSRS-informed) across several
adoption rates, with multiple random-seed replications per rate
(ABMs are stochastic — one run per setting proves nothing).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from model.cobweb_model import CobwebModel

DISTRICTS = ["Nuwara Eliya", "Badulla", "Kandy", "Kurunegala", "Matale"]
CROPS = ["Carrot", "Leek", "Tomato", "Cabbage", "Big Onion (Local)"]
BASE_PRICE = {"Carrot": 120.0, "Leek": 150.0, "Tomato": 90.0, "Cabbage": 70.0, "Big Onion (Local)": 180.0}
BASE_EXTENT = {"Carrot": 2300.0, "Leek": 1350.0, "Tomato": 2800.0, "Cabbage": 2200.0, "Big Onion (Local)": 4000.0}

N_SEASONS = 24            # 12 years, matches your real data span
N_FARMERS_PER_CELL = 15
ADOPTION_RATES = [0.0, 0.1, 0.3, 0.5, 0.7]
N_REPLICATIONS = 10       # random seeds per adoption rate — increase for your final results


def crash_frequency(price_series: pd.Series, threshold: float = -0.25) -> float:
    """A 'crash' = season-over-season price drop worse than threshold
    (default -25%). Returns fraction of seasons that were crashes."""
    pct_change = price_series.pct_change().dropna()
    return (pct_change < threshold).mean()


def run_single(adoption_rate: float, seed: int) -> dict:
    msrs_enabled = adoption_rate > 0.0
    model = CobwebModel(
        n_farmers_per_district_crop=N_FARMERS_PER_CELL,
        districts=DISTRICTS, crops=CROPS,
        base_price=BASE_PRICE, base_extent=BASE_EXTENT,
        msrs_enabled=msrs_enabled, adoption_rate=adoption_rate,
        rng=seed,
    )
    for _ in range(N_SEASONS):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()

    result = {"adoption_rate": adoption_rate, "seed": seed}
    for crop in CROPS:
        prices = df[f"price_{crop}"]
        result[f"crash_freq_{crop}"] = crash_frequency(prices)
        result[f"price_volatility_{crop}"] = prices.pct_change().std()
        result[f"equilibrium_score_{crop}"] = df[f"equilibrium_{crop}"].mean()
        result[f"mean_profit_{crop}"] = df[f"mean_profit_{crop}"].mean()
        result[f"loss_rate_{crop}"] = df[f"loss_rate_{crop}"].mean()
    return result


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_full_sweep() -> pd.DataFrame:
    results = []
    for rate in ADOPTION_RATES:
        for seed in range(N_REPLICATIONS):
            print(f"[experiment] adoption_rate={rate} seed={seed}...")
            results.append(run_single(rate, seed))
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_full_sweep()
    df.to_csv(OUTPUT_DIR / "counterfactual_sweep_results.csv", index=False)

    # Compute aggregate cross-crop metrics
    crash_cols = [c for c in df.columns if c.startswith("crash_freq_")]
    vol_cols = [c for c in df.columns if c.startswith("price_volatility_")]
    eq_cols = [c for c in df.columns if c.startswith("equilibrium_score_")]
    profit_cols = [c for c in df.columns if c.startswith("mean_profit_")]
    loss_cols = [c for c in df.columns if c.startswith("loss_rate_")]

    df["mean_crash_freq"] = df[crash_cols].mean(axis=1)
    df["mean_price_volatility"] = df[vol_cols].mean(axis=1)
    df["mean_equilibrium_score"] = df[eq_cols].mean(axis=1)
    df["mean_farmer_profit"] = df[profit_cols].mean(axis=1)
    df["mean_loss_rate"] = df[loss_cols].mean(axis=1)

    summary = df.groupby("adoption_rate")[
        ["mean_crash_freq", "mean_price_volatility", "mean_equilibrium_score", "mean_farmer_profit", "mean_loss_rate"]
    ].agg(["mean", "std"])

    print("\n=== Comprehensive TAF Multi-Metric Summary by Adoption Rate ===")
    print(summary)
    summary.to_csv(OUTPUT_DIR / "counterfactual_summary.csv")

