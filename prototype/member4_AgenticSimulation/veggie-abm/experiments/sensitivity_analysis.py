"""
Sensitivity analysis: which model parameters most affect crash
frequency/volatility? Answering this gives you a defensible basis
for your chosen adoption-rate thresholds and dampening strength in
the report, rather than presenting them as arbitrary picks.

Uses SALib's Sobol method — needs N * (2D + 2) model runs where D
is the number of parameters, so this gets expensive fast. Start
small (N=64) to confirm it runs, then scale up for final results.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from model.cobweb_model import CobwebModel

DISTRICTS = ["Nuwara Eliya", "Badulla", "Kandy", "Kurunegala", "Matale"]
CROPS = ["Carrot"]
BASE_PRICE = {"Carrot": 120.0}
BASE_EXTENT = {"Carrot": 2300.0}
N_SEASONS = 20

PROBLEM = {
    "num_vars": 3,
    "names": ["price_elasticity", "adoption_rate", "dampening_strength"],
    "bounds": [[0.2, 1.0], [0.0, 1.0], [0.2, 1.0]],
}


def run_model_for_params(price_elasticity: float, adoption_rate: float, dampening_strength: float, seed: int) -> float:
    """dampening_strength isn't a CobwebModel constructor param yet —
    this is a hook: wire it into FarmerAgent._decide_with_msrs()'s
    dampening formula (currently hardcoded at 0.8) before running
    this for real. Left as a clear TODO rather than silently
    ignoring the parameter."""
    model = CobwebModel(
        n_farmers_per_district_crop=15,
        districts=DISTRICTS, crops=CROPS,
        base_price=BASE_PRICE, base_extent=BASE_EXTENT,
        msrs_enabled=adoption_rate > 0, adoption_rate=adoption_rate,
        price_elasticity=price_elasticity,
        dampening_strength=dampening_strength,
        rng=seed,
    )

    for _ in range(N_SEASONS):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    return df["price_Carrot"].pct_change().std()


def run_sensitivity_analysis(n_samples: int = 64):
    param_values = sobol_sample.sample(PROBLEM, n_samples)
    outputs = np.array([
        run_model_for_params(*params, seed=i)
        for i, params in enumerate(param_values)
    ])
    sobol_indices = sobol_analyze.analyze(PROBLEM, outputs)
    print("First-order sensitivity indices (S1) — higher = more influence on volatility:")
    for name, s1 in zip(PROBLEM["names"], sobol_indices["S1"]):
        print(f"  {name}: {s1:.3f}")
    return sobol_indices


if __name__ == "__main__":
    run_sensitivity_analysis(n_samples=64)
