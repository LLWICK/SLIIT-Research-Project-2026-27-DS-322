"""Check a persistent, shocked baseline against Colombo wholesale seasons.

Farmers do not see an advisory in this check. The persistence and the
shock scale are estimated from the same HARTI seasons, then many runs
are compared with the observed crash rate and the season-to-season lag.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pandas as pd

from simulation.calibration.compare import RATIO_GRID, series_moments
from simulation.inputs.harti import colombo_wholesale_seasons
from simulation.config.districts import ADVISORY_CROPS
from simulation.config.settings import BaselineSettings, default_settings
from simulation.market.price_process import PriceProcess, combine_price, fit_price_process
from simulation.metrics.crash import season_is_crash
from simulation.scenarios.advisory import run_advisory

N_REPLICATIONS = 40
# The HARTI comparison put every crop at the bottom of the slope grid.
# A larger ratio flips the season-to-season lag. This check keeps that floor.
SLOPE_RATIO = RATIO_GRID[0]


def fit_observed(settings: BaselineSettings | None = None) -> tuple[BaselineSettings, dict[str, PriceProcess], pd.DataFrame]:
    """Center each crop on its own season-mean price and fit the AR(1)."""

    settings = default_settings() if settings is None else settings
    observed = colombo_wholesale_seasons()
    processes: dict[str, PriceProcess] = {}
    centered = []
    for crop in settings.crops:
        rows = observed[observed["crop_code"] == crop.crop_code]
        if rows.empty:
            raise ValueError(f"no Colombo wholesale seasons for {crop.crop_code}")
        process = fit_price_process(rows)
        processes[crop.crop_code] = process
        centered.append(replace(crop, reference_price=process.center))
    return replace(settings, crops=tuple(centered)), processes, observed


def _lag(prices: pd.Series) -> float | None:
    if len(prices) < 5 or float(prices.std(ddof=1)) == 0:
        return None
    value = float(prices.corr(prices.shift(1)))
    if value != value:
        return None
    return value


def _one_rate(prices: list[float], crashes: int) -> dict[str, float | None]:
    changes = len(prices) - 1
    return {
        "crash_rate": 0.0 if changes < 1 else crashes / changes,
        "lag": _lag(pd.Series(prices, dtype=float)),
    }


def simulate_baseline(
    settings: BaselineSettings,
    processes: dict[str, PriceProcess],
    n_seasons: int,
    replications: int = N_REPLICATIONS,
) -> pd.DataFrame:
    rows = []
    advisory = {crop: process for crop, process in processes.items() if crop in ADVISORY_CROPS}
    persistence = {crop: process.persistence for crop, process in advisory.items()}
    shock_std = {crop: process.shock_std for crop, process in advisory.items()}
    for seed in range(replications):
        frame = run_advisory(
            adoption_rate=0.0,
            seed=seed,
            n_seasons=n_seasons,
            slope_ratio=SLOPE_RATIO,
            settings=settings,
            persistence=persistence,
            shock_std=shock_std,
            shock_seed=seed,
        )
        for crop_code, crop_rows in frame.groupby("crop_code", sort=True):
            ordered = crop_rows.sort_values("season_index")
            prices = [float(price) for price in ordered["price"]]
            rows.append(
                {
                    "seed": seed,
                    "crop_code": str(crop_code),
                    **_one_rate(prices, int(ordered["is_crash"].sum())),
                }
            )
    drop_fraction = settings.crash_rule.drop_fraction
    for crop_code, process in processes.items():
        if crop_code in ADVISORY_CROPS:
            continue
        for seed in range(replications):
            rng = random.Random(seed)
            prices: list[float] = []
            previous: float | None = None
            crashes = 0
            for _ in range(n_seasons):
                shock = rng.gauss(0.0, process.shock_std)
                if previous is None:
                    price = max(0.0, process.center + shock)
                else:
                    price = combine_price(previous, process.center, process.persistence, shock)
                    if season_is_crash(price, previous, drop_fraction):
                        crashes += 1
                prices.append(price)
                previous = price
            rows.append({"seed": seed, "crop_code": crop_code, **_one_rate(prices, crashes)})
    return pd.DataFrame(rows)


def _number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def format_check(
    simulated: pd.DataFrame,
    processes: dict[str, PriceProcess],
    observed: pd.DataFrame,
    drop_fraction: float,
) -> str:
    lines = [
        "Member 3 score: 0 at 80% of the district target, 1 at 120%.",
        "Demo capacities in member3.config are not used.",
        "No advisory. Persistence and shock scale are fitted to these same seasons.",
        f"Supply response uses slope ratio {SLOPE_RATIO:.2f}, the smallest ratio on the HARTI grid.",
        "",
    ]
    for crop_code, process in processes.items():
        rows = observed[observed["crop_code"] == crop_code].copy()
        moments = series_moments(rows, drop_fraction)
        observed_rate = 0.0 if moments.n_pairs == 0 else moments.crashes / moments.n_pairs
        crop = simulated[simulated["crop_code"] == crop_code]
        scope = "district planting" if crop_code in ADVISORY_CROPS else "price only, no district table"
        lines.append(
            f"{crop_code} ({scope}): persistence {process.persistence:.2f}, shock std {process.shock_std:.1f} Rs/kg, center {process.center:.1f}"
        )
        lines.append(
            "  HARTI crash rate "
            f"{observed_rate:.2f}, lag {_number(moments.lag_corr)}, deseasoned lag {_number(moments.lag_corr_deseasoned)}"
        )
        lines.append(
            f"  simulated crash rate {crop['crash_rate'].mean():.2f}, lag {crop['lag'].mean():.2f}"
            f"  ({len(crop)} runs)"
        )
    return "\n".join(lines)


def main() -> None:
    settings, processes, observed = fit_observed()
    n_seasons = int(observed[observed["crop_code"] == "carrot"]["ordinal"].nunique())
    simulated = simulate_baseline(settings, processes, n_seasons=n_seasons)
    print(format_check(simulated, processes, observed, settings.crash_rule.drop_fraction))


if __name__ == "__main__":
    main()
