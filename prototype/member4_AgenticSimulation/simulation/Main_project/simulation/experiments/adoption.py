"""Adoption sweep using Member 3's pressure score.

The baseline crash rate is printed next to the HARTI Colombo wholesale
rate. A lower simulated rate is not evidence that the advisory prevents
historical crashes unless the baseline already produces those crashes.
"""

from __future__ import annotations

import pandas as pd

from simulation.calibration.compare import compare_with_harti
from simulation.config.districts import ADVISORY_CROPS
from simulation.config.settings import default_settings
from simulation.scenarios.advisory import run_advisory

ADOPTION_RATES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
N_SEASONS = 16
N_REPLICATIONS = 10


def crash_rate(frame: pd.DataFrame) -> float:
    changes = len(frame) - 1
    if changes < 1:
        return 0.0
    return float(frame["is_crash"].sum()) / changes


def price_volatility(frame: pd.DataFrame) -> float:
    changes = frame["price"].pct_change()
    changes = changes.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(changes) < 2:
        return 0.0
    return float(changes.std(ddof=1))


def sweep(
    n_seasons: int = N_SEASONS,
    replications: int = N_REPLICATIONS,
    n_farmers: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rate in ADOPTION_RATES:
        for offset in range(replications):
            frame = run_advisory(
                adoption_rate=rate,
                seed=offset,
                n_seasons=n_seasons,
                n_farmers=n_farmers,
            )
            for crop_code, crop_rows in frame.groupby("crop_code", sort=True):
                ordered = crop_rows.sort_values("season_index")
                rows.append(
                    {
                        "adoption_rate": rate,
                        "seed": offset,
                        "crop_code": str(crop_code),
                        "crash_rate": crash_rate(ordered),
                        "price_volatility": price_volatility(ordered),
                        "mean_area_ha": float(ordered["area_ha"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["adoption_rate", "crop_code"], sort=True)[["crash_rate", "price_volatility", "mean_area_ha"]]
        .mean()
        .reset_index()
    )


def harti_crash_rates() -> dict[str, float]:
    rates = {}
    for result in compare_with_harti():
        if result.crop_code not in ADVISORY_CROPS:
            continue
        pairs = result.moments.n_pairs
        rates[result.crop_code] = 0.0 if pairs == 0 else result.moments.crashes / pairs
    return rates


def opening_shock() -> pd.DataFrame:
    """Carrot starts 20% above its reference. Season 1 is the first season that can see the score."""

    settings = default_settings()
    initial = {
        "carrot": settings.crop("carrot").reference_price * 1.2,
        "leeks": settings.crop("leeks").reference_price,
    }
    rows = []
    for rate in ADOPTION_RATES:
        frame = run_advisory(adoption_rate=rate, seed=0, n_seasons=2, initial_prices=initial)
        carrot = frame[frame["crop_code"] == "carrot"].sort_values("season_index")
        rows.append(
            {
                "adoption_rate": rate,
                "season_0_ha": float(carrot["area_ha"].iloc[0]),
                "season_1_ha": float(carrot["area_ha"].iloc[1]),
            }
        )
    return pd.DataFrame(rows)


def format_summary(summary: pd.DataFrame, harti_rates: dict[str, float] | None, shock: pd.DataFrame) -> str:
    lines = [
        "Fixed adopters. Fair shares are the checked district targets, not an equal split.",
        "Tomato is left out. It has no checked district table.",
        "",
    ]
    for crop_code in ADVISORY_CROPS:
        lines.append(crop_code)
        crop = summary[summary["crop_code"] == crop_code]
        for row in crop.itertuples(index=False):
            lines.append(
                f"  adoption {row.adoption_rate:4.0%}"
                f"  crash rate {row.crash_rate:5.2f}"
                f"  volatility {row.price_volatility:5.2f}"
                f"  mean area {row.mean_area_ha:7.1f} ha"
            )
        if harti_rates is not None and crop_code in harti_rates:
            lines.append(f"  HARTI Colombo wholesale crash rate {harti_rates[crop_code]:5.2f}")
        baseline = crop[crop["adoption_rate"] == 0.0]
        if harti_rates is not None and not baseline.empty and crop_code in harti_rates:
            if float(baseline["crash_rate"].iloc[0]) + 0.05 < harti_rates[crop_code]:
                lines.append(
                    "  The baseline crash rate is below the HARTI rate. A lower rate with the score is not evidence that the advisory prevents those crashes."
                )
        lines.append("")
    lines.append("Carrot price starts 20% above its reference. Season 0 cannot see a score yet.")
    for row in shock.itertuples(index=False):
        lines.append(
            f"  adoption {row.adoption_rate:4.0%}"
            f"  season 0 {row.season_0_ha:7.1f} ha"
            f"  season 1 {row.season_1_ha:7.1f} ha"
        )
    return "\n".join(lines).rstrip()


def main() -> None:
    summary = summarize(sweep())
    try:
        harti_rates = harti_crash_rates()
    except (FileNotFoundError, ValueError) as exc:
        print(f"HARTI crash rates were not loaded: {type(exc).__name__}")
        harti_rates = None
    print(format_summary(summary, harti_rates, opening_shock()))


if __name__ == "__main__":
    main()
