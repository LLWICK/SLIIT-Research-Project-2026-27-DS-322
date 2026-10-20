"""Small checks for saved-model helpers. They do not load the Experiment C models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from htfe.training.evaluate_saved_c import moving_block_ci, shock_slice
from htfe.training.saved_inference import (
    CROPS,
    HORIZONS,
    SCORE_MARKETS,
    complete_issue_weeks,
    exceeded_flag,
    interval_from_prices,
)


def test_exceeded_flag_keeps_a_ratio_above_one() -> None:
    assert exceeded_flag(1.25) == 1
    assert exceeded_flag(1.0) == 0
    assert exceeded_flag(0.4) == 0


def test_interval_orders_the_tails_before_qhat() -> None:
    lower, median, upper = interval_from_prices(10.0, 5.0, 8.0, 1.0)
    assert (lower, median, upper) == (7.0, 5.0, 11.0)


def test_moving_block_interval_is_the_constant_when_values_do_not_vary() -> None:
    low, high = moving_block_ci(np.ones(12), block_length=4, reps=40, seed=1)
    assert low == 1.0
    assert high == 1.0


def test_packet_issue_week_is_the_latest_complete_monday() -> None:
    held_out = "nuwara_eliya"
    assert held_out in SCORE_MARKETS
    rows = []
    for issue in ("2025-09-22", "2025-09-29"):
        for crop in CROPS:
            for market in SCORE_MARKETS:
                for horizon in HORIZONS:
                    if issue == "2025-09-29" and market == held_out and int(horizon) == 4:
                        continue
                    rows.append(
                        {
                            "crop_key": crop,
                            "market_key": market,
                            "horizon": int(horizon),
                            "issue": issue,
                        }
                    )
    assert complete_issue_weeks(pd.DataFrame(rows)) == ["2025-09-22"]


def test_empty_shock_window_is_not_evaluable() -> None:
    frame = pd.DataFrame(
        {
            "forecast_issue_week": ["2024-06-03", "2024-06-10"],
            "y_true": [10.0, 12.0],
            "y_pred": [11.0, 9.0],
        }
    )
    table, note = shock_slice(frame)
    assert int(table.iloc[0]["n"]) == 0
    assert "not evaluable" in note
    assert "2" in note
