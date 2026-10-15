"""Tree chronology and point-metric definitions. No model retrain."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from htfe.config import OUTPUTS, STUDIO_DATA
from htfe.models.trees import selection_masks
from htfe.training.evaluate import assert_safe_evaluation_dir, extended_point_metrics, project_interval_around_median
from htfe.training.run_experiment_a import fit_quantile_trees


def _frame() -> pd.DataFrame:
    rows = []
    for year, count, split in ((2021, 8, "train"), (2022, 6, "train"), (2023, 5, "calibrate"), (2024, 4, "test")):
        for index in range(count):
            rows.append(
                {
                    "target_week": pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(weeks=index),
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


class NotebookProtocolTests(unittest.TestCase):
    def test_selection_masks_keep_2023_out_of_selection(self) -> None:
        masks = selection_masks(
            _frame(),
            min_selector_train=5,
            min_selector_validation=4,
            min_final_train=8,
            min_calibration=4,
        )
        self.assertEqual(int(masks["selector_train"].sum()), 8)
        self.assertEqual(int(masks["selector_validation"].sum()), 6)
        self.assertEqual(int(masks["final_train"].sum()), 14)
        self.assertEqual(int(masks["calibration"].sum()), 5)
        self.assertLessEqual(masks["selector_validation_max_target"], pd.Timestamp("2022-12-31"))
        self.assertLessEqual(masks["final_train_max_target"], pd.Timestamp("2022-12-31"))
        self.assertFalse(bool(np.any(masks["selector_validation"] & masks["calibration"])))

    def test_extended_metrics_keep_mae_and_mape_separate(self) -> None:
        metrics = extended_point_metrics(np.array([100.0, 200.0]), np.array([110.0, 150.0]))
        self.assertAlmostEqual(metrics["mae"], 30.0)
        self.assertAlmostEqual(metrics["mape"], 17.5)
        self.assertAlmostEqual(metrics["mean_bias"], -20.0)
        low, mid, high = project_interval_around_median(np.array([9.0]), np.array([5.0]), np.array([8.0]))
        self.assertEqual(float(low[0]), 5.0)
        self.assertEqual(float(mid[0]), 5.0)
        self.assertEqual(float(high[0]), 9.0)

    def test_quantile_fit_is_not_blocked_as_a_second_score_folder(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_safe_evaluation_dir(OUTPUTS / "experiment_a")
        with self.assertRaises(RuntimeError):
            assert_safe_evaluation_dir(STUDIO_DATA)
        with self.assertRaises(Exception) as caught:
            fit_quantile_trees(pd.DataFrame(), [], "A", model_dir=OUTPUTS / "experiment_a" / "models")
        self.assertNotIn("Refusing", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
