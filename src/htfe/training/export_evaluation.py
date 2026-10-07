"""Write the training-and-evaluation sheet the studio and the viva both read."""
from __future__ import annotations

import json

import pandas as pd

from htfe.training.config import ARTIFACTS, STUDIO_DATA

from htfe.config import OUTPUTS
REPO = OUTPUTS
DOC = OUTPUTS / "notes" / "Model-Training-and-Evaluation-IT23415836.md"


def _test(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    test = frame[frame["slice"].eq("test") & frame["target_form"].eq("price")].copy()
    return test


def _rows(frame: pd.DataFrame, experiment: str) -> list[dict]:
    rows = []
    for row in frame.sort_values(["model", "horizon"]).itertuples(index=False):
        item = {
            "experiment": experiment,
            "model": row.model,
            "horizon": int(row.horizon),
            "n_scored": int(row.n_scored),
            "mae": round(float(row.mae), 2),
            "rmse": round(float(row.rmse), 2),
            "mape": round(float(row.mape), 2),
            "pinball": None if pd.isna(row.pinball) else round(float(row.pinball), 2),
            "picp": None if pd.isna(row.picp) else round(float(row.picp), 2),
            "interval_width": None if pd.isna(row.interval_width) else round(float(row.interval_width), 2),
        }
        if hasattr(row, "picp_cqr") and not pd.isna(row.picp_cqr):
            item["picp_cqr"] = round(float(row.picp_cqr), 2)
            item["interval_width_cqr"] = round(float(row.interval_width_cqr), 2)
        else:
            item["picp_cqr"] = None
            item["interval_width_cqr"] = None
        rows.append(item)
    return rows


def _markdown(rows: list[dict]) -> str:
    lines = [
        "# How the models were trained, what they output, and how they were scored",
        "",
        "Member 2. Carrot, leeks, and tomato. Score markets: Dambulla, Colombo, Meegoda, Nuwara Eliya. The label is the wholesale price, in LKR per kg, at the target week.",
        "",
        "## How training was done",
        "",
        "Each row is one crop, one market, one forecast Monday, and one horizon. Horizons are 1, 2, 4, 8, and 12 weeks. A forecast Monday may use only information already published by that Monday. The target price is the HARTI wholesale price at the later week. Missing prices are not filled with zero, and a row is scored only when both the last observed price and the target price were originally published.",
        "",
        "The split is chronological. Issue weeks through 2022-12-31 are train. Issue weeks in 2023 are calibration. Issue weeks in 2024 and 2025 are test. The series is not shuffled.",
        "",
        "The model predicts the log change from the last observed price, then converts it back with `last price × exp(prediction)`.",
        "",
        "- Experiment A uses price lags and season only.",
        "- Experiment B adds origin-district weather, diesel, USD/LKR, inflation, and the cultivated extent of seasons that have already finished.",
        "- Experiment C, current achieved hectares divided by target hectares, was not trained. That file is not on disk.",
        "",
        "SARIMAX is a univariate seasonal baseline, order (1,1,1)×(1,0,0,52). It does not take the weather or macro columns, so its error is the same in A and B.",
        "",
        "LightGBM and XGBoost are quantile models at 0.05, 0.50, and 0.95. Optuna tuned them on horizon 4 only, using pinball loss of the median price on the 2023 calibration weeks, then those settings were frozen for every horizon. The seed is 42. LightGBM used 15 trials. XGBoost used 8.",
        "",
        "Intervals are the 0.05 and 0.95 forecasts. Split conformal calibration adds a constant, fit only on 2023, to both tails. The 2024–2025 prices are not used to choose that constant.",
        "",
        "## What the output is",
        "",
        "For each scored row the saved prediction file has the issue week, the target week, the horizon, the true price, the median forecast, and the lower and upper forecast. The studio price is the horizon-1 Experiment B LightGBM median for the latest test week of that crop and market, with the 2023 conformal adjustment on the interval. Sending cultivation hectares does not change it, because that column was not in the model.",
        "",
        "## How performance was measured",
        "",
        "On the 2024–2025 test rows, for each model and horizon:",
        "",
        "- MAE and RMSE in LKR per kg",
        "- MAPE in percent",
        "- Pinball loss of the median forecast",
        "- PICP, the share of true prices inside the 5–95% interval, before and after conformal calibration",
        "- Mean interval width, before and after calibration",
        "",
        "There is no accuracy, F1, or confusion matrix. The target is a price, not a class.",
        "",
        "## Experiment B test scores",
        "",
        "| Model | Horizon | n | MAE | RMSE | MAPE | Pinball | PICP raw | PICP calibrated | Width calibrated |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["experiment"] != "B":
            continue
        pinball = "" if row["pinball"] is None else f"{row['pinball']:.2f}"
        raw = "" if row["picp"] is None else f"{row['picp']:.2f}"
        calibrated = "" if row["picp_cqr"] is None else f"{row['picp_cqr']:.2f}"
        width = "" if row["interval_width_cqr"] is None else f"{row['interval_width_cqr']:.2f}"
        lines.append(
            f"| {row['model']} | {row['horizon']} | {row['n_scored']} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['mape']:.2f} | {pinball} | {raw} | {calibrated} | {width} |"
        )
    lines.extend(
        [
            "",
            "## What the scores say",
            "",
            "At 1 week, SARIMAX has the lowest MAE, 62.22. LightGBM B is 75.00 and XGBoost B is 74.10. At 4 weeks, LightGBM B is 105.98, ahead of XGBoost B at 106.80 and SARIMAX at 122.64. At 12 weeks, LightGBM B is 106.66 against SARIMAX at 181.44.",
            "",
            "Against Experiment A, LightGBM’s MAE falls at every horizon once weather, macros, and lagged extent are added. The 4-week drop is 10.12%, from 117.91 to 105.98. The 12-week drop is 17.91%, from 129.93 to 106.66.",
            "",
            "A separate fit that adds only one of those groups, with the same LightGBM settings, shows that at 4 weeks weather helps most on its own, then macros, then lagged extent. Together they help more than any one group, and the gains overlap. At 12 weeks, weather alone does not beat A. The long-horizon gain is the combination.",
            "",
            "Calibrated coverage is about 79–84%. The nominal interval is 90%. Calibration widened the bands and still did not reach 90%. That is reported as a shortfall.",
            "",
            "MAPE above 50% at 8 and 12 weeks is the difficulty of a long wholesale-price forecast, and it is worse for tomato than for leeks. The crop-and-market breakdown is in `docs/Experiment-AB-Comparison-IT23415836.md`.",
            "",
            "Experiment C is not in this table. A seasonal Census total and a synthetic planting curve were not used as current cultivation progress.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    a_rows = _rows(_test(ARTIFACTS / "experiment_a" / "metrics_by_model_horizon.csv"), "A")
    b_rows = _rows(_test(ARTIFACTS / "experiment_b" / "metrics_by_model_horizon.csv"), "B")
    payload = {
        "label": "wholesale price, LKR/kg, at the target week",
        "split": {
            "train": "issue weeks through 2022-12-31",
            "calibration": "issue weeks in 2023",
            "test": "issue weeks in 2024 and 2025",
        },
        "horizons": [1, 2, 4, 8, 12],
        "models": ["sarimax", "lightgbm", "xgboost"],
        "quantiles": [0.05, 0.50, 0.95],
        "optuna": "horizon 4, median pinball on 2023, 15 LightGBM trials, 8 XGBoost trials, seed 42",
        "experiment_c": "not trained",
        "rows": a_rows + b_rows,
    }
    (STUDIO_DATA / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    DOC.write_text(_markdown(b_rows), encoding="utf-8")
    print(f"wrote {STUDIO_DATA / 'evaluation.json'}", flush=True)
    print(f"wrote {DOC}", flush=True)


if __name__ == "__main__":
    main()
