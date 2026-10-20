"""Write the training-and-evaluation sheet the studio and the viva both read."""
from __future__ import annotations

import json
import math
from pathlib import Path

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


def _one(rows: list[dict], experiment: str, model: str, horizon: int) -> dict | None:
    for row in rows:
        if row["experiment"] == experiment and row["model"] == model and row["horizon"] == horizon:
            return row
    return None


def _score_paragraph(rows: list[dict]) -> str:
    def mae(experiment: str, model: str, horizon: int) -> str:
        row = _one(rows, experiment, model, horizon)
        return "not scored" if row is None else f"{row['mae']:.2f}"

    a4 = _one(rows, "A", "lightgbm", 4)
    b4 = _one(rows, "B", "lightgbm", 4)
    drop = ""
    if a4 is not None and b4 is not None and a4["mae"]:
        change = (b4["mae"] - a4["mae"]) / a4["mae"] * 100
        drop = f" LightGBM MAE at 4 weeks changes by {change:.2f}% from A ({a4['mae']:.2f}) to B ({b4['mae']:.2f})."
    coverage = ""
    if b4 is not None and b4.get("picp_cqr") is not None:
        picp = float(b4["picp_cqr"])
        if picp >= 90:
            coverage = f" The 4-week LightGBM calibrated coverage is {picp:.2f}%. The intended level is 90%."
        else:
            coverage = f" The 4-week LightGBM calibrated coverage is {picp:.2f}%, which is below the intended 90%."
    return (
        f"At 1 week, SARIMAX MAE is {mae('B', 'sarimax', 1)}, LightGBM B is {mae('B', 'lightgbm', 1)}, "
        f"and XGBoost B is {mae('B', 'xgboost', 1)}. "
        f"At 4 weeks, LightGBM B is {mae('B', 'lightgbm', 4)}, XGBoost B is {mae('B', 'xgboost', 4)}, "
        f"and SARIMAX is {mae('B', 'sarimax', 4)}. "
        f"At 12 weeks, LightGBM B is {mae('B', 'lightgbm', 12)} and SARIMAX is {mae('B', 'sarimax', 12)}."
        f"{drop}{coverage} Read the table above for the full set. Do not quote a number that is not in that table."
    )


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
        "- Experiment C adds the national as-of cultivation ratio on the rows that have a real DOA report. It is a matched-row test against B* and is not mixed into the full-sample A/B table. District rows are audit-only. LSTM was not run.",
        "",
        "SARIMAX is a univariate seasonal baseline, order (1,1,1)×(1,0,0,52). It does not take the weather or macro columns, so its error is the same in A and B.",
        "",
        "LightGBM and XGBoost are quantile models at 0.05, 0.50, and 0.95. Hyperparameters are the notebook defaults, seed 42. Optuna is not used. The tree count is chosen by early stopping on 2022 target weeks after fitting through 2021, then the trees are refit through 2022. 2023 is used only for the interval.",
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
            _score_paragraph(rows),
            "",
            "Experiment C is not in this full-sample table. The nominal interval is 90%. Report the measured calibrated PICP. Do not describe that measured share as 90% unless it is at least 90.",
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
        "optuna": "not used. Fixed notebook hyperparameters, seed 42. Tree count selected on 2022 targets, then refit through 2022.",
        "tree_protocol": "fit through 2021, early-stop on 2022, refit through 2022, 2023 only for qhat",
        "experiment_c": "matched-row national test, separate from this full-sample table",
        "lstm": "not run",
        "rows": a_rows + b_rows,
    }
    (STUDIO_DATA / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(_markdown(a_rows + b_rows), encoding="utf-8")
    _refresh_screens()
    print(f"wrote {STUDIO_DATA / 'evaluation.json'}", flush=True)
    print(f"wrote {DOC}", flush=True)


def _refresh_screens() -> None:
    """Point the studio 4-week cards and the interval chart at this training run."""
    from htfe.training.evaluate import project_interval_around_median

    predictions = pd.read_csv(ARTIFACTS / "experiment_b" / "predictions.csv")
    cqr = json.loads((ARTIFACTS / "experiment_b" / "cqr.json").read_text(encoding="utf-8"))
    qhat = float(cqr["qhat_by_model"]["lightgbm"]["4"])
    test = predictions[
        predictions["model"].eq("lightgbm") & predictions["split"].eq("test") & predictions["horizon"].eq(4)
    ].copy()
    low, mid, high = project_interval_around_median(
        test["y_pred_q05"].to_numpy(dtype=float),
        test["y_pred"].to_numpy(dtype=float),
        test["y_pred_q95"].to_numpy(dtype=float),
    )
    test["lower"] = pd.Series(low - qhat, index=test.index).clip(lower=0.01)
    test["upper"] = high + qhat
    test["point"] = mid
    test["week_start"] = pd.to_datetime(test["target_week"]).dt.strftime("%Y-%m-%d")
    test["market_key"] = test["market"].astype(str).str.lower().str.replace(" ", "_", regex=False)
    covered = (test["y_true"] >= test["lower"]) & (test["y_true"] <= test["upper"])
    chart = [
        {
            "crop": row.crop,
            "market": row.market_key,
            "week_start": row.week_start,
            "y_true": round(float(row.y_true), 4),
            "point": round(float(row.point), 4),
            "lower": round(float(row.lower), 4),
            "upper": round(float(row.upper), 4),
        }
        for row in test.itertuples(index=False)
    ]
    (STUDIO_DATA / "predictions.json").write_text(json.dumps({"lightgbm": chart}), encoding="utf-8")

    def block(frame: pd.DataFrame) -> dict:
        width = (frame["upper"] - frame["lower"]).mean()
        return {
            "n_scored": int(len(frame)),
            "mae": round(float((frame["y_true"] - frame["point"]).abs().mean()), 2),
            "mape": round(float(((frame["y_true"] - frame["point"]).abs() / frame["y_true"]).mean() * 100), 2),
            "picp": round(float(covered.loc[frame.index].mean() * 100), 2),
            "interval_width": round(float(width), 2),
        }

    models = json.loads((STUDIO_DATA / "models.json").read_text(encoding="utf-8"))
    overall = block(test)
    measured = overall["picp"]
    if measured is not None and float(measured) >= 90:
        coverage_note = f"Measured calibrated PICP is {measured:.2f}%. The intended level is 90%."
    else:
        coverage_note = f"Measured calibrated PICP is {measured}%. That is below the intended 90%."
    overall["note"] = (
        "Experiment B LightGBM, horizon 4, 2024-2025 test. PICP uses the 2023 conformal adjustment on the ordered interval. "
        + coverage_note
    )
    models["lightgbm"]["metrics"]["overall"] = overall
    models["lightgbm"]["metrics"]["by_crop"] = {crop: block(part) for crop, part in test.groupby("crop")}
    models["lightgbm"]["metrics"]["by_market"] = {market: block(part) for market, part in test.groupby("market_key")}
    models["lightgbm"]["metrics"]["by_season"] = {}
    models["lightgbm"]["train_curve"] = []
    models["lightgbm"]["qhat_by_horizon"] = {
        str(horizon): float(value) for horizon, value in cqr["qhat_by_model"]["lightgbm"].items()
    }
    models["lightgbm"]["qhat"] = float(cqr["qhat_by_model"]["lightgbm"]["1"])

    metrics = pd.read_csv(ARTIFACTS / "experiment_b" / "metrics_by_model_horizon.csv")
    horizon4 = metrics[metrics["slice"].eq("test") & metrics["target_form"].eq("price") & metrics["horizon"].eq(4)]
    for model_name, key in (("xgboost", "xgboost"), ("sarimax", "sarima")):
        match = horizon4[horizon4["model"].eq(model_name)]
        block_metrics = models.get(key, {}).get("metrics")
        if match.empty or not isinstance(block_metrics, dict):
            continue
        item = match.iloc[0]
        block_metrics["overall"] = {
            "n_scored": int(item["n_scored"]),
            "mae": round(float(item["mae"]), 2),
            "rmse": round(float(item["rmse"]), 2),
            "mape": round(float(item["mape"]), 2),
            "pinball": None if pd.isna(item["pinball"]) else round(float(item["pinball"]), 2),
            "picp": None if "picp_cqr" not in item or pd.isna(item["picp_cqr"]) else round(float(item["picp_cqr"]), 2),
            "interval_width": None
            if "interval_width_cqr" not in item or pd.isna(item["interval_width_cqr"])
            else round(float(item["interval_width_cqr"]), 2),
            "note": "Horizon 4, 2024-2025 test, this notebook-protocol run. Not pooled across horizons.",
        }
        block_metrics["by_crop"] = {}
        block_metrics["by_market"] = {}
        block_metrics["by_season"] = {}
    (STUDIO_DATA / "models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")

    comparison = json.loads((STUDIO_DATA / "comparison.json").read_text(encoding="utf-8"))
    ids = {"sarimax": "sarima", "lightgbm": "lightgbm", "xgboost": "xgboost"}
    for row in comparison:
        model = next((name for name, label in ids.items() if row.get("id") == label), None)
        if model is None:
            continue
        match = horizon4[horizon4["model"].eq(model)]
        if match.empty:
            continue
        item = match.iloc[0]
        row["n_scored"] = int(item["n_scored"])
        row["mae"] = round(float(item["mae"]), 2)
        row["rmse"] = round(float(item["rmse"]), 2)
        row["mape"] = round(float(item["mape"]), 2)
        row["pinball"] = None if pd.isna(item["pinball"]) else round(float(item["pinball"]), 2)
        if model == "lightgbm":
            row["picp"] = overall["picp"]
            row["interval_width"] = overall["interval_width"]
        elif "picp_cqr" in item and not pd.isna(item["picp_cqr"]):
            row["picp"] = round(float(item["picp_cqr"]), 2)
            row["interval_width"] = round(float(item["interval_width_cqr"]), 2)
    _refresh_naive(comparison, test)
    (STUDIO_DATA / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")


def _refresh_naive(comparison: list[dict], lightgbm_test: pd.DataFrame) -> None:
    """Replace the old pooled naive rows with this run's 4-week test rows."""
    from htfe.config import built_file
    from htfe.training.baselines import score_same_rows

    table_path = built_file("training_table.parquet")
    if not table_path.is_file():
        return
    scored = score_same_rows(lightgbm_test, pd.read_parquet(table_path))
    for row in comparison:
        block = scored.get(row.get("id"))
        if block is None:
            continue
        row["n_scored"] = block["n_scored"]
        row["mae"] = block["mae"]
        row["rmse"] = block["rmse"]
        row["mape"] = block["mape"]
        row["pinball"] = None
        row["picp"] = None
        row["interval_width"] = None


def _json_ready(value):
    """JSON has no NaN. A missing SARIMAX interval is null, not a number."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if value is pd.NA or (not isinstance(value, (str, bytes, list, dict)) and pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def export_this_run_studio(out_dir: Path | None = None) -> Path:
    """Studio JSON for this run. Does not write app/data or the full-sample A/B cards."""
    from htfe.training.evaluate import assert_safe_evaluation_dir

    destination = assert_safe_evaluation_dir(out_dir or (OUTPUTS / "evaluation" / "studio_json"))
    destination.mkdir(parents=True, exist_ok=True)
    evaluation_root = OUTPUTS / "evaluation"
    c_root = OUTPUTS / "experiment_c"
    benchmark = pd.read_csv(evaluation_root / "common_key_benchmark.csv")
    seasonal = pd.read_csv(evaluation_root / "seasonal_naive_benchmark.csv")
    best = pd.read_csv(evaluation_root / "best_model_by_horizon.csv")
    naive_bootstrap = pd.read_csv(evaluation_root / "model_vs_naive_block_bootstrap.csv")
    b_saved = pd.read_csv(OUTPUTS / "experiment_b" / "metrics_by_model_horizon.csv")
    b_saved = b_saved[b_saved["slice"].eq("test") & b_saved["target_form"].eq("price")]
    c_metrics = pd.read_csv(c_root / "metrics_c.csv")
    comparison = pd.read_csv(c_root / "b_star_vs_c.csv")
    probabilistic = pd.read_csv(c_root / "evaluation" / "final_probabilistic_evaluation_c.csv")
    crossing = pd.read_csv(c_root / "evaluation" / "quantile_crossing_diagnostic_c.csv")
    conditional = pd.read_csv(c_root / "evaluation" / "conditional_interval_evaluation_c.csv")
    novelty_bootstrap = pd.read_csv(c_root / "evaluation" / "bstar_vs_c_block_bootstrap.csv")
    shap_path = c_root / "evaluation" / "shap_lightgbm_c_h4.csv"
    shap = pd.read_csv(shap_path) if shap_path.is_file() else pd.DataFrame()
    member3 = json.loads((c_root / "inference" / "member3_forecast_c.json").read_text(encoding="utf-8"))
    status = json.loads((c_root / "status.json").read_text(encoding="utf-8"))
    b_example = json.loads((evaluation_root / "inference" / "saved_b_example.json").read_text(encoding="utf-8"))

    def dump(name: str, payload: dict) -> None:
        text = json.dumps(_json_ready(payload), indent=2, default=str, allow_nan=False)
        (destination / name).write_text(text, encoding="utf-8")

    dump(
        "meta.json",
        {
            "project": "J26-DS-322",
            "component": "Hybrid Temporal Forecasting Engine",
            "target": "weekly wholesale price, LKR/kg",
            "saved_artifact_tree_selection": (
                "Eligible targets through 2021-12-31, early stopping on 2022 target weeks, "
                "refit through 2022-12-31. 2023 is used only for qhat. "
                "This is the fit behind the Experiment A, B, and C cards from this run."
            ),
            "tree_fit": "htfe.training.run_experiment_a.fit_quantile_trees",
            "notebook_protocol_produced_these_cards": True,
            "notebook_protocol": (
                "selector training through 2021-12-31, early stopping on 2022 target weeks, "
                "refit through 2022-12-31, 2023 reserved for conformal calibration"
            ),
            "app_data_rewritten": False,
            "experiment_c_trained": bool(status.get("experiment_c_trained")),
            "experiment_c_status": status.get("status"),
            "spatial_scope": "national",
            "district_rows_used_for_scoring": False,
            "sarimax_order": "(1, 1, 1) x (1, 0, 0, 52)",
            "lstm": "not run",
            "measured_picp_pct_by_horizon": member3.get("measured_picp_pct_by_horizon"),
            "saved_b_reproduction": b_example,
        },
    )
    dump(
        "evaluation.json",
        {
            "label": "this run, read from saved prediction files, not copied from a notebook printout",
            "full_sample_experiment_b_test": b_saved.to_dict(orient="records"),
            "matched_row_experiment_c": c_metrics.to_dict(orient="records"),
            "fair_common_key_benchmark": benchmark.to_dict(orient="records"),
            "note": (
                "full_sample_experiment_b_test is the saved Experiment B file. "
                "fair_common_key_benchmark is the identical-key subset. "
                "Neither replaces app/data."
            ),
        },
    )
    dump(
        "ablation.json",
        {
            "b_star_vs_c": comparison.to_dict(orient="records"),
            "interpretation": (
                "A negative MAE change means C has lower MAE than B* on the matched rows of this run. "
                "C is lower only where mae_change_pct is negative. MAE and MAPE are separate columns."
            ),
        },
    )
    package = {
        "point_forecast_benchmark": benchmark.to_dict(orient="records"),
        "best_point_model_by_horizon": best.to_dict(orient="records"),
        "seasonal_naive_common_subset": seasonal.to_dict(orient="records"),
        "model_vs_naive_bootstrap": naive_bootstrap.to_dict(orient="records"),
        "probabilistic_evaluation_c": probabilistic.to_dict(orient="records"),
        "quantile_crossing_diagnostic_c": crossing.to_dict(orient="records"),
        "conditional_interval_evaluation_c": conditional.to_dict(orient="records"),
        "bstar_vs_c_bootstrap": novelty_bootstrap.to_dict(orient="records"),
        "shap_top_features_h4_c": shap.to_dict(orient="records"),
        "measured_picp_pct_by_horizon": member3.get("measured_picp_pct_by_horizon"),
        "coverage_level_aim": 0.90,
        "coverage_level_note": member3.get("coverage_level_note"),
    }
    dump("final_evaluation_package.json", package)
    return destination


if __name__ == "__main__":
    main()
