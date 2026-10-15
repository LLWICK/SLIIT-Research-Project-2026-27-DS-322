"""Train Experiment A on the primary training table.

From the repo root, in Command Prompt, after `set PYTHONPATH=src`:
  python -u -m htfe.training.run_experiment_a
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from htfe.training.config import ARTIFACTS, CAL_END, EVAL_MARKETS, HORIZONS, TRAIN_END
from htfe.training.evaluate import metrics_tables
from htfe.features.sets import (
    PRICE_HISTORY,
    SEASON_FEATURES,
    feature_sets,
    historical_columns,
    macros_ready,
)
from htfe.models.trees import (
    NOTEBOOK_TREE_PROTOCOL,
    fit_selected_lightgbm,
    fit_selected_xgboost,
    selection_masks,
)

RANDOM_STATE = 42
EVAL_IDS = {market.lower().replace(" ", "_") for market in EVAL_MARKETS}
MARKET_NAME = {market.lower().replace(" ", "_"): market for market in EVAL_MARKETS}
OUT = ARTIFACTS / "experiment_a"


def _eligible(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_week = pd.to_datetime(frame["target_week"])
    issue = pd.to_datetime(frame["forecast_issue_week"])
    keep = (
        frame["forecast_horizon_weeks"].eq(horizon)
        & frame["last_observed_price_lkr_kg"].notna()
        & frame["target_is_observed"].astype(bool)
        & frame["target_price_lkr_kg"].notna()
        & frame["last_observed_price_lkr_kg"].gt(0)
        & frame["target_price_lkr_kg"].gt(0)
        & issue.notna()
    )
    leak_train = frame["split"].eq("train") & target_week.gt(pd.Timestamp(TRAIN_END))
    leak_cal = frame["split"].eq("calibrate") & target_week.gt(pd.Timestamp(CAL_END))
    subset = frame.loc[keep & ~leak_train & ~leak_cal].copy()
    subset["y_logret"] = np.log(subset["target_price_lkr_kg"] / subset["last_observed_price_lkr_kg"])
    return subset


def _assert_split(frame: pd.DataFrame) -> None:
    issue = pd.to_datetime(frame["forecast_issue_week"])
    train = frame["split"].eq("train")
    calibrate = frame["split"].eq("calibrate")
    test = frame["split"].eq("test")
    if bool(train.any()) and issue[train].max() > pd.Timestamp(TRAIN_END):
        raise AssertionError("train issue week is after 2022-12-31")
    if bool(calibrate.any()) and not bool(issue[calibrate].dt.year.eq(2023).all()):
        raise AssertionError("calibration issue weeks are not all in 2023")
    if bool(test.any()) and not bool(issue[test].dt.year.isin([2024, 2025]).all()):
        raise AssertionError("test issue weeks are not all in 2024-2025")


def _design(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    numeric_names = [name for name in columns if name != "season"]
    pieces = [frame[numeric_names].apply(pd.to_numeric, errors="coerce")]
    names = list(numeric_names)
    for name in ("crop_id", "market_id", "season"):
        series = frame[name]
        if not isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("category")
        codes = series.cat.codes.astype(float)
        pieces.append(codes.where(codes >= 0).rename(name))
        names.append(name)
    return pd.concat(pieces, axis=1), names


def _score_frame(
    score: pd.DataFrame,
    point: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    model: str,
    *,
    raw_quantiles: bool = False,
) -> pd.DataFrame:
    market = score["market_id"].map(MARKET_NAME).fillna(score["market_id"])
    if raw_quantiles:
        width_low = np.asarray(low, dtype=float)
        width_high = np.asarray(high, dtype=float)
    else:
        width_low = np.minimum(low, high)
        width_high = np.maximum(low, high)
    return pd.DataFrame(
        {
            "unique_id": score["crop_id"].astype(str) + "__" + market.astype(str),
            "crop": score["crop_id"].astype(str),
            "market": market.astype(str),
            "forecast_issue_week": pd.to_datetime(score["forecast_issue_week"]),
            "target_week": pd.to_datetime(score["target_week"]),
            "horizon": score["forecast_horizon_weeks"].astype(int),
            "y_true": score["target_price_lkr_kg"].astype(float),
            "y_pred": point,
            "y_pred_q05": width_low,
            "y_pred_q95": width_high,
            "model": model,
            "split": score["split"].astype(str),
            "is_shock": score["is_shock"].astype(bool),
            "season": score["season"].astype(str),
            "target_form": "price",
            "feature_set": "A",
        }
    )


def _status(trained_a: bool, a_reason: str, macro_reason: str, b_ready: bool, c_reason: str) -> dict:
    return {
        "A": {"status": "trained" if trained_a else "not_trained", "reason": a_reason},
        "B": {
            "status": "not_trained" if b_ready else "unavailable",
            "reason": (
                macro_reason + " This task trains Experiment A only."
                if b_ready
                else macro_reason
            ),
        },
        "C": {
            "status": "unavailable",
            "reason": c_reason,
        },
        "S": {
            "status": "not_run",
            "reason": "S is the synthetic calendar curve and was not trained. It is not Experiment C, and its scores are not in the A/B/C accuracy table.",
        },
    }


def fit_quantile_trees(
    frame: pd.DataFrame,
    columns: list[str],
    feature_set: str,
    *,
    model_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Notebook 1 tree fit: 2021 selector train, 2022 early stop, refit through 2022.

    Fixed LightGBM and XGBoost defaults are used. Optuna is not run. 2023 is not
    an early-stopping set. This is the fit the experiment entry points run, and
    model_dir may be the experiment model folder the studio scores are built from.
    """
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    parts = []
    records: list[dict] = []
    for horizon in HORIZONS:
        subset = _eligible(frame, horizon)
        subset = subset.copy()
        subset["crop_id"] = pd.Categorical(subset["crop_id"].astype(str))
        subset["market_id"] = pd.Categorical(subset["market_id"].astype(str))
        subset["season"] = pd.Categorical(subset["season"].astype(str), categories=["Maha", "Yala"])
        design, names = _design(subset, columns)
        categories = {
            "crop_id": [str(value) for value in subset["crop_id"].cat.categories],
            "market_id": [str(value) for value in subset["market_id"].cat.categories],
            "season": [str(value) for value in subset["season"].cat.categories],
        }
        y = subset["y_logret"].to_numpy(dtype=float)
        masks = selection_masks(subset)
        score_mask = subset["market_id"].isin(EVAL_IDS) & subset["split"].isin(["calibrate", "test"])
        score = subset.loc[score_mask]
        x_score = design.loc[score_mask]
        origin = score["last_observed_price_lkr_kg"].to_numpy(dtype=float)
        fitted: dict[str, dict] = {"lightgbm": {}, "xgboost": {}}
        for alpha, label in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95")):
            lgb_model, lgb_record = fit_selected_lightgbm(design, y, masks, alpha, None)
            xgb_model, xgb_record = fit_selected_xgboost(design, y, masks, alpha, None)
            fitted["lightgbm"][label] = lgb_model
            fitted["xgboost"][label] = xgb_model
            for record in (lgb_record, xgb_record):
                record["feature_set"] = feature_set
                record["horizon"] = int(horizon)
                record["optuna_used"] = False
                records.append(record)
            if model_dir is not None:
                for family, model in (("lightgbm", lgb_model), ("xgboost", xgb_model)):
                    joblib.dump(
                        {
                            "model": model,
                            "columns": names,
                            "feature_columns": list(columns),
                            "feature_set": feature_set,
                            "horizon": int(horizon),
                            "quantile": alpha,
                            "label": label,
                            "protocol": NOTEBOOK_TREE_PROTOCOL,
                            "random_state": RANDOM_STATE,
                            "categories": categories,
                        },
                        model_dir / f"{family}_{feature_set}_h{int(horizon)}_{label}.joblib",
                    )

        def _prices(family: str) -> dict[str, np.ndarray]:
            out = {}
            for label, model in fitted[family].items():
                raw = np.asarray(model.predict(x_score), dtype=float)
                out[label] = np.maximum(origin * np.exp(raw), 0.01)
            return out

        for family in ("lightgbm", "xgboost"):
            prices = _prices(family)
            scored = _score_frame(score, prices["q50"], prices["q05"], prices["q95"], family, raw_quantiles=True)
            scored["feature_set"] = feature_set
            parts.append(scored)
    if model_dir is not None:
        payload = {
            "protocol": NOTEBOOK_TREE_PROTOCOL,
            "selector_train": "eligible training targets through 2021-12-31",
            "selector_validation": "eligible training targets in 2022",
            "final_refit": "all eligible training targets through 2022-12-31",
            "conformal_calibration": "2023 only; not used for model selection",
            "optuna_used": False,
            "saved_studio_cards": "produced when this function is called by the experiment entry points",
            "records": records,
        }
        (model_dir / "training_protocol.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), records


def main() -> None:
    global OUT
    from htfe.training.publish import begin_staging, discard_staging, publish_staging

    official = OUT
    stage = begin_staging(official.name)
    OUT = stage
    published = False
    try:
        _train_experiment_a()
        publish_staging(stage, official)
        published = True
    finally:
        OUT = official
        if not published:
            discard_staging(stage)


def _train_experiment_a() -> None:
    from htfe.config import built_file
    from htfe.models.sarimax import fit_notebook_from_table
    from htfe.training.conformal import cqr_tables

    table_path = built_file("training_table.parquet")
    if not table_path.is_file():
        raise FileNotFoundError(table_path)
    print(f"notebook protocol on {table_path}", flush=True)
    print("fit through 2021, early-stop on 2022, refit through 2022, 2023 only for qhat", flush=True)
    print("fixed LightGBM and XGBoost hyperparameters, seed 42, Optuna is not used", flush=True)
    table = pd.read_parquet(table_path)
    _assert_split(table)
    expected = list(PRICE_HISTORY) + list(SEASON_FEATURES)
    columns = historical_columns(table)
    if columns != expected:
        raise RuntimeError(f"Experiment A features are missing: {set(expected) - set(columns)}")
    sets = feature_sets(table)
    if sets["A"] != columns:
        raise RuntimeError("feature set A does not match the price-lag and season columns")
    ready, macro_reason = macros_ready(table)
    c_reason = "Experiment C is the national matched-row run in run_experiment_c. It is not fit inside Experiment A."
    print(macro_reason, flush=True)
    print(c_reason, flush=True)
    tree_rows, records = fit_quantile_trees(table, columns, "A", model_dir=OUT / "models")
    print(f"tree rows {len(tree_rows)}", flush=True)
    sarimax, sarimax_meta = fit_notebook_from_table(table, partial_path=OUT / "sarimax_partial.csv")
    sarimax_note = f"fitted {len(sarimax)} SARIMAX rows at fixed order (1, 1, 1) x (1, 0, 0, 52)"
    print(sarimax_note, flush=True)
    parts = [tree_rows]
    if not sarimax.empty:
        parts.append(sarimax)
    predictions = pd.concat(parts, ignore_index=True)
    by_model, by_series = metrics_tables(predictions)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    cqr_rows, cqr_payload = cqr_tables(predictions)
    (OUT / "cqr.json").write_text(json.dumps(cqr_payload, indent=2), encoding="utf-8")
    (OUT / "sarimax_order.json").write_text(json.dumps(sarimax_meta, indent=2), encoding="utf-8")
    by_model = by_model.merge(
        cqr_rows[["model", "horizon", "qhat", "picp_cqr", "interval_width_cqr"]],
        on=["model", "horizon"],
        how="left",
    )
    by_model.loc[~by_model["slice"].eq("test"), ["qhat", "picp_cqr", "interval_width_cqr"]] = np.nan
    by_model.to_csv(OUT / "metrics_by_model_horizon.csv", index=False)
    by_series.to_csv(OUT / "metrics_by_series.csv", index=False)
    tuned = {
        "used": False,
        "reason": "Notebook 1 uses fixed LightGBM and XGBoost hyperparameters. Optuna is not run.",
        "lightgbm": {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators_cap": 500,
            "early_stopping_rounds": 50,
        },
        "xgboost": {
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators_cap": 500,
            "early_stopping_rounds": 50,
        },
        "tree_count_records": len(records),
    }
    manifest = {
        "experiment": "A",
        "description": "Price lags and seasonal features only. No weather, macros, cultivation progress, or lagged extent.",
        "protocol": NOTEBOOK_TREE_PROTOCOL,
        "executed": True,
        "features": columns,
        "identity_codes": ["crop_id", "market_id"],
        "season_code": "season",
        "label": "target_price_lkr_kg at target_week",
        "model_target": "log(target_price / last_observed_price), inverted back to LKR/kg",
        "quantiles": [0.05, 0.50, 0.95],
        "random_state": RANDOM_STATE,
        "direct_multi_horizon": True,
        "split": {
            "train_issue_through": TRAIN_END,
            "selector_train": "eligible training targets through 2021-12-31",
            "selector_validation": "eligible training targets in 2022",
            "final_refit": "eligible training targets through 2022-12-31",
            "calibrate": "2023 issue weeks only, not used for early stopping",
            "test": "2024-2025",
            "shuffle": False,
        },
        "score_markets": list(EVAL_MARKETS),
        "train_pool_markets": ["Kandy", "Keppetipola", "Thambuttegama"],
        "score_rule": "last observed price and target price both originally observed",
        "optuna": tuned,
        "sarimax": sarimax_note,
        "sarimax_order": sarimax_meta,
        "lstm": "not run",
        "feature_sets": {name: value if value is None else value for name, value in sets.items()},
        "rows": int(len(table)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    payload = _status(
        True,
        "Experiment A trained on price lags and seasonal features.",
        macro_reason,
        ready,
        c_reason,
    )
    payload["sarimax"] = sarimax_note
    payload["protocol"] = NOTEBOOK_TREE_PROTOCOL
    payload["artifacts"] = {
        "predictions": str(OUT / "predictions.csv"),
        "metrics": str(OUT / "metrics_by_model_horizon.csv"),
        "manifest": str(OUT / "manifest.json"),
    }
    (OUT / "experiment_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    test = by_model[by_model["slice"].eq("test")][["model", "horizon", "mae", "rmse", "mape", "pinball", "picp", "interval_width", "n_scored"]]
    print(test.to_string(index=False), flush=True)
    print(f"wrote {OUT / 'experiment_status.json'}", flush=True)


if __name__ == "__main__":
    main()
