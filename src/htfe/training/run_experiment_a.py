"""Train Experiment A on the primary training table.

Usage from prototype/member2_ForecastingEngine:
  python -u -m training.run_experiment_a
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from htfe.training.config import ARTIFACTS, CAL_END, EVAL_MARKETS, HORIZONS, PANEL_PATH, TRAIN_END
from htfe.training.coverage_audit import write_audit
from htfe.training.evaluate import metrics_tables
from htfe.features.sets import (
    PRICE_HISTORY,
    SEASON_FEATURES,
    feature_sets,
    historical_columns,
    macros_ready,
    progress_available,
)
from htfe.models.trees import _fit_lgbm, _fit_xgb

RANDOM_STATE = 42
LGB_TRIALS = 15
XGB_TRIALS = 8
EVAL_IDS = {market.lower().replace(" ", "_") for market in EVAL_MARKETS}
MARKET_NAME = {market.lower().replace(" ", "_"): market for market in EVAL_MARKETS}
OUT = ARTIFACTS / "experiment_a"
LATEST_PREDICTIONS = ARTIFACTS / "latest" / "predictions.csv"


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    residual = y_true - y_pred
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual)
    return float(np.mean(loss))


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


def _score_frame(score: pd.DataFrame, point: np.ndarray, low: np.ndarray, high: np.ndarray, model: str) -> pd.DataFrame:
    market = score["market_id"].map(MARKET_NAME).fillna(score["market_id"])
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


def _fit_horizon(subset: pd.DataFrame, columns: list[str], lgb_params: dict, xgb_params: dict) -> pd.DataFrame:
    design, _names = _design(subset, columns)
    y = subset["y_logret"].to_numpy(dtype=float)
    train = subset["split"].eq("train").to_numpy()
    cal = subset["split"].eq("calibrate").to_numpy()
    if int(train.sum()) < 50 or int(cal.sum()) < 20:
        raise RuntimeError("not enough train or calibration rows")
    x_train, y_train = design.iloc[np.flatnonzero(train)], y[train]
    x_cal, y_cal = design.iloc[np.flatnonzero(cal)], y[cal]
    score_mask = subset["market_id"].isin(EVAL_IDS) & subset["split"].isin(["calibrate", "test"])
    score = subset.loc[score_mask]
    x_score = design.loc[score_mask]
    origin = score["last_observed_price_lkr_kg"].to_numpy(dtype=float)

    def prices(model) -> np.ndarray:
        raw = np.asarray(model.predict(x_score), dtype=float)
        return np.maximum(origin * np.exp(raw), 0.01)

    lgb = {}
    for alpha, label in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95")):
        model, _curve = _fit_lgbm(x_train, y_train, x_cal, y_cal, alpha, lgb_params)
        lgb[label] = prices(model)
    xgb = {}
    for alpha, label in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95")):
        model = _fit_xgb(x_train, y_train, x_cal, y_cal, alpha, xgb_params)
        xgb[label] = prices(model)
    return pd.concat(
        [
            _score_frame(score, lgb["q50"], lgb["q05"], lgb["q95"], "lightgbm"),
            _score_frame(score, xgb["q50"], xgb["q05"], xgb["q95"], "xgboost"),
        ],
        ignore_index=True,
    )


def _tune(frame: pd.DataFrame, columns: list[str]) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    subset = _eligible(frame, 4)
    design, _names = _design(subset, columns)
    y = subset["y_logret"].to_numpy(dtype=float)
    train = subset["split"].eq("train").to_numpy()
    cal = subset["split"].eq("calibrate").to_numpy()
    x_train, y_train = design.iloc[np.flatnonzero(train)], y[train]
    x_cal, y_cal = design.iloc[np.flatnonzero(cal)], y[cal]
    y_price = subset.loc[subset["split"].eq("calibrate"), "target_price_lkr_kg"].to_numpy(dtype=float)
    origin = subset.loc[subset["split"].eq("calibrate"), "last_observed_price_lkr_kg"].to_numpy(dtype=float)

    def objective_lgb(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model, _curve = _fit_lgbm(x_train, y_train, x_cal, y_cal, 0.5, params)
        price = np.maximum(origin * np.exp(np.asarray(model.predict(x_cal), dtype=float)), 0.01)
        return _pinball(y_price, price)

    def objective_xgb(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = _fit_xgb(x_train, y_train, x_cal, y_cal, 0.5, params)
        price = np.maximum(origin * np.exp(np.asarray(model.predict(x_cal), dtype=float)), 0.01)
        return _pinball(y_price, price)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    lgb_study = optuna.create_study(direction="minimize", sampler=sampler)
    lgb_study.optimize(objective_lgb, n_trials=LGB_TRIALS, show_progress_bar=False)
    xgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    xgb_study.optimize(objective_xgb, n_trials=XGB_TRIALS, show_progress_bar=False)
    return {
        "lightgbm": dict(lgb_study.best_params),
        "xgboost": dict(xgb_study.best_params),
        "lightgbm_best_pinball": lgb_study.best_value,
        "xgboost_best_pinball": xgb_study.best_value,
        "lightgbm_trials": LGB_TRIALS,
        "xgboost_trials": XGB_TRIALS,
        "tuned_on": "horizon 4, q50 pinball on price, calibration issue weeks in 2023",
    }


def _reuse_sarimax(table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if not LATEST_PREDICTIONS.exists():
        return pd.DataFrame(), "artifacts/latest/predictions.csv is missing, and SARIMAX was not refit"
    predictions = pd.read_csv(LATEST_PREDICTIONS)
    sarimax = predictions[predictions["model"].eq("sarimax")].copy()
    if sarimax.empty:
        return pd.DataFrame(), "SARIMAX rows are missing from artifacts/latest/predictions.csv, and SARIMAX was not refit"
    sarimax["forecast_issue_week"] = pd.to_datetime(sarimax["week_start"]).astype("datetime64[ns]")
    sarimax["target_week"] = pd.to_datetime(sarimax["target_week"]).astype("datetime64[ns]")
    sarimax["horizon"] = sarimax["horizon"].astype(int)
    sarimax["crop"] = sarimax["crop"].astype(str)
    sarimax["market"] = sarimax["market"].astype(str)
    eligible = []
    for horizon in HORIZONS:
        part = _eligible(table, horizon)
        part = part[part["market_id"].isin(EVAL_IDS) & part["split"].isin(["calibrate", "test"])].copy()
        part["horizon"] = part["forecast_horizon_weeks"].astype(int)
        part["crop_id"] = part["crop_id"].astype(str)
        part["forecast_issue_week"] = pd.to_datetime(part["forecast_issue_week"]).astype("datetime64[ns]")
        eligible.append(part)
    keys = pd.concat(eligible, ignore_index=True)
    keys["market_id"] = keys["market_id"].astype(str)
    keys["market"] = keys["market_id"].map(MARKET_NAME)
    keys["target_week"] = pd.to_datetime(keys["target_week"]).astype("datetime64[ns]")
    merged = sarimax.merge(
        keys,
        left_on=["crop", "market", "forecast_issue_week", "horizon"],
        right_on=["crop_id", "market", "forecast_issue_week", "horizon"],
        how="inner",
        suffixes=("", "_table"),
    )
    if merged.empty:
        return pd.DataFrame(), "SARIMAX predictions do not match the training-table price target, and SARIMAX was not refit"
    if "is_shock_table" in merged.columns:
        merged["is_shock"] = merged["is_shock_table"]
    if "season_table" in merged.columns:
        merged["season"] = merged["season_table"]
    gap = (merged["y_true"] - merged["target_price_lkr_kg"]).abs()
    if float(gap.max()) > 1e-6:
        return pd.DataFrame(), "SARIMAX y_true does not match target_price_lkr_kg, and SARIMAX was not refit"
    week_gap = (pd.to_datetime(merged["target_week"]) - pd.to_datetime(merged["target_week_table"])).abs()
    if bool(week_gap.dt.days.gt(0).any()):
        return pd.DataFrame(), "SARIMAX target weeks do not match the training table, and SARIMAX was not refit"
    kept = _score_frame(
        merged,
        merged["y_pred"].to_numpy(dtype=float),
        np.full(len(merged), np.nan),
        np.full(len(merged), np.nan),
        "sarimax",
    )
    # _score_frame recomputes y_pred from arrays already in price space. Quantiles stay empty.
    return kept, f"reused {len(kept)} SARIMAX rows from artifacts/latest/predictions.csv; same price target and horizons"


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


def main() -> None:
    from htfe.config import built_file
    from htfe.training.build_training_table import main as rebuild

    print("rebuilding training table with forecast identity columns and as-of macros", flush=True)
    rebuild()
    table_path = built_file("training_table.parquet")
    table = pd.read_parquet(table_path)
    table["crop_id"] = pd.Categorical(table["crop_id"])
    table["market_id"] = pd.Categorical(table["market_id"])
    table["season"] = pd.Categorical(table["season"], categories=["Maha", "Yala"])
    panel = pd.read_parquet(PANEL_PATH)
    _assert_split(table)
    expected = list(PRICE_HISTORY) + list(SEASON_FEATURES)
    columns = historical_columns(table)
    if columns != expected:
        raise RuntimeError(f"Experiment A features are missing: {set(expected) - set(columns)}")
    sets = feature_sets(table)
    if sets["A"] != columns:
        raise RuntimeError("feature set A does not match the price-lag and season columns")
    print("writing coverage audit", flush=True)
    audit = write_audit(table, panel)
    ready, macro_reason = macros_ready(table)
    c_reason = (
        "Observed cultivation progress is non-null, but this task trains Experiment A only."
        if progress_available(table)
        else "observed cultivation data unavailable"
    )
    print(macro_reason, flush=True)
    print(c_reason, flush=True)
    print("optuna horizon 4", flush=True)
    tuned = _tune(table, columns)
    print(tuned, flush=True)
    parts = []
    for horizon in HORIZONS:
        print(f"fitting horizon {horizon}", flush=True)
        subset = _eligible(table, horizon)
        parts.append(_fit_horizon(subset, columns, tuned["lightgbm"], tuned["xgboost"]))
    sarimax, sarimax_note = _reuse_sarimax(table)
    print(sarimax_note, flush=True)
    if not sarimax.empty:
        parts.append(sarimax)
    predictions = pd.concat(parts, ignore_index=True)
    by_model, by_series = metrics_tables(predictions)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    by_model.to_csv(OUT / "metrics_by_model_horizon.csv", index=False)
    by_series.to_csv(OUT / "metrics_by_series.csv", index=False)
    manifest = {
        "experiment": "A",
        "description": "Price lags and seasonal features only. No weather, macros, cultivation progress, or lagged extent.",
        "features": columns,
        "identity_codes": ["crop_id", "market_id"],
        "season_code": "season",
        "label": "target_price_lkr_kg at target_week",
        "model_target": "log(target_price / last_observed_price), inverted back to LKR/kg",
        "quantiles": [0.05, 0.50, 0.95],
        "random_state": RANDOM_STATE,
        "split": {
            "train_issue_through": TRAIN_END,
            "calibrate": "2023",
            "test": "2024-2025",
            "shuffle": False,
        },
        "score_markets": list(EVAL_MARKETS),
        "train_pool_markets": ["Kandy", "Keppetipola", "Thambuttegama"],
        "score_rule": "last observed price and target price both originally observed",
        "optuna": tuned,
        "sarimax": sarimax_note,
        "feature_sets": {name: value if value is None else value for name, value in sets.items()},
        "coverage": audit,
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
    payload["artifacts"] = {
        "predictions": str(OUT / "predictions.csv"),
        "metrics": str(OUT / "metrics_by_model_horizon.csv"),
        "manifest": str(OUT / "manifest.json"),
        "coverage_md": audit["coverage_md"],
        "coverage_csv": audit["coverage_csv"],
        "weather_csv": audit["weather_csv"],
    }
    (OUT / "experiment_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    test = by_model[by_model["slice"].eq("test")][["model", "horizon", "mae", "rmse", "mape", "pinball", "picp", "interval_width", "n_scored"]]
    print(test.to_string(index=False), flush=True)
    print(f"wrote {OUT / 'experiment_status.json'}", flush=True)


if __name__ == "__main__":
    main()
