"""Fit methodology models: Optuna, A/B/C LightGBM, XGBoost, split CQR, SHAP."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from htfe.training.config import EVAL_MARKETS, HORIZONS
from htfe.training.evaluate import _block
from htfe.models.trees import QUANTILES, _design, _eligible, _fit_lgbm, _fit_xgb, _reconstruct

PINBALL_ALPHA = 0.5


def pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = PINBALL_ALPHA) -> float:
    residual = y_true - y_pred
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)
    return float(np.mean(loss))


def _xy(frame: pd.DataFrame, columns: list[str], horizon: int):
    subset = _eligible(frame, horizon, f"y_logret_h{horizon}")
    design, names = _design(subset, columns)
    design = design.copy()
    for source, name in (("crop", "crop_code"), ("market", "market_code")):
        codes = subset[source].astype("category").cat.codes.astype(float)
        design[name] = codes.to_numpy()
        if name not in names:
            names.append(name)
    y = subset[f"y_logret_h{horizon}"].to_numpy(dtype=float)
    train = subset["split"].eq("train").to_numpy()
    cal = subset["split"].eq("calibrate").to_numpy()
    return subset, design, names, y, train, cal


def tune_lightgbm(frame: pd.DataFrame, columns: list[str], n_trials: int = 25) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    subset, design, _names, y, train, cal = _xy(frame, columns, 4)
    x_train, y_train = design.iloc[np.flatnonzero(train)], y[train]
    x_cal, y_cal = design.iloc[np.flatnonzero(cal)], y[cal]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model, _curve = _fit_lgbm(x_train, y_train, x_cal, y_cal, PINBALL_ALPHA, params)
        return pinball(y_cal, np.asarray(model.predict(x_cal), dtype=float))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)


def tune_xgboost(frame: pd.DataFrame, columns: list[str], n_trials: int = 15) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    subset, design, _names, y, train, cal = _xy(frame, columns, 4)
    x_train, y_train = design.iloc[np.flatnonzero(train)], y[train]
    x_cal, y_cal = design.iloc[np.flatnonzero(cal)], y[cal]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = _fit_xgb(x_train, y_train, x_cal, y_cal, PINBALL_ALPHA, params)
        return pinball(y_cal, np.asarray(model.predict(x_cal), dtype=float))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)


def fit_lightgbm(
    frame: pd.DataFrame,
    columns: list[str],
    params: dict,
    feature_set: str,
    model_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Quantile LightGBM for every horizon. Saves boosters when model_dir is set."""
    import joblib

    rows: list[dict] = []
    live_vectors: list[dict] = []
    importance: dict[str, float] = {}
    curve: list[dict] = []
    saved: dict[str, str] = {}
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    for horizon in HORIZONS:
        subset, design, names, y, train, cal = _xy(frame, columns, horizon)
        if train.sum() < 50 or cal.sum() < 20:
            continue
        x_train, y_train = design.iloc[np.flatnonzero(train)], y[train]
        x_cal, y_cal = design.iloc[np.flatnonzero(cal)], y[cal]
        score_mask = (subset["market"].isin(EVAL_MARKETS) & subset["split"].isin(["calibrate", "test"])).to_numpy()
        score = subset.iloc[np.flatnonzero(score_mask)]
        x_score = design.iloc[np.flatnonzero(score_mask)]
        fitted = {}
        for alpha, label in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95")):
            model, this_curve = _fit_lgbm(x_train, y_train, x_cal, y_cal, alpha, params)
            fitted[label] = model
            if alpha == 0.50 and horizon == 4:
                curve = this_curve
                gains = dict(zip(names, model.feature_importances_.tolist(), strict=False))
                importance = {name: float(value) for name, value in gains.items()}
            if model_dir is not None:
                path = model_dir / f"lightgbm_{feature_set}_h{horizon}_{label}.joblib"
                joblib.dump(
                    {"model": model, "columns": names, "horizon": horizon, "alpha": alpha, "feature_set": feature_set},
                    path,
                )
                saved[f"h{horizon}_{label}"] = str(path)

        def _price(key: str) -> np.ndarray:
            raw = fitted[key].predict(x_score)
            return _reconstruct(score["price"], np.asarray(raw, dtype=float), "logret", score["recent_median_4"])

        point, low, high = _price("q50"), _price("q05"), _price("q95")
        for position, (_, row) in enumerate(score.iterrows()):
            rows.append(
                {
                    "unique_id": row["unique_id"],
                    "crop": row["crop"],
                    "market": row["market"],
                    "week_start": pd.Timestamp(row["week_start"]),
                    "target_week": pd.Timestamp(row[f"target_week_h{horizon}"]),
                    "horizon": horizon,
                    "y_true": float(row[f"y_price_h{horizon}"]),
                    "split": row["split"],
                    "is_shock": bool(row["is_shock"]),
                    "season": row["season"],
                    "target_form": "logret",
                    "feature_set": feature_set,
                    "model": f"lightgbm_{feature_set}",
                    "y_pred": float(point[position]),
                    "y_pred_q05": float(low[position]),
                    "y_pred_q95": float(high[position]),
                    "cultivation_progress": None
                    if "cultivation_progress" not in row or pd.isna(row["cultivation_progress"])
                    else float(row["cultivation_progress"]),
                }
            )
        if horizon == 1:
            indexed = score.copy()
            indexed["_pos"] = np.arange(len(indexed))
            last_rows = indexed.sort_values("week_start").groupby(["crop", "market"]).tail(1)
            for _, row in last_rows.iterrows():
                position = int(row["_pos"])
                vector = {}
                for name in names:
                    value = x_score.iloc[position][name]
                    vector[name] = None if pd.isna(value) else float(value)
                live_vectors.append(
                    {
                        "crop": row["crop"],
                        "market": str(row["market"]).lower().replace(" ", "_"),
                        "origin_price": float(row["price"]),
                        "forecast_week": pd.Timestamp(row["target_week_h1"]).strftime("%Y-%m-%d"),
                        "features": vector,
                    }
                )
    return pd.DataFrame(rows), {"importance": importance, "curve": curve, "saved": saved, "live_vectors": live_vectors}


def fit_xgboost(frame: pd.DataFrame, columns: list[str], params: dict, feature_set: str) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in HORIZONS:
        subset, design, _names, y, train, cal = _xy(frame, columns, horizon)
        if train.sum() < 50 or cal.sum() < 20:
            continue
        x_train, y_train = design.iloc[np.flatnonzero(train)], y[train]
        x_cal, y_cal = design.iloc[np.flatnonzero(cal)], y[cal]
        score_mask = (subset["market"].isin(EVAL_MARKETS) & subset["split"].isin(["calibrate", "test"])).to_numpy()
        score = subset.iloc[np.flatnonzero(score_mask)]
        x_score = design.iloc[np.flatnonzero(score_mask)]
        fitted = {}
        for alpha, label in zip(QUANTILES, ("q05", "q50", "q95"), strict=False):
            fitted[label] = _fit_xgb(x_train, y_train, x_cal, y_cal, alpha, params)

        def _price(key: str, fitted=fitted, score=score, x_score=x_score) -> np.ndarray:
            raw = fitted[key].predict(x_score)
            return _reconstruct(score["price"], np.asarray(raw, dtype=float), "logret", score["recent_median_4"])

        point, low, high = _price("q50"), _price("q05"), _price("q95")
        for position, (_, row) in enumerate(score.iterrows()):
            rows.append(
                {
                    "unique_id": row["unique_id"],
                    "crop": row["crop"],
                    "market": row["market"],
                    "week_start": pd.Timestamp(row["week_start"]),
                    "target_week": pd.Timestamp(row[f"target_week_h{horizon}"]),
                    "horizon": horizon,
                    "y_true": float(row[f"y_price_h{horizon}"]),
                    "split": row["split"],
                    "is_shock": bool(row["is_shock"]),
                    "season": row["season"],
                    "target_form": "logret",
                    "feature_set": feature_set,
                    "model": "xgboost",
                    "y_pred": float(point[position]),
                    "y_pred_q05": float(low[position]),
                    "y_pred_q95": float(high[position]),
                }
            )
    return pd.DataFrame(rows)


def _conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample split-conformal quantile. alpha=0.1 targets about 90% coverage."""
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_split_cqr(predictions: pd.DataFrame, model: str) -> tuple[pd.DataFrame, dict]:
    """Romano et al. split CQR on the 2023 calibration block, applied to later weeks.

    MAPIE's ConformalizedQuantileRegressor refits its own estimator. These rows
    already come from the Optuna LightGBM quantile models, so the same conformity
    score is applied directly and the MAPIE version is recorded when importable.
    """
    out = predictions.copy()
    out["y_pred_q05_raw"] = out["y_pred_q05"]
    out["y_pred_q95_raw"] = out["y_pred_q95"]
    qhats = {}
    mapie_version = None
    try:
        import mapie

        mapie_version = getattr(mapie, "__version__", "installed")
    except Exception:
        mapie_version = None
    mask = out["model"].eq(model)
    for horizon, group_index in out[mask].groupby("horizon").groups.items():
        block = out.loc[group_index]
        cal = block[block["split"].eq("calibrate")]
        scores = np.maximum(cal["y_pred_q05"].to_numpy() - cal["y_true"].to_numpy(), cal["y_true"].to_numpy() - cal["y_pred_q95"].to_numpy())
        qhat = _conformal_quantile(scores, 0.1)
        qhats[str(int(horizon))] = round(qhat, 4)
        out.loc[group_index, "y_pred_q05"] = out.loc[group_index, "y_pred_q05"] - qhat
        out.loc[group_index, "y_pred_q95"] = out.loc[group_index, "y_pred_q95"] + qhat
    return out, {
        "method": "split_cqr_romano2019",
        "mapie_version": mapie_version,
        "mapie_note": (
            "MAPIE is installed, but ConformalizedQuantileRegressor refits an estimator and does not accept the three already-fit LightGBM quantile boosters. "
            "The conformity score MAPIE uses for CQR was applied to those boosters on the 2023 block."
            if mapie_version
            else "MAPIE was not importable. Split CQR was applied directly."
        ),
        "alpha": 0.1,
        "qhat_by_horizon": qhats,
        "qhat": qhats.get("1"),
        "limit": "Time-ordered prices are not exchangeable, so this is an empirical calibration, not a universal 90% guarantee.",
    }


def shap_importance(model_path: Path, frame: pd.DataFrame, columns: list[str], sample_size: int = 400) -> list[dict]:
    import joblib
    import shap

    bundle = joblib.load(model_path)
    model = bundle["model"]
    names = bundle["columns"]
    subset, design, _names, _y, _train, _cal = _xy(frame, columns, 4)
    test = (subset["split"].eq("test") & subset["market"].isin(EVAL_MARKETS)).to_numpy()
    x_test = design.iloc[np.flatnonzero(test)]
    if x_test.empty:
        return []
    if len(x_test) > sample_size:
        x_test = x_test.sample(sample_size, random_state=42)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_test)
    if isinstance(values, list):
        values = values[0]
    mean_abs = np.abs(np.asarray(values)).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    peak = float(mean_abs[order[0]]) if len(order) else 1.0
    rows = []
    for index in order[:12]:
        rows.append({"feature": names[index], "importance": round(float(mean_abs[index] / peak), 3) if peak else 0.0, "std": 0.0})
    return rows


def ablation_payload(predictions: pd.DataFrame, progress_in_model: bool, note: str) -> dict:
    payload: dict = {"status": "complete" if progress_in_model else "partial", "note": note, "by_horizon": {}}
    for arm in ("A", "B", "C"):
        model = f"lightgbm_{arm}"
        group = predictions[predictions["model"].eq(model) & predictions["split"].eq("test")]
        if group.empty:
            payload[arm] = None
            continue
        metrics = _block(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
            group["y_pred_q05"].to_numpy(dtype=float),
            group["y_pred_q95"].to_numpy(dtype=float),
        )
        payload[arm] = {"metrics": metrics, "feature_set": arm}
        payload["by_horizon"][arm] = []
        for horizon, part in group.groupby("horizon"):
            payload["by_horizon"][arm].append(
                {
                    "horizon": int(horizon),
                    **_block(
                        part["y_true"].to_numpy(dtype=float),
                        part["y_pred"].to_numpy(dtype=float),
                        part["y_pred_q05"].to_numpy(dtype=float),
                        part["y_pred_q95"].to_numpy(dtype=float),
                    ),
                }
            )
    return payload


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
