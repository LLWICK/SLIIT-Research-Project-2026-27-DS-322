"""Pooled LightGBM and XGBoost quantile models, one direct model per horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.build_feature_frame import feature_columns
from training.config import CAL_END, EVAL_MARKETS, HORIZONS, TRAIN_END

QUANTILES = (0.05, 0.50, 0.95)
CATS = ("crop", "market", "disruption_flag")


def _eligible(frame: pd.DataFrame, horizon: int, target_col: str) -> pd.DataFrame:
    target_week = pd.to_datetime(frame[f"target_week_h{horizon}"])
    origin = pd.to_datetime(frame["week_start"])
    keep = frame["is_observed"] & frame[target_col].notna() & frame["price"].gt(0)
    # Training labels must not land in 2023+. Calibration labels must not land in 2024+.
    leak_train = (frame["split"] == "train") & (target_week > TRAIN_END)
    leak_cal = (frame["split"] == "calibrate") & (target_week > CAL_END)
    keep = keep & ~leak_train & ~leak_cal & origin.notna()
    return frame.loc[keep].copy()


def _design(frame: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    numeric = feature_columns(frame) if columns is None else [name for name in columns if name in frame.columns]
    pieces = [frame[numeric].apply(pd.to_numeric, errors="coerce")]
    cat_names = []
    for name in CATS:
        codes = frame[name].astype("category")
        pieces.append(codes.cat.codes.rename(name).astype(float).where(codes.cat.codes >= 0))
        cat_names.append(name)
    design = pd.concat(pieces, axis=1)
    return design, numeric + cat_names


def _reconstruct(origin_price: pd.Series, prediction: np.ndarray, target_form: str, recent_median: pd.Series) -> np.ndarray:
    if target_form == "logret":
        price = origin_price.to_numpy(dtype=float) * np.exp(prediction)
    elif target_form == "ratio":
        price = recent_median.to_numpy(dtype=float) * prediction
    else:
        price = prediction
    return np.maximum(price, 0.01)


def _fit_lgbm(x_train, y_train, x_cal, y_cal, alpha: float, params: dict | None = None):
    import lightgbm as lgb

    settings = {
        "objective": "quantile",
        "alpha": alpha,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
        "verbosity": -1,
        "random_state": 42,
    }
    if params:
        settings.update(params)
    settings["objective"] = "quantile"
    settings["alpha"] = alpha
    model = lgb.LGBMRegressor(**settings)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model.fit(x_train, y_train, eval_set=[(x_cal, y_cal)], callbacks=callbacks)
    curve = []
    result = getattr(model, "evals_result_", None) or {}
    series = (result.get("valid_0") or {})
    metric = next(iter(series.values()), []) if series else []
    for index, value in enumerate(metric, start=1):
        curve.append({"iter": index, "mae": float(value)})
    return model, curve


def _fit_xgb(x_train, y_train, x_cal, y_cal, alpha: float, params: dict | None = None):
    import xgboost as xgb

    settings = {
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "n_jobs": 1,
        "random_state": 42,
    }
    if params:
        settings.update(params)
    settings["objective"] = "reg:quantileerror"
    settings["quantile_alpha"] = alpha
    model = xgb.XGBRegressor(**settings)
    model.fit(x_train, y_train, eval_set=[(x_cal, y_cal)], verbose=False)
    return model


def predict_boosters(frame: pd.DataFrame, target_form: str = "logret") -> tuple[pd.DataFrame, dict]:
    """Fit pooled trees and return prediction rows for evaluation markets only."""
    target_name = {"logret": "y_logret_h", "ratio": "y_ratio_h", "level": "y_price_h"}[target_form]
    model_names = {"logret": ("lightgbm", "xgboost"), "ratio": ("lightgbm_ratio",), "level": ("lightgbm_level",)}[target_form]
    rows: list[dict] = []
    curves: dict[str, list] = {name: [] for name in model_names}
    importance: dict[str, float] = {}

    for horizon in HORIZONS:
        subset = _eligible(frame, horizon, f"{target_name}{horizon}")
        if subset.empty:
            continue
        design, names = _design(subset)
        y = subset[f"{target_name}{horizon}"].to_numpy(dtype=float)
        train = subset["split"].eq("train").to_numpy()
        cal = subset["split"].eq("calibrate").to_numpy()
        if train.sum() < 50 or cal.sum() < 20:
            continue
        x_train, y_train = design.iloc[np.flatnonzero(train)], y[train]
        x_cal, y_cal = design.iloc[np.flatnonzero(cal)], y[cal]
        score_mask = (
            subset["market"].isin(EVAL_MARKETS) & subset["split"].isin(["calibrate", "test"])
        ).to_numpy()
        score = subset.iloc[np.flatnonzero(score_mask)]
        x_score = design.iloc[np.flatnonzero(score_mask)]

        fitted = {}
        if "lightgbm" in model_names or "lightgbm_ratio" in model_names or "lightgbm_level" in model_names:
            lgb_name = next(name for name in model_names if name.startswith("lightgbm"))
            for alpha, label in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95")):
                if target_form != "logret" and alpha != 0.50:
                    continue
                model, curve = _fit_lgbm(x_train, y_train, x_cal, y_cal, alpha)
                fitted[label] = model
                if alpha == 0.50:
                    curves[lgb_name] = curve
                    gains = dict(zip(names, model.feature_importances_.tolist(), strict=False))
                    for feature, gain in gains.items():
                        importance[feature] = importance.get(feature, 0.0) + float(gain)
        if "xgboost" in model_names:
            for alpha, label in ((0.05, "x05"), (0.50, "x50"), (0.95, "x95")):
                fitted[label] = _fit_xgb(x_train, y_train, x_cal, y_cal, alpha)

        def _price(key: str) -> np.ndarray:
            raw = fitted[key].predict(x_score)
            return _reconstruct(score["price"], np.asarray(raw, dtype=float), target_form, score["recent_median_4"])

        lgb_point = _price("q50") if "q50" in fitted else None
        xgb_point = _price("x50") if "x50" in fitted else None
        lgb_lo = _price("q05") if "q05" in fitted else None
        lgb_hi = _price("q95") if "q95" in fitted else None
        xgb_lo = _price("x05") if "x05" in fitted else None
        xgb_hi = _price("x95") if "x95" in fitted else None

        for position, (_, row) in enumerate(score.iterrows()):
            common = {
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
                "target_form": target_form,
                "cultivation_intensity": float(row["cultivation_intensity"]) if pd.notna(row["cultivation_intensity"]) else np.nan,
            }
            if lgb_point is not None:
                rows.append(
                    {
                        **common,
                        "model": "lightgbm" if target_form == "logret" else f"lightgbm_{target_form}",
                        "y_pred": float(lgb_point[position]),
                        "y_pred_q05": float(lgb_lo[position]) if lgb_lo is not None else np.nan,
                        "y_pred_q95": float(lgb_hi[position]) if lgb_hi is not None else np.nan,
                    }
                )
            if xgb_point is not None:
                rows.append(
                    {
                        **common,
                        "model": "xgboost",
                        "y_pred": float(xgb_point[position]),
                        "y_pred_q05": float(xgb_lo[position]) if xgb_lo is not None else np.nan,
                        "y_pred_q95": float(xgb_hi[position]) if xgb_hi is not None else np.nan,
                    }
                )
    return pd.DataFrame(rows), {"curves": curves, "importance": importance}
