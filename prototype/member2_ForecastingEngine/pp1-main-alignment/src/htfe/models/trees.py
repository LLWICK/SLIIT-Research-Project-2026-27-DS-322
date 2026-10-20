"""Pooled LightGBM and XGBoost quantile models, one direct model per horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd

from htfe.features.frame import feature_columns
from htfe.training.config import CAL_END, TRAIN_END

QUANTILES = (0.05, 0.50, 0.95)
CATS = ("crop", "market", "disruption_flag")
TREE_PROTOCOL = "notebook_2022_early_stop"
NOTEBOOK_TREE_PROTOCOL = TREE_PROTOCOL
SELECTOR_TRAIN_END = pd.Timestamp("2021-12-31")
SELECTOR_VALID_START = pd.Timestamp("2022-01-01")
SELECTOR_VALID_END = pd.Timestamp("2022-12-31")


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


def selection_masks(
    frame: pd.DataFrame,
    target_column: str = "target_week",
    *,
    min_selector_train: int = 50,
    min_selector_validation: int = 20,
    min_final_train: int = 50,
    min_calibration: int = 20,
) -> dict:
    """2021 fit, 2022 early-stop, refit through 2022. 2023 stays out of selection.

    This is the Notebook 1 tree chronology used by the experiment entry points.
    """
    target_week = pd.to_datetime(frame[target_column], format="mixed").dt.normalize()
    split = frame["split"].astype(str)
    train_rows = split.eq("train")
    selector_train = train_rows & target_week.le(SELECTOR_TRAIN_END)
    selector_validation = train_rows & target_week.ge(SELECTOR_VALID_START) & target_week.le(SELECTOR_VALID_END)
    final_train = train_rows.copy()
    calibration = split.eq("calibrate")
    test = split.eq("test")
    if bool((selector_train & selector_validation).any()):
        raise RuntimeError("Internal selector train and validation rows overlap.")
    if bool((selector_validation & calibration).any()):
        raise RuntimeError("2023 calibration rows entered early-stopping validation.")
    if bool((final_train & calibration).any()):
        raise RuntimeError("Calibration rows entered final tree fitting.")
    if bool((final_train & test).any()):
        raise RuntimeError("Held-out test rows entered final tree fitting.")
    if int(selector_train.sum()) < min_selector_train:
        raise RuntimeError("Not enough rows before 2022 for selector training.")
    if int(selector_validation.sum()) < min_selector_validation:
        raise RuntimeError("Not enough 2022 rows for internal early stopping.")
    if int(final_train.sum()) < min_final_train:
        raise RuntimeError("Not enough rows for final pre-2023 model fitting.")
    if int(calibration.sum()) < min_calibration:
        raise RuntimeError("Not enough untouched 2023 rows for CQR calibration.")
    selector_train_max = target_week[selector_train].max()
    selector_validation_min = target_week[selector_validation].min()
    if (
        pd.notna(selector_train_max)
        and pd.notna(selector_validation_min)
        and selector_train_max >= selector_validation_min
    ):
        raise RuntimeError("Internal model-selection split is not chronological.")
    return {
        "selector_train": selector_train.to_numpy(),
        "selector_validation": selector_validation.to_numpy(),
        "final_train": final_train.to_numpy(),
        "calibration": calibration.to_numpy(),
        "test": test.to_numpy(),
        "selector_train_max_target": selector_train_max,
        "selector_validation_min_target": selector_validation_min,
        "selector_validation_max_target": target_week[selector_validation].max(),
        "final_train_max_target": target_week[final_train].max(),
    }


def _lgb_settings(alpha: float, params: dict | None) -> dict:
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
    settings.pop("early_stopping_rounds", None)
    return settings


def _xgb_settings(alpha: float, params: dict | None, *, early_stopping: bool) -> dict:
    settings = {
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": 42,
    }
    if early_stopping:
        settings["early_stopping_rounds"] = 50
    if params:
        settings.update({key: value for key, value in params.items() if key != "early_stopping_rounds" or early_stopping})
    settings["objective"] = "reg:quantileerror"
    settings["quantile_alpha"] = alpha
    if not early_stopping:
        settings.pop("early_stopping_rounds", None)
    return settings


def _stamp_or_none(value) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def fit_selected_lightgbm(design: pd.DataFrame, y: np.ndarray, masks: dict, alpha: float, params: dict | None = None):
    """Select the tree count on 2022 targets, then refit through 2022-12-31.

    The 2023 rows are not passed to either fit. Callers keep them for qhat.
    """
    import lightgbm as lgb

    settings = _lgb_settings(alpha, params)
    cap = int(settings["n_estimators"])
    selector = lgb.LGBMRegressor(**settings)
    selector_train = np.flatnonzero(masks["selector_train"])
    selector_valid = np.flatnonzero(masks["selector_validation"])
    final_train = np.flatnonzero(masks["final_train"])
    selector.fit(
        design.iloc[selector_train],
        y[masks["selector_train"]],
        eval_set=[(design.iloc[selector_valid], y[masks["selector_validation"]])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    best = getattr(selector, "best_iteration_", None)
    n_estimators = cap if best is None or int(best) < 1 else int(best)
    final_settings = dict(settings)
    final_settings["n_estimators"] = n_estimators
    model = lgb.LGBMRegressor(**final_settings)
    model.fit(design.iloc[final_train], y[masks["final_train"]])
    record = {
        "family": "lightgbm",
        "protocol": NOTEBOOK_TREE_PROTOCOL,
        "quantile": float(alpha),
        "n_estimators": n_estimators,
        "selector_best_iteration": None if best is None else int(best),
        "selector_train_rows": int(len(selector_train)),
        "selector_validation_rows": int(len(selector_valid)),
        "final_train_rows": int(len(final_train)),
        "calibration_rows_unused": int(np.asarray(masks["calibration"]).sum()),
        "selector_train_target_end": _stamp_or_none(masks["selector_train_max_target"]),
        "selector_validation_target_start": _stamp_or_none(masks["selector_validation_min_target"]),
        "selector_validation_target_end": _stamp_or_none(masks["selector_validation_max_target"]),
        "final_train_target_end": _stamp_or_none(masks["final_train_max_target"]),
    }
    return model, record


def fit_selected_xgboost(design: pd.DataFrame, y: np.ndarray, masks: dict, alpha: float, params: dict | None = None):
    """Same 2022 early-stop then pre-2023 refit. XGBoost's best_iteration is zero-based."""
    import xgboost as xgb

    selector_settings = _xgb_settings(alpha, params, early_stopping=True)
    cap = int(selector_settings["n_estimators"])
    selector = xgb.XGBRegressor(**selector_settings)
    selector_train = np.flatnonzero(masks["selector_train"])
    selector_valid = np.flatnonzero(masks["selector_validation"])
    final_train = np.flatnonzero(masks["final_train"])
    selector.fit(
        design.iloc[selector_train],
        y[masks["selector_train"]],
        eval_set=[(design.iloc[selector_valid], y[masks["selector_validation"]])],
        verbose=False,
    )
    best = getattr(selector, "best_iteration", None)
    n_estimators = cap if best is None else int(best) + 1
    final_settings = _xgb_settings(alpha, params, early_stopping=False)
    final_settings["n_estimators"] = n_estimators
    model = xgb.XGBRegressor(**final_settings)
    model.fit(design.iloc[final_train], y[masks["final_train"]], verbose=False)
    record = {
        "family": "xgboost",
        "protocol": NOTEBOOK_TREE_PROTOCOL,
        "quantile": float(alpha),
        "n_estimators": n_estimators,
        "selector_best_iteration_zero_based": None if best is None else int(best),
        "selector_train_rows": int(len(selector_train)),
        "selector_validation_rows": int(len(selector_valid)),
        "final_train_rows": int(len(final_train)),
        "calibration_rows_unused": int(np.asarray(masks["calibration"]).sum()),
        "selector_train_target_end": _stamp_or_none(masks["selector_train_max_target"]),
        "selector_validation_target_start": _stamp_or_none(masks["selector_validation_min_target"]),
        "selector_validation_target_end": _stamp_or_none(masks["selector_validation_max_target"]),
        "final_train_target_end": _stamp_or_none(masks["final_train_max_target"]),
    }
    return model, record
