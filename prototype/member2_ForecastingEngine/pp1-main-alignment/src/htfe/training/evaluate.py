"""Score originally observed prices only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from htfe.config import OUTPUTS, STUDIO_DATA


def _block(y_true: np.ndarray, y_pred: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    error = y_pred - y_true
    ape = np.abs(error) / np.clip(y_true, 1e-6, None)
    out = {
        "n_scored": int(len(y_true)),
        "mae": round(float(np.mean(np.abs(error))), 2) if len(y_true) else None,
        "rmse": round(float(np.sqrt(np.mean(error**2))), 2) if len(y_true) else None,
        "mape": round(float(np.mean(ape) * 100), 2) if len(y_true) else None,
    }
    if np.isfinite(low).any() and np.isfinite(high).any():
        covered = (y_true >= low) & (y_true <= high)
        width = high - low
        residual = y_true - y_pred
        pinball = np.mean(np.where(residual >= 0, 0.5 * residual, -0.5 * residual))
        out["pinball"] = round(float(pinball), 2)
        out["picp"] = round(float(np.nanmean(covered) * 100), 2)
        out["interval_width"] = round(float(np.nanmean(width)), 2)
    else:
        out["pinball"] = None
        out["picp"] = None
        out["interval_width"] = None
    return out


def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame["y_true"].to_numpy(dtype=float),
        frame["y_pred"].to_numpy(dtype=float),
        frame["y_pred_q05"].to_numpy(dtype=float),
        frame["y_pred_q95"].to_numpy(dtype=float),
    )


def metrics_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = predictions[predictions["split"].eq("test")].copy()
    by_model = []
    by_series = []
    for (model, horizon, target_form), group in test.groupby(["model", "horizon", "target_form"], sort=True):
        block = _block(*_arrays(group))
        by_model.append({"model": model, "horizon": int(horizon), "target_form": target_form, "slice": "test", **block})
        shock = group[group["is_shock"].astype(bool)]
        if not shock.empty:
            by_model.append(
                {"model": model, "horizon": int(horizon), "target_form": target_form, "slice": "shock_2024q1", **_block(*_arrays(shock))}
            )
        for (unique_id, crop, market), series in group.groupby(["unique_id", "crop", "market"]):
            by_series.append(
                {
                    "model": model,
                    "horizon": int(horizon),
                    "target_form": target_form,
                    "unique_id": unique_id,
                    "crop": crop,
                    "market": market,
                    **_block(*_arrays(series)),
                }
            )
    return pd.DataFrame(by_model), pd.DataFrame(by_series)


def slice_metrics(predictions: pd.DataFrame, model: str, target_form: str = "logret") -> dict:
    """Headline metrics pooled over horizons, for the studio cards."""
    group = predictions[
        predictions["model"].eq(model) & predictions["split"].eq("test") & predictions["target_form"].eq(target_form)
    ]
    overall = _block(*_arrays(group)) if not group.empty else _block(np.array([]), np.array([]), np.array([]), np.array([]))
    by_crop = {}
    by_market = {}
    by_season = {}
    for crop, part in group.groupby("crop"):
        by_crop[str(crop)] = _block(*_arrays(part))
    for market, part in group.groupby("market"):
        by_market[str(market).lower().replace(" ", "_")] = _block(*_arrays(part))
    for season, part in group.groupby(group["season"].str.lower()):
        by_season[str(season)] = _block(*_arrays(part))
    return {"overall": overall, "by_crop": by_crop, "by_market": by_market, "by_season": by_season}


def normalize_key(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )


def extended_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Point metrics from Notebook 1's final benchmark cell. MAE and MAPE stay separate."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    absolute = np.abs(error)
    denominator = np.abs(y_true) + np.abs(y_pred)
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / sst) if sst > 0 else float("nan")
    return {
        "n_scored": int(len(y_true)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(absolute / np.clip(np.abs(y_true), 1e-6, None)) * 100),
        "smape": float(np.mean(2 * absolute / np.clip(denominator, 1e-6, None)) * 100),
        "wape": float(np.sum(absolute) / np.clip(np.sum(np.abs(y_true)), 1e-6, None) * 100),
        "mean_bias": float(np.mean(error)),
        "r2": r2,
    }


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def project_interval_around_median(
    q05: np.ndarray, q50: np.ndarray, q95: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the q50 point forecast and order the reported tails around it."""
    low = np.asarray(q05, dtype=float)
    mid = np.asarray(q50, dtype=float)
    high = np.asarray(q95, dtype=float)
    return np.minimum.reduce([low, mid, high]), mid, np.maximum.reduce([low, mid, high])


def interval_score_90(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    alpha = 0.10
    score = (high - low) + (2.0 / alpha) * (low - y) * (y < low) + (2.0 / alpha) * (y - high) * (y > high)
    return float(np.mean(score))


def assert_safe_evaluation_dir(path: Path) -> Path:
    """Benchmark files must not replace Experiment A/B metrics or the studio JSON."""
    resolved = path.resolve()
    blocked = [
        (OUTPUTS / "experiment_a").resolve(),
        (OUTPUTS / "experiment_b").resolve(),
        STUDIO_DATA.resolve(),
    ]
    if any(resolved == blocked_path or blocked_path in resolved.parents for blocked_path in blocked):
        raise RuntimeError(f"Refusing to write evaluation tables into {resolved}.")
    return resolved


def _round_frame(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].round(digits)
    return out


def mase_scale_from_training_table(table: pd.DataFrame) -> pd.DataFrame:
    """52-week MASE scale on observed prices through 2022-12-31.

    analysis_ready_panel.parquet is not on disk. Horizon-1 observed target prices
    are the weekly published prices, matched on the calendar week 52 weeks earlier.
    """
    work = table.loc[
        table["forecast_horizon_weeks"].eq(1) & table["target_is_observed"].astype(bool),
        ["crop_id", "market_id", "target_week", "target_price_lkr_kg"],
    ].copy()
    work["crop_key"] = normalize_key(work["crop_id"])
    work["market_key"] = normalize_key(work["market_id"])
    work["week_start"] = pd.to_datetime(work["target_week"], format="mixed").dt.normalize()
    work["observed_price"] = pd.to_numeric(work["target_price_lkr_kg"], errors="coerce")
    work = work.dropna(subset=["observed_price"]).drop_duplicates(["crop_key", "market_key", "week_start"])
    lagged = work[["crop_key", "market_key", "week_start", "observed_price"]].rename(
        columns={"week_start": "lag_week", "observed_price": "seasonal_lag_52_train"}
    )
    work["lag_week"] = work["week_start"] - pd.Timedelta(weeks=52)
    merged = work.merge(lagged, on=["crop_key", "market_key", "lag_week"], how="left")
    scale_source = merged[
        merged["week_start"].le(pd.Timestamp("2022-12-31"))
        & merged["observed_price"].notna()
        & merged["seasonal_lag_52_train"].notna()
    ].copy()
    scale_source["seasonal_abs_difference"] = (scale_source["observed_price"] - scale_source["seasonal_lag_52_train"]).abs()
    scale = (
        scale_source.groupby(["crop_key", "market_key"], as_index=False)["seasonal_abs_difference"]
        .mean()
        .rename(columns={"seasonal_abs_difference": "mase_scale_52w"})
    )
    if scale.empty or bool(scale["mase_scale_52w"].isna().any()) or bool(scale["mase_scale_52w"].le(0).any()):
        raise RuntimeError("Invalid 52-week MASE scale from the training table.")
    return scale


def _mase(y_true: pd.Series, y_pred: pd.Series, scale: pd.Series) -> float:
    return float(np.mean(np.abs(y_pred.to_numpy(dtype=float) - y_true.to_numpy(dtype=float)) / scale.to_numpy(dtype=float)))


def write_common_key_benchmark(
    predictions: pd.DataFrame | None = None,
    training: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Identical-key benchmark for LightGBM, XGBoost, SARIMAX, and last-value naive.

    Reads the saved Experiment B predictions. Does not refit and does not rewrite
    outputs/experiment_a, outputs/experiment_b, or app/data.
    """
    from htfe.training.evaluate_saved_c import moving_block_ci

    destination = assert_safe_evaluation_dir(out_dir or (OUTPUTS / "evaluation"))
    destination.mkdir(parents=True, exist_ok=True)
    if predictions is None:
        predictions = pd.read_csv(OUTPUTS / "experiment_b" / "predictions.csv")
    if training is None:
        from htfe.config import built_file

        training = pd.read_parquet(built_file("training_table.parquet"))
    pred = predictions.copy()
    for column in ("forecast_issue_week", "target_week"):
        pred[column] = pd.to_datetime(pred[column], format="mixed").dt.normalize()
    pred["horizon"] = pd.to_numeric(pred["horizon"], errors="raise").astype(int)
    pred["model"] = pred["model"].astype(str).str.strip().str.lower()
    pred["split"] = pred["split"].astype(str).str.strip().str.lower()
    pred["crop_key"] = normalize_key(pred["crop"])
    pred["market_key"] = normalize_key(pred["market"])
    expected_target = pred["forecast_issue_week"] + pd.to_timedelta(pred["horizon"] * 7, unit="D")
    if not bool(expected_target.eq(pred["target_week"]).all()):
        raise RuntimeError("Prediction target weeks are not horizon times seven days after the issue week.")
    table = training.copy()
    table["forecast_date"] = pd.to_datetime(table["forecast_date"], format="mixed").dt.normalize()
    table["crop_key"] = normalize_key(table["crop_id"])
    table["market_key"] = normalize_key(table["market_id"])
    scale = mase_scale_from_training_table(table)
    test = pred[pred["split"].eq("test")].copy()
    if not set(test["forecast_issue_week"].dt.year.unique()).issubset({2024, 2025}):
        raise RuntimeError("Held-out rows contain issue years outside 2024-2025.")
    key = ["crop_key", "market_key", "forecast_issue_week", "target_week", "horizon"]
    core = ["lightgbm", "xgboost", "sarimax"]
    rows: list[dict] = []
    seasonal_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    key_counts: list[dict] = []
    common_keys: list[pd.DataFrame] = []
    for horizon in sorted(test["horizon"].unique()):
        block = test[test["horizon"].eq(horizon)].copy()
        key_sets = {}
        for model in core:
            part = block[block["model"].eq(model)]
            if part.duplicated(key).any():
                raise RuntimeError(f"Duplicate test keys for {model}, horizon {horizon}.")
            key_sets[model] = set(map(tuple, part[key].itertuples(index=False, name=None)))
        common = key_sets["lightgbm"] & key_sets["xgboost"] & key_sets["sarimax"]
        if not common:
            raise RuntimeError(f"No common held-out keys at horizon {horizon}.")
        common_frame = pd.DataFrame(list(common), columns=key)
        common_keys.append(common_frame)
        key_counts.append(
            {
                "horizon": int(horizon),
                "lightgbm_keys": len(key_sets["lightgbm"]),
                "xgboost_keys": len(key_sets["xgboost"]),
                "sarimax_keys": len(key_sets["sarimax"]),
                "common_keys": len(common),
            }
        )
        reference = table[table["forecast_horizon_weeks"].eq(horizon)][
            ["crop_key", "market_key", "forecast_date", "last_observed_price_lkr_kg", "price_lag_52w"]
        ].drop_duplicates(["crop_key", "market_key", "forecast_date"])
        truth = block[block["model"].eq("lightgbm")].merge(common_frame, on=key, how="inner")
        truth["forecast_date"] = truth["forecast_issue_week"]
        baseline = truth.merge(reference, on=["crop_key", "market_key", "forecast_date"], how="left", validate="many_to_one")
        baseline = baseline.merge(scale, on=["crop_key", "market_key"], how="left", validate="many_to_one")
        if bool(baseline["last_observed_price_lkr_kg"].isna().any()):
            raise RuntimeError(f"Persistence baseline is incomplete at horizon {horizon}.")
        if bool(baseline["mase_scale_52w"].isna().any()):
            raise RuntimeError(f"MASE scale is incomplete on common keys at horizon {horizon}.")
        model_parts = {}
        for model in core:
            part = block[block["model"].eq(model)].merge(common_frame, on=key, how="inner")
            part = part.merge(scale, on=["crop_key", "market_key"], how="left", validate="many_to_one")
            model_parts[model] = part
            metrics = extended_point_metrics(part["y_true"].to_numpy(dtype=float), part["y_pred"].to_numpy(dtype=float))
            rows.append({"model": model, "horizon": int(horizon), **metrics, "mase_52w": _mase(part["y_true"], part["y_pred"], part["mase_scale_52w"])})
        naive = extended_point_metrics(
            baseline["y_true"].to_numpy(dtype=float),
            baseline["last_observed_price_lkr_kg"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model": "naive_last",
                "horizon": int(horizon),
                **naive,
                "mase_52w": _mase(baseline["y_true"], baseline["last_observed_price_lkr_kg"], baseline["mase_scale_52w"]),
            }
        )
        seasonal_truth = baseline[baseline["price_lag_52w"].notna()].copy()
        if len(seasonal_truth):
            seasonal_keys = seasonal_truth[key].drop_duplicates()
            seasonal_metrics = extended_point_metrics(
                seasonal_truth["y_true"].to_numpy(dtype=float),
                seasonal_truth["price_lag_52w"].to_numpy(dtype=float),
            )
            share = len(seasonal_truth) / len(baseline) * 100
            seasonal_rows.append(
                {
                    "model": "seasonal_naive_52",
                    "horizon": int(horizon),
                    **seasonal_metrics,
                    "mase_52w": _mase(seasonal_truth["y_true"], seasonal_truth["price_lag_52w"], seasonal_truth["mase_scale_52w"]),
                    "seasonal_eligible_share_pct": share,
                }
            )
            for model in core:
                part = block[block["model"].eq(model)].merge(seasonal_keys, on=key, how="inner")
                part = part.merge(scale, on=["crop_key", "market_key"], how="left", validate="many_to_one")
                metrics = extended_point_metrics(part["y_true"].to_numpy(dtype=float), part["y_pred"].to_numpy(dtype=float))
                seasonal_rows.append(
                    {
                        "model": model,
                        "horizon": int(horizon),
                        **metrics,
                        "mase_52w": _mase(part["y_true"], part["y_pred"], part["mase_scale_52w"]),
                        "seasonal_eligible_share_pct": len(part) / len(baseline) * 100,
                    }
                )
        naive_pair = baseline[key + ["y_true", "last_observed_price_lkr_kg"]].copy()
        naive_pair["ae_naive"] = (naive_pair["last_observed_price_lkr_kg"] - naive_pair["y_true"]).abs()
        for index, model in enumerate(core):
            part = model_parts[model][key + ["y_true", "y_pred"]].copy()
            paired = part.merge(naive_pair[key + ["ae_naive"]], on=key, how="inner", validate="one_to_one")
            paired["ae_difference_model_minus_naive"] = (paired["y_pred"] - paired["y_true"]).abs() - paired["ae_naive"]
            weekly = (
                paired.groupby("target_week", as_index=False)["ae_difference_model_minus_naive"].mean().sort_values("target_week")
            )
            low, high = moving_block_ci(
                weekly["ae_difference_model_minus_naive"],
                block_length=4,
                reps=2000,
                seed=1000 + int(horizon) + index * 100,
            )
            bootstrap_rows.append(
                {
                    "model": model,
                    "horizon": int(horizon),
                    "n_matched_rows": int(len(paired)),
                    "n_target_weeks": int(len(weekly)),
                    "mean_mae_difference_model_minus_naive": float(paired["ae_difference_model_minus_naive"].mean()),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "ci_entirely_below_zero": bool(high < 0),
                    "ci_entirely_above_zero": bool(low > 0),
                }
            )
    benchmark = pd.DataFrame(rows)
    naive_mae = benchmark.loc[benchmark["model"].eq("naive_last"), ["horizon", "mae"]].rename(columns={"mae": "naive_mae"})
    benchmark = benchmark.merge(naive_mae, on="horizon", how="left", validate="many_to_one")
    benchmark["mae_skill_vs_naive_pct"] = (1.0 - benchmark["mae"] / benchmark["naive_mae"]) * 100
    benchmark = benchmark.drop(columns=["naive_mae"]).sort_values(["horizon", "mae", "model"]).reset_index(drop=True)
    seasonal = pd.DataFrame(seasonal_rows).sort_values(["horizon", "mae", "model"]).reset_index(drop=True)
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values(["horizon", "model"]).reset_index(drop=True)
    counts = pd.DataFrame(key_counts)
    best = (
        benchmark[benchmark["model"].isin(core)]
        .sort_values(["horizon", "mae", "rmse", "model"])
        .groupby("horizon", as_index=False)
        .first()
    )
    simple = benchmark[["model", "horizon", "n_scored", "mae", "rmse", "mape"]].copy()
    check = simple[simple["model"].isin(core + ["naive_last"])].pivot(index="horizon", columns="model", values="n_scored")
    if not bool(check.nunique(axis=1).eq(1).all()):
        raise RuntimeError("LightGBM, XGBoost, SARIMAX, and last-value naive are not on identical keys.")
    _round_frame(scale).to_csv(destination / "mase_scale_52w_by_series.csv", index=False)
    _round_frame(benchmark).to_csv(destination / "common_key_benchmark.csv", index=False)
    _round_frame(seasonal).to_csv(destination / "seasonal_naive_benchmark.csv", index=False)
    counts.to_csv(destination / "common_key_counts.csv", index=False)
    pd.concat(common_keys, ignore_index=True).drop_duplicates().to_csv(destination / "common_test_keys.csv", index=False)
    _round_frame(best).to_csv(destination / "best_model_by_horizon.csv", index=False)
    _round_frame(bootstrap).to_csv(destination / "model_vs_naive_block_bootstrap.csv", index=False)
    note = (
        "Computed from outputs/experiment_b/predictions.csv and the training table. "
        "These rows are the intersection of LightGBM, XGBoost, and SARIMAX test keys. "
        "They do not replace outputs/experiment_a or outputs/experiment_b metric files, "
        "and they do not replace the full-sample studio cards. "
        "MAE is LKR per kg. MAPE, sMAPE, and WAPE are percents. "
        "mae_skill_vs_naive_pct is the MAE reduction against the last observed price. "
        "mase_52w divides absolute error by the pre-2023 seasonal scale. "
        "The scale uses horizon-1 observed target prices because analysis_ready_panel.parquet is not on disk. "
        "A negative model-minus-naive difference means the fitted model has lower absolute error than persistence. "
        "Calibrated PICP is in outputs/experiment_b/metrics_by_model_horizon.csv. Report that measured share. "
        "The nominal interval is 90%."
    )
    (destination / "benchmark_note.txt").write_text(note, encoding="utf-8")
    return {
        "directory": str(destination),
        "horizons": int(counts["horizon"].nunique()) if len(counts) else 0,
        "common_keys": int(counts["common_keys"].sum()) if len(counts) else 0,
    }


if __name__ == "__main__":
    print(write_common_key_benchmark(), flush=True)
