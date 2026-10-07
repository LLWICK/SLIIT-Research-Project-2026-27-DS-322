"""Write htfe-studio JSON from real artefacts, keeping the keys the screens read."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from htfe.training.config import EVAL_MARKETS, STUDIO_DATA
from htfe.training.evaluate import slice_metrics

from htfe.config import origin_map_path
MAP_PATH = origin_map_path()


def _dump(name: str, payload) -> None:
    path = STUDIO_DATA / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    raise TypeError(type(value))


def _metric_row(model: str, model_id: str, backend: str, metrics: dict) -> dict:
    overall = metrics["overall"]
    return {
        "model": model,
        "id": model_id,
        "n_scored": overall.get("n_scored") or 0,
        "mae": overall.get("mae"),
        "rmse": overall.get("rmse"),
        "mape": overall.get("mape"),
        "pinball": overall.get("pinball"),
        "picp": overall.get("picp"),
        "interval_width": overall.get("interval_width"),
        "backend": backend,
    }


def export(frame: pd.DataFrame, predictions: pd.DataFrame, curves: dict, importance: dict[str, float], meta: dict) -> None:
    STUDIO_DATA.mkdir(parents=True, exist_ok=True)
    backup = STUDIO_DATA.parent / "data_demo_backup"
    if not backup.exists():
        shutil.copytree(STUDIO_DATA, backup)

    test = predictions[predictions["split"].eq("test")]
    headline = {
        "naive_last": slice_metrics(predictions, "naive_last", "level"),
        "seasonal_naive_52": slice_metrics(predictions, "seasonal_naive_52", "level"),
        "sarimax": slice_metrics(predictions, "sarimax", "level"),
        "lightgbm": slice_metrics(predictions, "lightgbm", "logret"),
        "xgboost": slice_metrics(predictions, "xgboost", "logret"),
    }
    comparison = [
        _metric_row("Last-value naive", "naive_last", "naive_last", headline["naive_last"]),
        _metric_row("Seasonal naive (52)", "seasonal_naive_52", "seasonal_naive_52", headline["seasonal_naive_52"]),
        _metric_row("SARIMAX (1,1,1)(1,0,0,52)", "sarima", "sarimax", headline["sarimax"]),
        _metric_row("LightGBM quantile", "lightgbm", "lightgbm", headline["lightgbm"]),
        _metric_row("XGBoost quantile", "xgboost", "xgboost", headline["xgboost"]),
        {
            "model": "LSTM-style MLP (not run)",
            "id": "lstm",
            "n_scored": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "pinball": None,
            "picp": None,
            "interval_width": None,
            "backend": "not_run_pp1",
        },
    ]

    def _model_block(key: str, backend: str, curve_key: str | None) -> dict:
        return {
            "backend": backend,
            "qhat": None,
            "note": "Raw 5/95 quantiles. Conformal calibration is not in this run." if key == "lightgbm" else None,
            "metrics": headline.get(key) or headline.get("autoarima") or {"overall": {}, "by_crop": {}, "by_market": {}, "by_season": {}},
            "train_curve": (curves.get(curve_key) if curve_key else []) or [],
        }

    models = {
        "lightgbm": _model_block("lightgbm", "lightgbm", "lightgbm"),
        "xgboost": _model_block("xgboost", "xgboost", "xgboost"),
        "sarima": _model_block("sarimax", "sarimax", None),
        "lstm": {
            "backend": "not_run_pp1",
            "qhat": None,
            "metrics": {"overall": {}, "by_crop": {}, "by_market": {}, "by_season": {}},
            "train_curve": [],
        },
    }

    lgb = test[(test["model"] == "lightgbm") & (test["horizon"] == 1) & (test["target_form"] == "logret")].copy()
    pred_rows = []
    for _, row in lgb.sort_values(["crop", "market", "target_week"]).iterrows():
        pred_rows.append(
            {
                "crop": row["crop"],
                "market": str(row["market"]).lower().replace(" ", "_"),
                "week_start": pd.Timestamp(row["target_week"]).strftime("%Y-%m-%d"),
                "y_true": row["y_true"],
                "point": row["y_pred"],
                "lower": row["y_pred_q05"],
                "upper": row["y_pred_q95"],
                "q05": row["y_pred_q05"],
                "q50": row["y_pred"],
                "q95": row["y_pred_q95"],
                "season": str(row["season"]).lower(),
                "disruption_flag": False,
                "cultivation_intensity": None if pd.isna(row.get("cultivation_intensity")) else float(row["cultivation_intensity"]),
            }
        )

    live = []
    for (crop, market), group in lgb.groupby(["crop", "market"]):
        last = group.sort_values("week_start").iloc[-1]
        intensity = last.get("cultivation_intensity")
        live.append(
            {
                "crop": crop,
                "market": str(market).lower().replace(" ", "_"),
                "forecast_week": pd.Timestamp(last["target_week"]).strftime("%Y-%m-%d"),
                "cultivation_intensity": 1.0 if pd.isna(intensity) else float(intensity),
                "predicted_price": float(last["y_pred"]),
                "lower_price": float(last["y_pred_q05"]),
                "upper_price": float(last["y_pred_q95"]),
                "coverage_level": 0.9,
                "commitment_source": "historical_dcs_proxy",
                "model_version": meta.get("version", "real-pp1"),
            }
        )

    eval_panel = frame[frame["market"].isin(EVAL_MARKETS)].sort_values("week_start")
    series = []
    for (crop, market), group in eval_panel.groupby(["crop", "market"]):
        points = []
        for _, row in group.iterrows():
            points.append(
                {
                    "week": pd.Timestamp(row["week_start"]).strftime("%Y-%m-%d"),
                    "price": None if not bool(row["is_observed"]) else float(row["price"]),
                    "intensity": None if pd.isna(row["cultivation_intensity"]) else float(row["cultivation_intensity"]),
                    "season": str(row["season"]).lower(),
                    "split": "calibration" if row["split"] == "calibrate" else row["split"],
                    "disruption": row["disruption_flag"] not in {"none", ""},
                }
            )
        series.append({"crop": crop, "market": str(market).lower().replace(" ", "_"), "points": points})

    by_series = []
    for (crop, market), group in eval_panel.groupby(["crop", "market"]):
        by_series.append(
            {
                "crop": crop,
                "market": str(market).lower().replace(" ", "_"),
                "rows": int(len(group)),
                "observed": int(group["is_observed"].sum()),
                "start": pd.Timestamp(group["week_start"].min()).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(group["week_start"].max()).strftime("%Y-%m-%d"),
                "coverage_pct": round(float(group["is_observed"].mean() * 100), 1),
            }
        )

    origin_raw = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    origin_map = {}
    for crop, markets in origin_raw.items():
        origin_map[crop] = {}
        for market, spec in markets.items():
            if market not in EVAL_MARKETS:
                continue
            origin_map[crop][str(market).lower().replace(" ", "_")] = [
                str(district).lower().replace(" ", "_") for district in spec.get("supply_districts") or []
            ]

    ranked = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:12]
    peak = ranked[0][1] if ranked else 1
    importance_rows = [
        {"feature": name, "importance": round(value / peak, 3) if peak else 0, "std": 0.0} for name, value in ranked
    ]

    split_rows = {}
    for name, label in (("train", "train"), ("calibrate", "calibration"), ("test", "test")):
        part = eval_panel[eval_panel["split"].eq(name)]
        split_rows[label] = {
            "n": int(len(part)),
            "start": pd.Timestamp(part["week_start"].min()).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(part["week_start"].max()).strftime("%Y-%m-%d"),
        }

    cobweb = _cobweb(eval_panel)
    dictionary = [
        {"column": "crop", "dtype": "str", "role": "key", "notes": "carrot / leeks / tomato"},
        {"column": "market", "dtype": "str", "role": "key", "notes": "Wholesale observation market"},
        {"column": "week_start", "dtype": "date", "role": "key", "notes": "Monday on or after 1 January starts week 1"},
        {"column": "price", "dtype": "float LKR/kg", "role": "target", "notes": "Null when HARTI did not publish a price"},
        {"column": "is_observed", "dtype": "bool", "role": "quality", "notes": "Metrics use this filter"},
        {"column": "cultivation_intensity", "dtype": "float", "role": "feature", "notes": "Lagged DCS extent / 2015–2019 benchmark"},
        {"column": "commitment_source", "dtype": "str", "role": "feature", "notes": "historical_dcs_proxy"},
    ]

    _dump(
        "meta",
        {
            "project": "J26-DS-322",
            "member": "Garusingarachchi Y.B · IT23415836 · Member 2",
            "engine": "Hybrid Temporal Forecasting Engine",
            "version": meta.get("version", "real-pp1"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": meta.get("seed", 42),
            "confidence_level": 0.9,
            "commitment_source": "historical_dcs_proxy",
            "panel_grain": "crop × market × week",
            "elapsed_s": meta.get("elapsed_s"),
            "n_panel": int(eval_panel["is_observed"].sum()),
            "n_scored_test": int(headline["lightgbm"]["overall"].get("n_scored") or 0),
            "interval_note": "Raw LightGBM quantiles. Not conformal.",
            "backends": {"lightgbm": "lightgbm", "xgboost": "xgboost", "sarima": "sarimax"},
            "libraries": meta.get("libraries") or {},
        },
    )
    _dump(
        "coverage",
        {
            "by_series": by_series,
            "exclusions": [
                {"item": "Unobserved weeks", "reason": "Gaps stay blank. Long gaps are not interpolated, and every metric uses observed prices only."},
                {"item": "Retail series", "reason": "The scored panel is wholesale weekly prices."},
                {"item": "Potato, red onion, local big onion", "reason": "Not in the HARTI workbook, or too incomplete to model."},
                {"item": "Live commitments", "reason": "Cultivation intensity is a lagged Census extent proxy, not a HARTI registration feed."},
            ],
        },
    )
    _dump("split", split_rows)
    _dump("data_dictionary", dictionary)
    _dump("origin_map", origin_map)
    _dump("series", series)
    _dump("cobweb", cobweb)
    _dump("comparison", comparison)
    _dump(
        "ablation",
        {
            "status": "not_run",
            "note": "Feature ablation and the week-2/5/8/12 re-forecast are later work. This screen is not using synthetic scores.",
        },
    )
    _dump("commitment_weeks", [])
    _dump("importance", importance_rows)
    _dump("live_forecast", live)
    _dump("models", models)
    _dump("predictions", {"lightgbm": pred_rows})


def _cobweb(panel: pd.DataFrame) -> list[dict]:
    work = panel[panel["is_observed"]].copy()
    work["season_year"] = work["year"]
    maha_early = (work["season"] == "Maha") & (work["week"] <= 13)
    work.loc[maha_early, "season_year"] = work.loc[maha_early, "year"] - 1
    grouped = (
        work.groupby(["crop", "market", "season", "season_year"], as_index=False)
        .agg(price=("price", "mean"), intensity=("cultivation_intensity", "median"))
        .sort_values(["crop", "market", "season_year", "season"])
    )
    rows = []
    for (crop, market), group in grouped.groupby(["crop", "market"]):
        group = group.reset_index(drop=True)
        for index in range(len(group) - 1):
            current = group.iloc[index]
            nxt = group.iloc[index + 1]
            if pd.isna(current["intensity"]) or pd.isna(nxt["price"]):
                continue
            rows.append(
                {
                    "crop": crop,
                    "market": str(market).lower().replace(" ", "_"),
                    "season_t": f"{str(current['season']).lower()}_{int(current['season_year'])}",
                    "intensity_t": round(float(current["intensity"]), 3),
                    "price_t1": round(float(nxt["price"]), 2),
                    "season_t1": f"{str(nxt['season']).lower()}_{int(nxt['season_year'])}",
                }
            )
    return rows
