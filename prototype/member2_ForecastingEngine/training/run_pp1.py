"""Train the progress-presentation models and write artefacts.

Usage (from member2_ForecastingEngine):
  python -m training.run_pp1
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from training import baselines, sarima_auto
from training.build_feature_frame import build as build_features
from training.config import ARTIFACTS, FEATURE_PATH
from training.evaluate import metrics_tables
from training.export_studio_json import export
from training.plot_pp1_figures import write_figures
from training.trees import predict_boosters


def _lib(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "not-installed"


def main() -> None:
    started = time.perf_counter()
    print("building feature frame")
    frame = build_features()
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FEATURE_PATH, index=False)
    print(f"feature rows={len(frame)}")

    print("baselines")
    parts = [baselines.predict(frame)]
    print("autoarima / sarimax walk-forward")
    sarima_rows, sarima_meta = sarima_auto.predict(frame)
    parts.append(sarima_rows)
    print(f"sarima rows={len(sarima_rows)} order={sarima_meta.get('order')}")

    print("lightgbm + xgboost log-return")
    tree_rows, tree_meta = predict_boosters(frame, "logret")
    parts.append(tree_rows)
    print("lightgbm ratio (median only)")
    ratio_rows, _ = predict_boosters(frame, "ratio")
    parts.append(ratio_rows)

    predictions = pd.concat([part for part in parts if part is not None and not part.empty], ignore_index=True)
    by_model, by_series = metrics_tables(predictions)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = ARTIFACTS / f"{stamp}_pp1"
    latest = ARTIFACTS / "latest"
    for folder in (run_dir, latest):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "figures").mkdir(exist_ok=True)
        (folder / "models").mkdir(exist_ok=True)

    predictions.to_csv(latest / "predictions.csv", index=False)
    by_model.to_csv(latest / "metrics_by_model_horizon.csv", index=False)
    by_series.to_csv(latest / "metrics_by_series.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    by_model.to_csv(run_dir / "metrics_by_model_horizon.csv", index=False)
    by_series.to_csv(run_dir / "metrics_by_series.csv", index=False)

    origin_map = json.loads(json.dumps(__import__("yaml").safe_load(
        (Path(__file__).resolve().parents[2] / "data_pipeline" / "maps" / "origin_map.yaml").read_text(encoding="utf-8")
    )))
    write_figures(frame, by_model, tree_meta["importance"], origin_map, latest / "figures")
    write_figures(frame, by_model, tree_meta["importance"], origin_map, run_dir / "figures")

    elapsed = round(time.perf_counter() - started, 1)
    meta = {
        "version": f"real-pp1-{stamp}",
        "seed": 42,
        "elapsed_s": elapsed,
        "sarima": sarima_meta,
        "python": platform.python_version(),
        "libraries": {
            "lightgbm": _lib("lightgbm"),
            "xgboost": _lib("xgboost"),
            "statsmodels": _lib("statsmodels"),
            "statsforecast": _lib("statsforecast"),
        },
    }
    (latest / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (latest / "models" / "sarima_order.json").write_text(json.dumps(sarima_meta, indent=2), encoding="utf-8")

    export(frame, predictions, tree_meta["curves"], tree_meta["importance"], meta)
    print(f"done in {elapsed}s")
    headline = by_model[(by_model["slice"] == "test") & (by_model["horizon"] == 1)]
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
