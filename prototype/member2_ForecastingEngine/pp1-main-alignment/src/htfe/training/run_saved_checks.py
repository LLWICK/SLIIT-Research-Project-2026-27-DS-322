"""Write saved-model and on-disk evaluation checks. Does not train.

Usage from the repo root:
  python -u -m htfe.training.run_saved_checks
"""
from __future__ import annotations

from htfe.training.evaluate import write_common_key_benchmark
from htfe.training.evaluate_saved_c import write_evaluation
from htfe.training.export_evaluation import export_this_run_studio
from htfe.training.reforecast_c import write_reforecast
from htfe.training.saved_inference import OUT, manifest_check, write_inference, write_json


def main() -> None:
    print("common-key benchmark from the saved Experiment B predictions", flush=True)
    benchmark = write_common_key_benchmark()
    print("scoring frozen Experiment C models", flush=True)
    inference = write_inference()
    evaluation = write_evaluation()
    print("dynamic same-target reforecast from the saved models", flush=True)
    walk = write_reforecast(OUT)
    studio = export_this_run_studio()
    shap_note = evaluation["shap_note"]
    only_notebook = [
        "Colab Drive mount and the notebook cell printouts.",
    ]
    if "stays only in the notebook" in shap_note or "did not run" in shap_note:
        only_notebook.insert(1, shap_note)
    check = manifest_check(only_notebook)
    check["common_key_horizons"] = benchmark["horizons"]
    check["studio_json"] = str(studio)
    write_json(OUT / "inference" / "manifest_check.json", {**check, "reforecast": walk, "inference": inference})
    if check["missing_required"]:
        missing = ", ".join(check["missing_required"])
        raise RuntimeError("Required artifacts are missing: " + missing)
    print(f"manifest required files present: {check['required_count']}", flush=True)
    print(f"reforecast checked rows: {walk.get('checked')}", flush=True)
    print(f"wrote {OUT / 'inference' / 'manifest_check.json'}", flush=True)


if __name__ == "__main__":
    main()
