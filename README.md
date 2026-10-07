# Member 2 — Hybrid Temporal Forecasting Engine

IT23415836 · Garusingarachchi Y.B · J26-DS-322

This branch is the forecast only. Data fusion, crop viability, and the season simulator stay on the team branch. Trained models are not in this branch. Training writes them to `outputs/`, which is not committed.

## Folders

| Folder | What it is |
| --- | --- |
| `src/htfe/data` | Prices, weather, macros, finished-season extent, and the monthly cultivation loader |
| `src/htfe/features` | Lags, season, and feature sets A and B |
| `src/htfe/models` | SARIMAX, LightGBM, XGBoost, and conformal intervals |
| `src/htfe/training` | Experiment A, Experiment B, and the runtime cultivation update |
| `src/htfe/api` | Forecast service |
| `app` | Streamlit studio |
| `data/member1` | Drop folder for Member 1's files |
| `tests` | As-of cultivation check and the fallback data load |
| `outputs` | Created when you train |

## Member 1's files

Put these in `data/member1/` when they are ready:

- `harti_prices_weekly_long.csv`
- `weather_weekly_open_meteo.csv`
- `macros_published.csv`
- `dcs_highland_seasonal.csv`
- `cultivation_monthly.csv` (optional; Experiment C stays off until this file is here)

If the first four files are not all in that folder, the current external data folder is used (`COBWEB_DATA_ROOT`, otherwise `D:\SLIIT\4th Year\Reaserch\data sets`). `cultivation_monthly.csv` is read from `data/member1/cultivation_monthly.csv` or from `processed/cultivation_monthly.csv`. The loader does not search the rest of the disk.

## Commands

From this folder, with the package on the path:

```powershell
$env:PYTHONPATH = "src"
python -m htfe.training.build_training_table
python -m htfe.training.run_experiment_a
python -m htfe.training.run_experiment_b
```

The studio reads the saved Experiment B scores. The dashboard scores a new achieved/target record with the separate runtime model after that model has been trained into `outputs/`.
