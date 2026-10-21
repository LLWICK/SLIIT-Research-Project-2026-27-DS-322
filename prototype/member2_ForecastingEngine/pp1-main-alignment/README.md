# Hybrid Temporal Forecasting Engine

IT23415836, Garusingarachchi Y.B. Module J26-DS-322.

Member 2's forecast. It predicts the weekly wholesale price, in LKR per kg, for carrot, leeks, and tomato. The score markets are Dambulla, Colombo, Meegoda, and Nuwara Eliya. Horizons are 1, 2, 4, 8, and 12 weeks. Data fusion, crop viability, and the season simulator stay on the team branch.

## Experiments

Experiment A uses price lags and the season. Experiment B adds origin weather, diesel, the USD/LKR rate, inflation, and lagged finished-season extent. Experiment C is a separate matched-row test. B* uses the Experiment B features on rows that have a national cultivation report. C adds the as-of ratio of achieved hectares to target hectares. A ratio above 1 is kept. District rows stay in the audit and are not model inputs.

SARIMAX is order (1, 1, 1) x (1, 0, 0, 52). LightGBM and XGBoost are quantile models at 0.05, 0.50, and 0.95, with fixed hyperparameters and seed 42. LSTM was not run.

The split follows the forecast issue week. Training issue weeks run through 2022-12-31. Tree size is chosen by fitting through 2021 and early-stopping on 2022 target weeks, then the trees are refit through 2022. 2023 sets the interval adjustment only. The test is issue weeks in 2024 and 2025. Rows are not shuffled.

The dashboard forecast is the runtime model. It is not the accuracy table.

## Results

On the full 2024–2025 sample, the lowest 1-week MAE is SARIMAX at 60.96 LKR/kg (n=1200). The lowest 4-week MAE is XGBoost B at 107.56 (n=1164). LightGBM B at 4 weeks has MAE 110.14 and MAPE 43.96%. Calibrated coverage on that same slice is 83.42%. The 5–95% interval is meant to cover 90% of test prices. 83.42% is the measured share.

On the matched cultivation rows, LightGBM C has MAE 59.75 and B* has MAE 57.88 at 1 week (n=462). At 4 weeks, LightGBM C has MAE 80.04 and B* has MAE 83.58 (n=453).

## How to run

From the repository root, in Command Prompt. `set PYTHONPATH=src` has to be in that same window.

```
set PYTHONPATH=src
python -u -m htfe.training.build_training_table
python -u -m htfe.training.run_experiment_a
python -u -m htfe.training.run_experiment_b
python -u -m htfe.training.run_experiment_c
python -u -m htfe.training.export_evaluation
python -u -m streamlit run app/app.py
```

Experiment A reads `outputs/panel/training_table.parquet`, so the table build comes first. `python -u -m htfe.training.run_saved_checks` scores files already on disk. It does not train.

The four Member 1 files go in `data/member1/`: `harti_prices_weekly_long.csv`, `weather_weekly_open_meteo.csv`, `macros_published.csv`, and `dcs_highland_seasonal.csv`. If any of the four is missing, the loader uses the folder in `COBWEB_DATA_ROOT`. Experiment C reads `doa_cultivation_progress.csv` from `data/member1` or the processed data folder. That file keeps its name.

## Scores and models

The studio reads the JSON in `app/data`. Fitted tree models are joblib files under `outputs/experiment_a/models`, `outputs/experiment_b/models`, `outputs/experiment_c/models`, `outputs/experiment_c/models_b_star`, and `outputs/runtime_model/models`. SARIMAX forecasts stay in `outputs/experiment_a/predictions.csv`. `outputs/` is gitignored. A clone has the studio JSON in `app/data`. The fitted files appear only after training is run on that machine.

## Folders

- `src/htfe` — panel build, features, models, and the training scripts.
- `app` — Streamlit studio and the JSON it displays.
- `tests` — as-of cultivation checks, the split check, and saved-model checks.
- `data` — drop folder for Member 1's files.
- `outputs` — training output. Not committed.

`app/Notebooks` holds the written method.
