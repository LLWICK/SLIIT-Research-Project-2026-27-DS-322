# How the Member 2 project works

Read this from the top. Each section names the next file to open. The screens only display files that training already wrote. They do not fit models.

This checkout is Member 2 of J26-DS-322, IT23415836, Garusingarachchi Y.B. The forecast code is `src/htfe`. The studio is `app`. Fitted files sit in `outputs`, and git ignores that folder. A clone still has the studio JSON in `app/data`. The joblib models appear only after a training run on that machine.

Start with `README.md` if you want the short version of the same story. Then come back here.

## 1. The idea in one page

The project forecasts the weekly wholesale price, in LKR per kg, for three crops: carrot, leeks, and tomato. The score markets are Dambulla, Colombo, Meegoda, and Nuwara Eliya. The horizons are 1, 2, 4, 8, and 12 weeks.

The economic story is the Sri Lankan cobweb. A high price this season encourages more planting. More supply arrives later, and the price falls. A low price then leads to less planting, and the price rises again. Member 2's job is the price forecast that sits inside that story. Member 2 does not decide what to plant, and this branch does not run a season simulator.

Four people share the work:

- Member 1 supplies the data. The handoff is a weekly panel: HARTI wholesale prices, origin weather, published macros, and seasonal extent.
- Member 2 owns the forecast engine. That is this repository.
- Member 3 receives one forecast packet and would use it for crop viability. This branch does not start Member 3's app.
- Member 4 is the season simulator. This branch does not run it. The chain buttons for Member 3 and Member 4 only show a note.

Three extra wholesale markets, Kandy, Keppetipola, and Thambuttegama, can enter the tree fit. They are the train pool. Reported scores use the four score markets only.

Two kinds of model appear in the accuracy tables. SARIMAX is a seasonal time-series model on the price itself. LightGBM and XGBoost are tree models. Each tree predicts the log of (target price / last observed price), then the code turns that back into LKR per kg. There is one tree fit per horizon. That is direct multi-horizon forecasting: the 12-week model is not twelve steps of the 1-week model.

A third model exists only for the dashboard. It is the runtime model in `outputs/runtime_model`. It was trained with a labelled synthetic cultivation curve so a slider can change a progress column. That curve is not Experiment B and it is not Experiment C. Quote the dashboard price as the runtime model.

Open next: `src/htfe/config.py` for where files are found, then `src/htfe/data/build/scope.py` for crops, markets, horizons, and the year split.

## 2. The logic in the order it runs

### Two jobs, in speech

Say the work as two jobs.

1. Build the forecast table. One row is one crop, one market, one Monday, and one horizon. Every input on that row was already public on that Monday. The future wholesale price is the label.
2. Fit and score, and say which model is closest at each horizon. Closeness is MAE in LKR per kg on the 2024–2025 test. The closest model changes with the horizon. State one horizon at a time.

### What happens, in order

Sources arrive as CSV files. If all four contract files are in `data/member1`, the loader uses that folder. On this machine that folder holds only `.gitkeep`, so the loader uses `COBWEB_DATA_ROOT`. The default is `D:\SLIIT\4th Year\Reaserch\data sets`. The processed folder there currently holds the price, weather, macro, seasonal-extent, and DOA files.

`build_analysis_ready_panel` turns the price, weather, and seasonal-extent files into one weekly panel. A row is one crop, one market, one HARTI Monday. Short price holes of one or two weeks may be filled into a feature state. The published price used as a label stays missing when HARTI did not publish it.

`build()` in `src/htfe/training/build_training_table.py` reads that panel and explodes it by horizon. A Monday forecast sees prices and weather from earlier weeks only. The script writes `outputs/panel/training_table.parquet`. The manifests for Experiments A and B both record 54,243 rows in that table. `outputs/panel/feature_coverage.csv` records the same row count. `cultivation_progress_ratio` is empty on every one of those rows, because `cultivation_monthly.csv` is absent.

`assign_split` labels each row from the forecast issue week. Issue weeks through 2022-12-31 are train. Issue weeks in 2023 are calibration. Issue weeks in 2024 and 2025 are test. Rows are not shuffled.

Experiment A fits LightGBM and XGBoost on price lags and season, and fits SARIMAX on the price series. Experiment B fits the same trees again with weather, diesel, USD/LKR, inflation, and lagged finished-season extent added. Experiment B reuses the SARIMAX forecasts. It does not fit SARIMAX again. Experiment C keeps only the rows that have a real national cultivation report, fits B* on the Experiment B columns, and fits C with the national as-of ratio added on those same rows.

Scores are written to CSV and JSON. `export_evaluation` copies the full-sample A and B scores into `app/data`. The studio reads `app/data` for those A and B cards. Cultivation proof and the horizon decision read `outputs/experiment_c` and `outputs/evaluation`. The dashboard price is loaded from `app/data/live_forecast.json`, which the runtime model wrote. It is a different fit.

Open next: section 3, then `src/htfe/training/build_training_table.py`.

## 3. Inputs and outputs

Follow the chain. Each step says what it reads and what it writes.

### Contract files

`src/htfe/config.py` names the four contract files:

| Key | File |
| --- | --- |
| prices | `harti_prices_weekly_long.csv` |
| weather | `weather_weekly_open_meteo.csv` |
| macros | `macros_published.csv` |
| extent | `dcs_highland_seasonal.csv` |

`cultivation_monthly.csv` is a fifth, optional district-month file. It is absent from `data/member1` and from the processed folder. The district-month loader therefore attaches no hectares. The training-table column `cultivation_progress_ratio` stays null. Nothing is invented to fill it.

The Department of Agriculture extract keeps the name `doa_cultivation_progress.csv`. The code does not rename it to `cultivation_monthly.csv`. Experiment C reads it through `doa_progress_path()`, which checks `data/member1` first and then the processed folder. The Experiment C manifest records the processed path: `D:\SLIIT\4th Year\Reaserch\data sets\processed\doa_cultivation_progress.csv`.

### Weekly panel

`src/htfe/data/build/build_analysis_ready_panel.py` function `build()` reads:

- `harti_prices_weekly_long.csv`
- `dcs_highland_seasonal.csv` (it can call the HTML parser if that CSV is missing)
- `weather_weekly_open_meteo.csv`
- `src/htfe/data/maps/origin_map.yaml`

It keeps wholesale weekly rows for carrot, leeks, and tomato, and for the four score markets plus the three train-pool markets. It builds a Monday grid from 2016 through 2025, attaches lagged finished-season supply, joins origin weather, adds holiday flags, and applies the missingness rules. `promote()` writes `outputs/panel/analysis_ready_panel.parquet` and a dated copy under `outputs/panel/v<stamp>/`.

On this machine that parquet is not in `outputs/panel`. A copy is in the fallback folder `processed/latest/analysis_ready_panel.parquet`. `built_file()` uses the `outputs/panel` copy when it exists, and the fallback copy otherwise. The training table that the experiments actually read is the one in `outputs/panel`.

### Training table

`build()` in `src/htfe/training/build_training_table.py` reads the analysis-ready panel, `origin_map.yaml`, `dcs_highland_seasonal.csv`, and the macro file. It looks for `cultivation_monthly.csv` and, on this machine, finds none.

It writes:

- `outputs/panel/training_table.parquet`
- `outputs/panel/supply_origin_mapping.csv`
- `outputs/panel/cultivation_observed.parquet` (empty of observed monthly rows when the monthly file is absent)
- `outputs/panel/cultivation_synthetic.parquet` (the labelled calendar curve, kept off the training table)
- `outputs/notes/Training-Table-Columns-IT23415836.md`
- `outputs/notes/Cultivation-Modes-IT23415836.md`

`cultivation_is_synthetic` is false on the training table. The dictionary written beside it says the synthetic calendar is not a column of that table.

A forecast row uses `forecast_issue_week` as the Monday. `target_week` is that Monday plus `horizon_weeks` times 7 days. `target_price_lkr_kg` is blank when that future week was not published. Score rows need both a last observed price and an observed target price, both above zero.

### The split

`assign_split` in `src/htfe/training/config.py` reads the issue Monday and writes two columns: `split` and `is_shock`.

- train: issue week on or before 2022-12-31
- calibrate: issue week from 2023-01-01 through 2023-12-31
- test: issue week from 2024-01-01 through 2025-12-31
- shock flag: issue week from 2024-01-01 through 2024-03-31

The tree protocol inside the train block is named `notebook_2022_early_stop`. `selection_masks` in `src/htfe/models/trees.py` splits training targets again:

- selector fit: eligible training targets through 2021-12-31
- early stop: eligible training targets whose target week falls in 2022, patience 50, cap 500 trees
- saved trees: refit on all eligible training targets through 2022-12-31, using the tree count chosen above
- 2023 is not used to choose trees. 2023 is used only to compute `qhat`, the conformal adjustment
- seed 42
- quantiles 0.05, 0.50, and 0.95

LightGBM and XGBoost hyperparameters stay at the notebook defaults. `n_estimators` is the only size that early stopping changes. Optuna is not called by the experiment scripts. An older helper, `methodology_fit.py`, still contains tuning functions. The manifests say `"optuna": {"used": false}`.

### What each experiment may see

Feature lists are in `src/htfe/features/sets.py`.

Experiment A may see price history and season:

- price lags at 1, 2, 4, 8, and 52 weeks, plus the 4-week mean, the 8-week mean, and the 4-week percent change
- season, a season-transition flag, forecast week of year, and week within season
- crop and market enter as category codes in the design matrix

Experiment B may see everything in A, plus:

- origin weather: 1-week rain, 4-week rain sum, 8-week rain sum, 4-week mean temperature, 4-week rain anomaly
- diesel price, USD/LKR, and inflation
- lagged finished-season extent: previous season, same season last year, historical mean of finished seasons, and the change between the last two finished seasons

B is withheld if diesel, USD/LKR, and inflation are not all present as real as-of series. The current table has those three columns filled. Inflation uses reference month-end plus 21 days, so a forecast in the first three weeks of a month does not see that month's CCPI. B does not see `cultivation_progress_ratio`. The script stops if that column is not all null.

Experiment C may see the B columns only on matched rows, plus two columns:

- `cultivation_progress_ratio_real`, achieved hectares divided by target hectares
- `cultivation_target_exceeded_real`, 1 when that ratio is above 1

A ratio above 1 is kept. A non-positive target stays in the audit and is not turned into 1. No hectares are invented.

The match rule is in `attach_national_asof` in `src/htfe/data/build/doa_national_progress.py`. A report is attached when all of these hold:

- the crop matches
- the official season on the report matches the operational season of that Monday (October–March is Maha, April–September is Yala; `Maha 2024/25` is read as Maha 2024)
- `available_date` is on or before that Monday

The join is a backward as-of merge, so the latest eligible report is the one that sticks. Issue years 2016, 2017, 2020, and 2021 are excluded from the scored signal. Allowed issue years in the code are 2018, 2019, 2022, 2023, 2024, and 2025. There is no report-age cutoff in days.

District rows stay in the audit. The audit summary on disk records 172 source rows after the crop filter, 72 national rows, 100 district rows, and 72 valid national rows. Of 10,941 crop-market weeks, 2,688 have a national signal. The eligible C table has 12,531 rows across horizons, from `outputs/experiment_c/manifest.json`. Those rows are the matched set. They are a different sample from the 54,243-row full table.

B* is the Experiment B feature list trained on that matched set. C is the same rows with the two cultivation columns added. Compare C with B*, not with the full-sample Experiment B MAE.

The synthetic calendar is not a feature set in `feature_sets()`. It is written to `cultivation_synthetic.parquet` and used only when `run_runtime_model.py` trains the dashboard model.

### Protocol details that are easy to mix up

- Direct multi-horizon. Horizons 1, 2, 4, 8, and 12 each have their own quantile models.
- Trees predict `log(target_price / last_observed_price)` and convert back with last price times exp(prediction). The point forecast is the 0.50 quantile.
- SARIMAX order is (1, 1, 1) x (1, 0, 0, 52). It is not AutoARIMA. `statsforecast` is not in `requirements.txt`. The SARIMAX note in code says the order is fixed because that package is not installed. Leave it uninstalled.
- LSTM was not run. `app/data/comparison.json` has a row named "LSTM-style MLP (not run)" with zero scored rows.
- SARIMAX has no joblib. Its forecasts live in `outputs/experiment_a/predictions.csv`. The interval columns are empty. There is no PICP for SARIMAX.
- The 12-week LightGBM gap between A and B is the full B feature set against A. The saved full-sample numbers are LightGBM A MAE 159.26 and LightGBM B MAE 110.09, both with n=1071. That comparison adds weather, macros, and lagged extent together. There is no separate weather-only, macro-only, or extent-only fit in these experiments. Leave out any sentence that ranks "weather, then macro, then extent."

### What each experiment writes

Experiment A (`run_experiment_a.py`) reads `outputs/panel/training_table.parquet`. It writes under `outputs/experiment_a/`:

- `models/` — LightGBM and XGBoost joblib bundles for feature set A, one file per horizon and quantile, plus `training_protocol.json`
- `predictions.csv` — tree forecasts and SARIMAX forecasts together
- `sarimax_partial.csv` — SARIMAX rows saved while the walk is still running
- `sarimax_order.json` — the fixed order and the note about statsforecast
- `metrics_by_model_horizon.csv`, `metrics_by_series.csv`
- `cqr.json` — `qhat` by model and horizon
- `manifest.json`, `experiment_status.json`

`sarimax_gap_predictions.csv` and `sarimax_comparability.json` are also in that folder. The current `run_experiment_a.py` does not write those two names. I did not find a writer for them in the training modules I opened. Treat them as leftover files from an earlier run.

Experiment B (`run_experiment_b.py`) reads the same training table and the Experiment A prediction file, and only the SARIMAX rows from it. It writes under `outputs/experiment_b/`:

- `models/` — joblib bundles for feature set B
- `predictions.csv` — B tree forecasts plus the reused SARIMAX rows
- `metrics_by_model_horizon.csv`, `metrics_by_series.csv`, `cqr.json`
- `ab_comparison.csv`, `ab_comparison_by_crop_market.csv`
- `manifest.json`, `experiment_status.json`, `live_vectors.json`
- `outputs/notes/Experiment-AB-Comparison-IT23415836.md`

Experiment C (`run_experiment_c.py`) reads the training table, the Experiment B manifest (for the feature list and seed 42), and `doa_cultivation_progress.csv`. It writes only under `outputs/experiment_c/`. It checks that Experiment A, Experiment B, and `app/data` score files keep the same hashes.

- `models/` — C joblibs
- `models_b_star/` — B* joblibs
- `predictions_c.csv`, `predictions_b_star.csv`
- `metrics_c.csv`, `metrics_b_star.csv`, `b_star_vs_c.csv`
- `c_eligible_table.parquet`, `readiness_counts.csv`
- `lightgbm_h4_by_crop.csv`, `lightgbm_h4_by_market.csv`
- `cqr.json`, `cqr_b_star.json`, `manifest.json`, `status.json`, `audit_summary.json`
- `audit/` — the audited DOA extract, season mismatches, and coverage tables
- `dynamic_reforecast.csv`, `dynamic_reforecast_note.json`
- `live_vectors.json`, `member3_packet.json`
- `inference/` — the Member 3 file and the reproduction checks
- `evaluation/` — common-key, bootstrap, SHAP, shock, and interval checks

### Scores the studio reads

`python -u -m htfe.training.export_evaluation` reads the A and B metric files and Experiment B predictions. It writes:

- `app/data/evaluation.json` — full-sample A and B test rows
- `app/data/predictions.json` — Experiment B LightGBM horizon-4 chart points
- updates to `app/data/models.json` and `app/data/comparison.json` for the horizon-4 cards and the naive rows
- `outputs/notes/Model-Training-and-Evaluation-IT23415836.md`

The same module also has `export_this_run_studio()`, which writes `outputs/evaluation/studio_json/` and does not rewrite `app/data`. `run_saved_checks` calls that function. The README command `export_evaluation` is the one that refreshes `app/data`.

The studio then reads:

- `app/data` for the full-sample A and B cards, the dashboard series, and the horizon-4 interval chart
- `outputs/experiment_c` for the matched-row test, the Member 3 packet, and the C diagnostics
- `outputs/evaluation` for the common-key benchmark, the seasonal-naive table, and `best_model_by_horizon.csv`

### The packet for Member 3

Member 3 receives one file: `outputs/experiment_c/inference/member3_forecast_c.json`.

`twelve_forecasts` in `saved_inference.py` keeps the latest horizon-1 test row for each of 3 crops and 4 score markets, so 12 forecasts. The packet copies the price, the interval, the national progress, and the report date. It also stores measured calibrated PICP by horizon, taken from `metrics_c.csv` for LightGBM:

| Horizon | Measured PICP % |
| ---: | ---: |
| 1 | 85.71 |
| 2 | 91.29 |
| 4 | 91.17 |
| 8 | 93.74 |
| 12 | 93.56 |

0.90 is the intended level. Horizon 1 is below it. The packet's own note says that.

Member 3 does not need `predictions_c.csv` and does not need the joblibs. The twelve-forecast CSV in `inference/` still has a historical actual, because it is a reproduction check. The packet given across the team leaves that actual out.

There is also `outputs/experiment_c/member3_packet.json`. The studio page "Given to Member 3" reads `inference/member3_forecast_c.json`. Point at that inference file.

### The dashboard runtime model

`src/htfe/training/run_runtime_model.py` reads the training table, `cultivation_synthetic.parquet` (scenario `typical`), and the Experiment B feature list. It trains LightGBM with a column named `cultivation_progress`. The fit uses the older early-stopping helper on the 2023 calibration rows. It is not the 2022 selector protocol used by Experiments A, B, and C.

It writes `outputs/runtime_model/`:

- `models/` — runtime joblibs
- `predictions.csv`, `cqr.json`, `live_vectors.json`
- `service_manifest.json` — `cultivation_source` is `synthetic_calendar`, and `not_experiment_c` is true
- `commitment_records.csv`, `mechanism_check.json`

It also writes `app/data/live_forecast.json`. The carrot Colombo card on disk is issue week 2025-12-22, forecast week 2025-12-29, expected price 175.83. `progress_in_model` is true, so the dashboard slider can rescore. The stored `cultivation_progress` on that card is null. `app/data/meta.json` still says `progress_in_model` false. The dashboard follows `live_forecast.json`, not that meta flag.

`src/htfe/api/runtime.py` loads those runtime joblibs and can append a commitment record. `src/htfe/api/app.py` is a small FastAPI service around the same idea. The studio does not need you to start that API.

### Numbers worth memorising, and the file they come from

Full-sample 2024–2025 test, from `app/data/evaluation.json` (the same figures are in `outputs/evaluation/best_model_by_horizon.csv` for the winner at each horizon):

- 1 week, SARIMAX: MAE 60.96, MAPE 22.94%, n=1200. No interval.
- 2 weeks, SARIMAX: MAE 83.45, n=1188. No interval.
- 4 weeks, XGBoost B: MAE 107.56, MAPE 42.38%, n=1164.
- 4 weeks, LightGBM B: MAE 110.14, MAPE 43.96%, n=1164. Calibrated PICP 83.42%, which is below 90%. Raw PICP 75.34%.
- 4 weeks, LightGBM A: MAE 126.60, n=1164.
- 4 weeks, last-value naive, from `app/data/comparison.json`: MAE 142.91, MAPE 64.65%, n=1164. No interval.
- 8 weeks, LightGBM B: MAE 110.05, n=1116.
- 12 weeks, LightGBM B: MAE 110.09, n=1071.

Matched rows, LightGBM, from `outputs/experiment_c/b_star_vs_c.csv`:

- Horizon 1, n=462: C MAE 59.75, B* MAE 57.88. C is worse. C calibrated PICP 85.71%, below 90%.
- Horizon 4, n=453: C MAE 80.04, B* MAE 83.58.

Leave out the sentence "C helps at horizon 1." Horizon 1 is the horizon where LightGBM C has the higher MAE.

`outputs/evaluation/common_key_benchmark.csv` is a smaller intersection: a test row must be present for LightGBM, XGBoost, and SARIMAX. Horizon 1 there has n=1185, and the SARIMAX MAE is 60.69, not 60.96. The note beside it says these rows do not replace the full-sample cards. If you re-run `run_saved_checks`, the current `write_common_key_benchmark` also rebuilds `best_model_by_horizon.csv` from that intersection. The file on disk today still has the full-sample counts (n=1200 at 1 week). Quote the file you are pointing at, and say whether it is the full sample or the common-key subset.

Open next: the folder guide in section 4. If you only have time for one code file, open `src/htfe/training/build_training_table.py` through the `build()` function, then `src/htfe/features/sets.py`.

## 4. Folder structure

```
README.md
requirements.txt
data/member1/                  contract drop folder (.gitkeep only, in this checkout)
src/htfe/                      forecast engine
app/                           Streamlit studio, JSON, notebooks
tests/                         small locks on the as-of rule, the split, and saved helpers
outputs/                       gitignored training output
  panel/
  experiment_a/
  experiment_b/
  experiment_c/
  evaluation/
  figures/
  runtime_model/
  fixture/
  notes/
```

`outputs/_staging/` appears only while a run is writing. `publish.py` deletes it after a success or a failure. It is not a result folder.

### `src/htfe`

`config.py` reads nothing from disk until a caller asks. It defines the repo root, `data/member1`, `outputs`, `app/data`, and the fallback data root. `contract_path` returns one of the four CSVs. `cultivation_monthly_path` returns the monthly file or `None`. `built_file` prefers `outputs/panel/<name>` and otherwise `processed/latest/<name>`.

`__init__.py` files under `src/htfe` are package markers.

### `src/htfe/data/ingest`

These scripts fetch or tidy sources. The experiment commands do not re-download them. They write into the processed data folder, outside `outputs`, unless a docstring says otherwise.

- `harti_excel_prices.py` reads the HARTI workbook `Vegetable Prices.xlsx` and writes `harti_prices_weekly_long.csv` and `harti_prices_monthly_long.csv`. Weekly sheets use W1–W52 or W53. The cleaned price column is `price_lkr_per_kg`.
- `weather_open_meteo.py` reads the Open-Meteo archive API, with help from `weather_common.py` (locations, HARTI week assignment, retries). It writes daily, weekly, monthly, and location CSVs. The contract file is `weather_weekly_open_meteo.csv`, aligned to HARTI Mondays.
- `weather_nasa_power.py` is a second weather fetch, NASA POWER. It writes `weather_daily_nasa_power.csv`. The training weather join uses the Open-Meteo weekly file.
- `weather_common.py` is shared: location list, HTTP retries, HARTI week starts, and aggregation. It does not write the contract file by itself.
- `dcs_agriculture.py` downloads Department of Census and Statistics highland-crop HTML and related files into the raw folder.
- `cbsl_price_report.py` crawls Central Bank price-report PDFs. I did not find it on the path that builds the training table. The macro file used in training is `macros_published.csv`.

### `src/htfe/data/build`

- `scope.py` holds constants only: crops, markets, horizons, split dates, shock window, weather columns to keep. It reads and writes nothing.
- `build_analysis_ready_panel.py` reads the contract price, extent, and weather files plus `origin_map.yaml`. It writes the scoped interim CSV and, through `promote()`, `analysis_ready_panel.parquet`.
- `missingness.py` reads a panel frame and returns it with `is_observed`, `is_imputed`, gap length, and disruption flags. A hole of one or two weeks can be filled into the feature price. The observed flag stays false on that week, so it is not a scored target.
- `supply_lags.py` reads the panel, the seasonal extent file, and the origin map. It attaches extent only from seasons that have already finished. Yala of year S is finished after HARTI week 39 of S. Maha of year S is finished after week 13 of S+1. `assert_lagged_extent_examples` is a small lock the table builder runs first.
- `join_weather.py` reads weekly weather and the origin map. Weather is joined from supply districts, not from the market town. Weights use lagged finished-season extent when it exists, and equal weights when that extent is missing. The method name is `extent_weighted_weather`. `supply_weight` stays null. These are not measured trade shares.
- `holiday_flags.py` reads week-start dates and returns Sri Lanka holiday flags. The panel builder merges them. The training-table column list in `build()` does not keep those flags, and feature sets A, B, and C do not include them.
- `parse_dcs_highland.py` reads highland-crop HTML for carrot, leeks, and tomato and writes `dcs_highland_seasonal.csv`.
- `fetch_published_macros.py` reads CPC diesel pages, a CBSL USD/LKR file, and CCPI PDFs. It writes `macros_published.csv` only after published anchor values match. Inflation's available date is reference month-end plus 21 days.
- `macros.py` reads that published file and as-of joins it onto forecast Mondays. It writes a macro note under the notes folder. It does not invent a missing series.
- `cultivation_progress.py` reads `cultivation_monthly.csv` from the one known path. `load_monthly` returns `None` when the file is absent or the columns do not match year, month, crop, district, target hectares, achieved hectares. `doa_progress_path` finds `doa_cultivation_progress.csv` and does not pass it to this loader. `inspect_cultivation_sources` is what the studio status card calls. `assert_formula` checks achieved/target and that a ratio above 1 is kept.
- `doa_national_progress.py` reads the DOA CSV. It writes nothing by itself. Experiment C calls `audit_progress` and `attach_national_asof`, then writes the audit CSVs. District rows are dropped before the as-of join used for scoring.
- `cultivation_features.py` builds two side frames. `ObservedMonthlyProvider` is empty when the monthly file is absent. `SyntheticCalendarProvider` reads the seasonal extent file and `crop_calendar.yaml` and builds the logistic curve. `validate_cultivation_outputs` checks that the training table did not absorb that curve. `render_modes_audit` is the text written to the cultivation-modes note.
- `__init__.py` is the package marker.

### `src/htfe/data/maps`

- `origin_map.yaml` is the supply map. Each crop and market lists weather points and DCS supply districts. `assumption: true` means the origin list is a research judgement, not a published flow table. Carrot into Dambulla, Colombo, and Meegoda is mapped to Nuwara Eliya and Badulla weather, not to weather at the market town.
- `crop_calendar.yaml` is the synthetic planting window only. The file says these weeks are not observed planting dates. Yala planting in the file is weeks 14–22. Maha planting is weeks 40–48. Profiles are early, typical, and late. The runtime model uses `typical`.
- `name_maps.yaml` maps spellings (LEEK to leeks, Meegoda(DEC) to Meegoda, and similar). The panel builder copies it next to a promoted panel.

### `src/htfe/data/schemas` and `src/htfe/data/audit`

- `schemas/panel_schema.py` checks the weekly panel with pandera: crop, market, week, price, observed flag, and a unique crop-market-week key. `build_analysis_ready_panel.main` calls `validate` before writing.
- `audit/panel_audit.py` reads a panel and writes a coverage summary. It is a check script, not a step in `run_experiment_a`.
- `audit/profile_harti_excel.py` reads the tidy HARTI CSVs and writes coverage tables and a markdown profile under the processed audit folder.

### `src/htfe/features`

- `sets.py` reads a training-table frame and returns the column lists for A, B, and C. B is `None` when macros, weather, or lagged extent are incomplete. C is `None` when `cultivation_progress_ratio` has no observed values. That is why the full-sample path never trains C inside Experiment A. National progress is attached later, under a different column name, by `run_experiment_c.py`.
- `frame.py` is an older feature builder. It reads `analysis_ready_panel.parquet` and can write `feature_frame.parquet`. Experiments A, B, and C do not call it. The fallback `processed/latest` folder still has a `feature_frame.parquet`. The training table is the file the experiments read.
- `__init__.py` is the package marker.

### `src/htfe/models`

- `trees.py` holds the quantile settings and `notebook_2022_early_stop`. `fit_selected_lightgbm` and `fit_selected_xgboost` read a design matrix and the masks, and return a fitted model plus a record of row counts and dates. They write nothing. The experiment script writes the joblib. Quantiles are 0.05, 0.50, 0.95. Seed is 42. Early stopping patience is 50. The cap is 500. The older `_fit_lgbm` early-stops on whatever calibration frame the caller passes. The runtime script uses that older helper. Experiments A, B, and C use `fit_selected_*`.
- `sarimax.py` reads a weekly price frame. `fit_notebook_from_table` reads `harti_prices_weekly_long.csv`, rebuilds the Monday grid, fills 1–2 week holes the same way as the panel, and walks forward with `append(..., refit=False)`. It returns a prediction frame. The caller writes `predictions.csv`. There is no joblib. Order is (1, 1, 1) x (1, 0, 0, 52). `predict()` is an older entry point on a feature frame. Experiment A calls `fit_notebook_from_table`.
- `intervals.py` re-exports the conformal helpers. The experiment path uses `src/htfe/training/conformal.py`.

### `src/htfe/training`

Group the runners first, then the helpers.

Runners:

- `build_training_table.py` — reads the panel and contract files, writes the training table and the two cultivation side files. Described in section 3.
- `run_experiment_a.py` — reads the training table, writes Experiment A. `fit_quantile_trees` is the shared tree fit that B and C call too.
- `run_experiment_b.py` — reads the training table and Experiment A SARIMAX rows, writes Experiment B. It refuses to change the Experiment A folder.
- `run_experiment_c.py` — reads the training table, the B manifest, and the DOA file, writes Experiment C.
- `run_runtime_model.py` — reads the training table and the synthetic curve, writes `outputs/runtime_model` and `app/data/live_forecast.json`. It checks that A/B accuracy files did not change.
- `export_evaluation.py` — reads A/B metrics and B predictions, writes `app/data` JSON and the training note. `export_this_run_studio` reads C and evaluation files and writes `outputs/evaluation/studio_json` without touching `app/data`.
- `run_saved_checks.py` — reads saved predictions and joblibs, writes the common-key benchmark, the C evaluation folder, the inference folder, and the studio JSON copy. It does not train.
- `coverage_audit.py` — reads the training table and origin map, writes `outputs/panel/feature_coverage.csv`, `outputs/panel/weather_origin_audit.csv`, and `outputs/notes/Feature-Coverage-Audit-IT23415836.md`.
- `refresh_macros.py` — rebuilds macro availability and can refresh macro columns on the training table. It writes `outputs/notes/Macro-Availability-Audit-IT23415836.md`. I did not find that note in `outputs/notes` on this machine.
- `figures.py` — reads experiment CSVs, the training table, and the price file, and writes the PNGs in `outputs/figures`.
- `run_cultivation_fixture.py` — builds a tiny typed monthly table and writes `outputs/fixture`. The note in the file says those hectares are not Census or Department of Agriculture records.

Helpers:

- `config.py` — split dates and `assign_split`. Reads nothing.
- `publish.py` — `begin_staging` creates `outputs/_staging/<experiment>`. The run writes there. `publish_staging` moves the finished folder into place and only then deletes the previous official folder. `discard_staging` deletes the temporary folder if the run fails, and the last good official folder stays. This is why a failed train does not have to destroy the last good models.
- `conformal.py` — reads prediction rows, computes `qhat` on 2023 only, and returns calibrated PICP. The reported interval is ordered around the unchanged 0.50 forecast before `qhat` is added to both tails. Intended coverage is 90%. The code says time-ordered prices are not exchangeable, so this is an empirical adjustment.
- `evaluate.py` — `metrics_tables` scores test rows (MAE, RMSE, MAPE, pinball, raw PICP, width). `write_common_key_benchmark` reads Experiment B predictions and the training table and writes the files in `outputs/evaluation`. It does not rewrite Experiment A, Experiment B, or `app/data`.
- `evaluate_saved_c.py` — reads saved C predictions and eligible rows. It writes the CSVs in `outputs/experiment_c/evaluation`, including SHAP. It does not refit.
- `saved_inference.py` — loads C joblibs and writes `outputs/experiment_c/inference`, including the 12 forecasts and `member3_forecast_c.json`. It can also rebuild Experiment B live vectors.
- `reforecast_c.py` — reads the eligible C rows and the saved C models, and writes `dynamic_reforecast.csv` for one target week scored from earlier issue dates. It does not invent prices or hectares.
- `baselines.py` — last-value and 52-week seasonal naive. `score_same_rows` is what fills the naive lines in `app/data/comparison.json`. Last-value uses the last published price before the issue Monday, on the same rows as the reference model. Seasonal naive uses the published price 52 weeks before the target Monday, and drops a row when that price is missing.
- `align_experiment_a.py` — conformal helper used by the runtime model. Running it does not fit a second SARIMAX cache.
- `methodology_fit.py` — older fit and Optuna helpers, plus SHAP and ablation payload builders. The locked experiment entry points do not call the tuners.
- `__init__.py` — package marker.

### `src/htfe/api`

- `runtime.py` reads `outputs/runtime_model` joblibs, live vectors, and `commitment_records.csv`. `score_progress` can write a new commitment row and return a new interval. A future-dated report is ignored. A ratio above 1 is kept. `mechanism_check` writes its result when the runtime script calls it.
- `app.py` reads `app/data/live_forecast.json` and the runtime service manifest, and exposes a forecast request. It does not recommend a crop.

### `app`

- `app.py` is the Streamlit entry. It reads the studio JSON through `load_studio`, and it reads Experiment C status for the sidebar caption. It writes nothing except when the dashboard calls `write_forecast_packet`, which writes `app/outputs/forecast_packet.json` (a local packet, not the Member 3 inference file). Pages do not retrain.
- `.streamlit/config.toml` is theme colours only.
- `requirements.txt` lists Streamlit, Plotly, and pandas for the studio.
- `Notebooks/` holds the written method. See below.
- `views/` and `lib/` are the pages. Section 5 walks the pages in sidebar order.
- `data/` is the JSON the studio can open without `outputs`.

`app/lib/embed.py` knows paths to Member 3 and Member 4 apps in a parent prototype folder. `app.py` does not call it. Member 3 and Member 4 on this branch are the note in `nav.py`.

### `app/data`

The studio loader requires these JSON files: `meta`, `coverage`, `split`, `data_dictionary`, `origin_map`, `series`, `cobweb`, `comparison`, `ablation`, `commitment_weeks`, `importance`, `live_forecast`, `models`, `evaluation`. `predictions.json` is loaded by the Uncertainty page.

What they hold, from the files on disk:

- `evaluation.json` — full-sample A and B test scores by model and horizon. This is the Results page table.
- `comparison.json` — horizon-4 comparison, including last-value naive (n=1164, MAE 142.91, MAPE 64.65) and seasonal naive (n=1137, MAE 161.70, MAPE 73.05). LSTM is the unscored row.
- `models.json` — horizon-4 metric blocks and LightGBM `qhat` by horizon. The Uncertainty page reads `qhat`.
- `predictions.json` — Experiment B LightGBM horizon-4 points for the interval chart.
- `ablation.json` — full-sample LightGBM A and B by horizon. `C` is null. The note says C is not in this table.
- `live_forecast.json` — runtime dashboard cards.
- `meta.json` — labels, seed 42, `n_panel` 5761, `n_scored_test` 1200. `progress_in_model` here is false. The live cards disagree, and the dashboard uses the live cards.
- `split.json` — train 4380 rows (2016-01-04 to 2022-12-26), calibration 624 (2023), test 1260 (2024-01-01 to 2025-12-29). I did not find code on this branch that rebuilds these three counts from the 54,243-row training table. They are the saved panel summary on the Data page. A scored forecast count such as n=1200 is crop × market × issue week with an observed target, on the score markets. It is a different count.
- `coverage.json` — per-series price-week share, plus exclusion notes. The "price weeks %" column is that share. It is not PICP.
- `series.json` — price points for the Data page chart.
- `cobweb.json` — finished-season extent ratio against the next season's average price. The page says this is not current achieved hectares over target hectares.
- `origin_map.json` — the origin map the Features page draws.
- `data_dictionary.json` — the column dictionary table.
- `importance.json` — a stored bar chart. The first row is `price_lag_1w` with importance 1.0. This is not the Experiment C SHAP table.
- `commitment_weeks.json` — the list on Cultivation proof. The page treats an empty list as "the dashboard slider is not Experiment C."

### `app/Notebooks`

Two notebooks are in the folder.

- `01_prepare_train_evaluate_RESEARCH_LOCKED_FINAL_FIX1 (1).ipynb` — 66 cells. The opening headings walk the wholesale question, the upload files, the Excel audit, crops and markets, and the week definition. This is the written method the Python protocol is named after.
- `02_use_saved_models_RESEARCH_LOCKED_FINAL (1).ipynb` — 26 cells. It reads saved artifacts and reproduces a forecast. It is the "use the saved model" notebook.

I read the opening headings, not every cell. The commands in section 7 run the Python modules. Those modules are what wrote the scores in `outputs` and `app/data`.

### `data/member1`

Only `.gitkeep` is in the repo. Drop the four contract CSVs here when you want the loader to prefer the repo over `COBWEB_DATA_ROOT`. You can also drop `doa_cultivation_progress.csv` here, still under that name. `cultivation_monthly.csv` would be read only if you add a real district-month file. Do not rename the DOA file to create it.

### `tests`

- `test_doa_national_asof.py` locks the as-of rule on a tiny fixture. A future report is ignored. A ratio of 1.4 stays 1.4. A district row is not the scored signal. A wrong-season report is not attached. The full DOA file is not read.
- `test_cultivation_asof.py` calls `run_cultivation_fixture.main`. That proves the district-month loader on typed hectares and refuses a future report. Those hectares are not written into the training table.
- `test_notebook_protocol.py` locks `selection_masks`: 2022 validation does not overlap 2023 calibration, and the final fit stops at 2022-12-31. It also locks that MAE and MAPE stay separate numbers, and that the interval is ordered around the median before a width adjustment.
- `test_saved_checks.py` locks three helpers without loading the C joblibs: a ratio above 1 sets the exceeded flag, tails are ordered before `qhat` is added, and an empty 2024 Q1 window is reported as not evaluable.
- `test_fallback_prices.py` locks that, while `data/member1` is empty of the four files, prices load from the fallback path, and a cultivation file is accepted only when its name is `cultivation_monthly.csv`.

### `outputs/panel`

- `training_table.parquet` — the forecast table. Experiments read this.
- `feature_coverage.csv` — non-null share by column. 54,243 rows. `cultivation_progress_ratio` non-null share is 0.
- `supply_origin_mapping.csv` — crop, market, origin district. `supply_weight` is null.
- `weather_origin_audit.csv` — written by the coverage audit.
- `cultivation_observed.parquet` — observed monthly provider output.
- `cultivation_synthetic.parquet` — synthetic calendar. Runtime training reads it. The accuracy experiments do not.
- `analysis_ready_panel.parquet` is not in this folder. The fallback `processed/latest` copy is what `built_file` returns today.

### `outputs/experiment_a`

See section 3. Models are joblib files in `models/`. SARIMAX is inside `predictions.csv`, not in `models/`.

### `outputs/experiment_b`

See section 3. `live_vectors.json` is the latest horizon-1 test vector from this fit, used when the runtime model checks that it did not change the B feature values except by adding progress.

### `outputs/experiment_c`

See section 3. For a presentation, the files to open are `b_star_vs_c.csv`, `status.json`, and `inference/member3_forecast_c.json`.

`evaluation/shap_lightgbm_c_h4.csv` includes `cultivation_progress_ratio_real` with mean absolute SHAP 0, rank 18. The Features page hides the empty cultivation SHAP chart because of that zero.

### `outputs/evaluation`

Written by `write_common_key_benchmark` from Experiment B predictions and the training table.

- `common_key_benchmark.csv` — intersection of LightGBM, XGBoost, and SARIMAX. Smaller n than the full sample.
- `seasonal_naive_benchmark.csv` — the same common keys, further limited to rows with a 52-week price lag. Horizon 4 seasonal naive: n=1149, MAE 159.20, MAPE 73.52%.
- `best_model_by_horizon.csv` — on disk today, the full-sample winners (SARIMAX at 1 and 2 weeks, XGBoost at 4, LightGBM at 8 and 12). See the warning in section 3 about what a fresh `run_saved_checks` would rewrite.
- `common_key_counts.csv`, `common_test_keys.csv`
- `mase_scale_52w_by_series.csv` — seasonal scale from horizon-1 observed target prices through 2022, because the code comment says `analysis_ready_panel.parquet` is not in `outputs/panel`.
- `model_vs_naive_block_bootstrap.csv` — model absolute error minus last-value absolute error.
- `benchmark_note.txt` — says the common-key rows do not replace the full-sample cards.
- `inference/saved_b_example.json` — one Experiment B forecast reproduced from the saved joblib.
- `studio_json/` — a copy package for this run. It does not replace `app/data`.

### `outputs/figures`

PNGs drawn by `src/htfe/training/figures.py`. The studio shows them by page name from `app/lib/figures.py`. `shock_note.json` is the 2024 Q1 caption on Cultivation proof. The cultivation-progress SHAP image may exist as a file and still be hidden on the Features page.

### `outputs/runtime_model`

The dashboard model. `service_manifest.json` says the cultivation source is the synthetic calendar and that this is not Experiment C. Do not read its `predictions.csv` as an accuracy result.

### `outputs/fixture`

`monthly_target_achieved_FIXTURE.csv`, `asof_join_FIXTURE.csv`, and `fixture_report.json`. Typed hectares for the loader test. Not training data.

### `outputs/notes`

Written notes from builders and from earlier write-ups. This guide is `How-the-project-works.md`. Other files in the folder include the training-table dictionary, the cultivation-modes audit, the feature-coverage audit, the A/B comparison, the model-training sheet, a cultivation-progress gap note, and the two presentation notes `PP1-speaking-guide.md` and `PP1-two-functions.md`. Those two presentation notes are separate. This guide does not replace them.

Open next: section 5, and keep `app/app.py` beside you so the sidebar order matches the code.

## 5. Studio pages

Sidebar order is fixed in `app/app.py`: Dashboard, What we did, Results, Which model, Cultivation proof, Uncertainty, Data, Features.

The chain buttons above the pages are Data fusion, Price forecast, Crop viability, Season simulation. Choosing Member 1 opens `app/views/member1.py`. Choosing Member 3 or 4 opens the note in `app/lib/nav.py` and does not start their apps. The pages below are Member 2.

The sidebar caption says the dashboard is the runtime model and is not the Experiment B or Experiment C score.

### Dashboard

View: `app/views/dashboard.py`.

Logic: pick a crop and market. Show the saved runtime interval. The cultivation slider calls `score_progress` only when the live card says the progress column is in the model.

Reads: `app/data/live_forecast.json` for the price. `app/data/evaluation.json` for the small "saved full-sample test" caption (1-week SARIMAX MAE and 4-week XGBoost MAE). `b4_row()` reads LightGBM B horizon-4 calibrated PICP from `evaluation.json`.

Main numbers: the big price is the runtime expected price for that crop and market. The caption under it is Experiment B horizon-4 calibrated coverage, 83.42%, below the 90% intended level. That caption is not the coverage of the price on the card.

Leave out: calling this card the Experiment B score or the Experiment C score. The page says it is the runtime model. The "Continue to Member 3" button writes a local packet and opens the Member 3 note. It does not send `member3_forecast_c.json`.

### What we did

View: `app/views/training.py`.

Logic: the two jobs, in the words on the page. Then the question, the crops, the split, the feature sets, and a count of saved joblib files.

Reads: `app/data/evaluation.json` for horizons, label, split text, and the tree protocol. `outputs/experiment_a/manifest.json` for the protocol note. `outputs/evaluation/inference/saved_b_example.json` for the "saved Experiment B, no refit" card. Joblib counts are a directory listing, not a score.

Main numbers: horizons 1, 2, 4, 8, 12. Seed and quantiles appear in the protocol sentence. SARIMAX is listed with zero joblib files and the path `outputs/experiment_a/predictions.csv`.

Leave out: any claim that this page refits a model. A later forecast loads the joblib files. The page also says a new training run writes a temporary folder and replaces the files only after that run succeeds. That is `publish.py`.

### Results

View: `app/views/results.py`.

Logic: full-sample 2024–2025 scores. MAE is LKR per kg. MAPE is a percent. They are separate.

Reads: `app/data/evaluation.json` only, plus the Results figures.

Main numbers, all from that file:

- LightGBM B, 4 weeks: MAE 110.14, MAPE 43.96%, n=1164, raw PICP 75.34%, calibrated PICP 83.42%.
- LightGBM A, 4 weeks: MAE 126.60, n=1164.
- SARIMAX, 1 week: MAE 60.96, MAPE 22.94%, n=1200, and the interval cells are empty.
- XGBoost B, 4 weeks: MAE 107.56, MAPE 42.38%, n=1164.

The A-against-B table is trees only. SARIMAX is left out of that difference because it does not use the extra columns, so its error is the same in A and B.

Leave out: Experiment C. The page says the matched-row test is on Cultivation proof. Also leave out reading 83.42% as "the model achieved the 90% interval." 90% is the intended level. 83.42% is the measured share.

### Which model

View: `app/views/comparison.py`.

Logic: one horizon at a time. The decision line reads the best-model file. Then a horizon-4 full-sample table. Then the common-key benchmark.

Reads:

- `outputs/evaluation/best_model_by_horizon.csv` for the decision cards
- `app/data/comparison.json` for the horizon-4 table, including the naive rows
- `app/data/evaluation.json` for the XGBoost B, LightGBM B, and LightGBM A captions
- `app/data/models.json` for the LightGBM crop and market slices
- `outputs/evaluation/common_key_benchmark.csv`
- `outputs/evaluation/seasonal_naive_benchmark.csv` inside the expander "Seasonal naive on the rows that have a 52-week lag"
- `outputs/evaluation/model_vs_naive_block_bootstrap.csv`

Main numbers on the decision cards, from the best-model file as it sits today: 1 week SARIMAX MAE 60.96, n=1200. 2 weeks SARIMAX MAE 83.45, n=1188. 4 weeks XGBoost MAE 107.56, n=1164. 8 weeks LightGBM MAE 110.05, n=1116. 12 weeks LightGBM MAE 110.09, n=1071.

Seasonal naive has two counts. The horizon-4 caption and the comparison table use `comparison.json`: n=1137, MAE 161.70, MAPE 73.05%. The expander table uses `seasonal_naive_benchmark.csv`: at horizon 4, n=1149, MAE 159.20, MAPE 73.52%. The first count is the full-sample LightGBM B horizon-4 rows that still have a price 52 weeks before the target. The second count is the common-key subset that has `price_lag_52w`. Quote the table you are pointing at. The page caption may say 1137. That sentence belongs to the comparison file, not to the 52-week lag table.

Leave out: mixing those two seasonal-naive rows into one number. Leave out LSTM. The comparison file shows it as not run.

### Cultivation proof

View: `app/views/novelty.py`, with the tables drawn by `app/lib/experiment_c.py`.

Logic: national matched rows. B* and C share eligible rows. District rows were not features. The dashboard is not this result.

Reads, in the order the page uses them:

- `outputs/experiment_c/inference/member3_forecast_c.json` for the handoff and the PICP table
- `outputs/experiment_c/b_star_vs_c.csv` for the MAE comparison
- `app/data/ablation.json` for the full-sample A/B block. C is null there
- `outputs/experiment_c/evaluation/` for common keys, quantile crossing, conditional PICP, residuals, the 2024 Q1 slice, the bootstrap, and SHAP
- `outputs/experiment_c/dynamic_reforecast.csv` and its note
- `outputs/experiment_c/inference/forecasts_12_c.csv` for the twelve forecasts that still include a historical actual
- `outputs/figures/shock_note.json`

Main numbers to say out loud:

- LightGBM horizon 1: B* MAE 57.88, C MAE 59.75, n=462. C is worse. Calibrated PICP 85.71%, below 90%.
- LightGBM horizon 4: B* MAE 83.58, C MAE 80.04, n=453.
- Member 3 PICP: 85.71, 91.29, 91.17, 93.74, 93.56 at horizons 1, 2, 4, 8, and 12.

A positive MAE change on this page means C is less accurate than B* on that horizon. MAE and MAPE are different columns.

Leave out: "C helps at horizon 1." Leave out quoting these MAEs as if they were the full-sample Experiment B MAEs. Full-sample LightGBM B at 1 week is MAE 77.40 on n=1200. The matched test is n=462.

### Uncertainty

View: `app/views/uncertainty.py`.

Logic: the chart is Experiment B LightGBM on the 4-week test, after the 2023 adjustment. `qhat` is added on 2023 only.

Reads: `app/data/comparison.json` for the LightGBM horizon-4 PICP and width, `app/data/models.json` for `qhat` at horizons 1 and 4, and `app/data/predictions.json` for the band.

Main numbers: calibrated PICP 83.42%, below 90%. The mean width on the comparison row is 439.47 LKR per kg. SARIMAX has no band on this page.

Leave out: calling the band a guaranteed 90% interval. The measured share is 83.42%.

### Data

View: `app/views/data.py`.

Logic: the scored wholesale weeks, the split cards, a price chart, a cobweb scatter of finished-season extent against the next season's price, and a coverage table.

Reads: `app/data/meta.json`, `split.json`, `series.json`, `cobweb.json`, `coverage.json`. It also calls `member1_status()`, which looks for an older extract under `data/v1`. That extract is a different layout from the four contract files. If it is absent, the page uses the frozen counts.

Main numbers: the header prints `n_panel` from meta, 5,761 wholesale weekly rows. The cards print train 4,380, calibration 624, test 1,260. The coverage column is renamed to "price weeks %". The caption says this percentage is how many weeks in the series have a published wholesale price. It is not the forecast interval, and it is not the 90% coverage target.

Leave out: reading "price weeks %" as PICP. Leave out treating 1,260 as the scored test n. The 1-week SARIMAX score is n=1200.

### Features

View: `app/views/features.py`.

Logic: the origin map, the as-of ratio in words, a bar chart from the stored importance file, the column dictionary, and the Experiment C figures.

Reads: `app/data/origin_map.json`, `data_dictionary.json`, `importance.json`, and `outputs/experiment_c/evaluation/shap_lightgbm_c_h4.csv` for the sentence under the figures. Figures come from `outputs/figures`, and the page skips `shap_cultivation_progress_h4.png`.

Main number: mean absolute SHAP for `cultivation_progress_ratio_real` on the horizon-4 model is 0, so that chart is hidden. The importance bars are the stored file, where `price_lag_1w` is 1.0. SHAP does not replace the B* versus C MAE test.

The as-of sentence on the page matches the rule in section 3: same crop, official season match, report date on or before the forecast Monday. District rows stay in the audit.

Leave out: reading a zero SHAP bar as "the column was not in the model." The column is in the SHAP table at rank 18 with mean absolute SHAP 0. The MAE test is still the matched-row table.

### Member 1, off the sidebar

View: `app/views/member1.py`. It opens from the chain button, not from the radio list.

It shows the same split and coverage JSON. If an older `data/v1` extract is missing, the metrics fall back to `n_panel` 5,761, scored test weeks 1,200, LightGBM B horizon-4 calibrated PICP, and the Experiment C trained flag. The handoff text on that page still mentions NASA POWER and a Badulla market. The training join used in this branch is Open-Meteo weather and the origin map in `origin_map.yaml`. Score markets in `scope.py` are Dambulla, Colombo, Meegoda, and Nuwara Eliya. When the two pages disagree, quote `scope.py` and the evaluation file for the forecast, and treat the Member 1 blurb as the older handoff wording.

Open next: section 6, and do the route once before you present.

## 6. A learning route

Do these in order. Each step answers one question.

1. What is this project? Read section 1 of this note. Then read the first screen of `README.md`. Question answered: weekly wholesale prices for carrot, leeks, and tomato, and Member 2 owns the forecast.
2. Where do the files come from? Open `src/htfe/config.py`, the `CONTRACT` dictionary and `member1_ready`. Question answered: four named CSVs, monthly cultivation absent, DOA file not renamed.
3. What is the as-of rule? Open `attach_national_asof` in `src/htfe/data/build/doa_national_progress.py`, and the tests in `tests/test_doa_national_asof.py`. Question answered: a report counts only when the crop matches, the official season matches, and the report date is on or before that Monday.
4. How is the forecast table built? Open `build()` in `src/htfe/training/build_training_table.py`. Question answered: one row per crop, market, Monday, and horizon, and the future price is the label.
5. How is the split assigned? Open `assign_split` in `src/htfe/training/config.py` and `selection_masks` in `src/htfe/models/trees.py`. Question answered: fit through 2021, early-stop on 2022, refit through 2022, 2023 only for `qhat`.
6. What did Experiment A see? Open `historical_columns` in `src/htfe/features/sets.py` and the top of `src/htfe/training/run_experiment_a.py`. Question answered: price and season only.
7. What did Experiment B add? Open the rest of `sets.py` and `src/htfe/training/run_experiment_b.py`. Question answered: origin weather, diesel, USD/LKR, inflation, and lagged finished-season extent, still with an empty current-progress column.
8. Where is SARIMAX? Open `fit_notebook_from_table` in `src/htfe/models/sarimax.py`. Question answered: fixed order (1, 1, 1) x (1, 0, 0, 52), forecasts in `outputs/experiment_a/predictions.csv`, no joblib, no interval.
9. What is Experiment C? Open `src/htfe/training/run_experiment_c.py` through `_eligible_national`, then `outputs/experiment_c/b_star_vs_c.csv`. Question answered: same matched rows, C adds the national ratio, and at horizon 1 LightGBM C is worse than B*.
10. What does Member 3 get? Open `member3_packet` in `src/htfe/training/saved_inference.py` and the file `outputs/experiment_c/inference/member3_forecast_c.json`. Question answered: 12 horizon-1 forecasts and the measured PICP. They do not need the prediction CSV or the joblibs.
11. What does each studio page show? Walk the sidebar from Dashboard to Features, with section 5 beside you. Question answered: the dashboard is the runtime model; Results is full-sample A/B; Which model is one horizon at a time; Cultivation proof is the matched test.

If a detail is missing from both the code and the saved JSON, say you did not find it. The two places that are easy to over-claim are the 12-week A-versus-B gap and the horizon-1 C result. The gap is the full B set against A. Horizon 1 C does not beat B*.

## 7. Commands

From the repository root, in Command Prompt. `set PYTHONPATH=src` has to be in that same window.

```
set PYTHONPATH=src
python -u -m htfe.training.build_training_table
python -u -m htfe.training.run_experiment_a
python -u -m htfe.training.run_experiment_b
python -u -m htfe.training.run_experiment_c
python -u -m htfe.training.export_evaluation
python -u -m htfe.training.run_saved_checks
python -u -m streamlit run app/app.py
```

Experiment A reads `outputs/panel/training_table.parquet`, so the table build comes first. `run_saved_checks` scores files already on disk. It does not train. The studio command only displays saved files.

Do not run the training commands during the presentation. A failed run is supposed to leave the last good models in place: `src/htfe/training/publish.py` writes the new run under `outputs/_staging`, and it replaces the official folder only after that run finishes. If you see a staging folder, a run was interrupted or is still going. Leave the official `outputs/experiment_a`, `outputs/experiment_b`, and `outputs/experiment_c` folders as they are.

`statsforecast` is not a requirement. Do not install it to "improve" SARIMAX. The order is already fixed.
