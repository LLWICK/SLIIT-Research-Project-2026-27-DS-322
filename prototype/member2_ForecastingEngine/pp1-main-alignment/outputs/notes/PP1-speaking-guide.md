# PP1 speaking guide

Garusingarachchi Y.B. IT23415836. Member 2. J26-DS-322.

This is how you talk, and where you point. Read a paragraph out loud. The slides do not name the two jobs. You name them. The page called What we did writes them under Two functions. The page called Which model says which model is closest at each horizon.

The studio is already scored. Do not start a training script while you are speaking.

## Confidence

You can present Member 2 if you stay inside what the studio shows.

Open http://localhost:8501 once before you speak. The first Dashboard load is about 13 seconds. After that, the other pages come up in under a second. If you open it cold in front of the examiner, you will be waiting on the first page.

Stay with the numbers on the screen. Coverage did not reach 90 percent. No one model is the winner at every horizon. The closest model changes when the horizon changes.

The buttons for Member 3 and Member 4 only open a note. Their apps are not on this branch.

The folder `outputs/` is not in git. A clone does not bring the fitted models with it. The file for Member 3 has to be copied by hand.

## Folder structure

```
src/htfe
  data/ingest
  data/build
  data/maps
  features
  models
  training
  api
app
  app.py
  views
  lib
  data
  Notebooks
data/member1
outputs
  experiment_a
  experiment_b
  experiment_c
  evaluation
  figures
  panel
  runtime_model
  notes
  fixture
tests
README.md
```

`src/htfe` is the forecast code. The studio does not live here.

`src/htfe/data/ingest` reads the source files: HARTI prices, weather, the highland season file, and the published macro reports.

`src/htfe/data/build` turns those sources into the weekly panel. It also applies the national cultivation rule: a report is used only when the crop matches, the official season matches, and the report date is on or before that Monday. `doa_national_progress.py` is that rule. `cultivation_monthly.csv` is not read there.

`src/htfe/data/maps` holds the origin map and the crop calendar. Weather is joined through the supply district, not through the market town.

`src/htfe/features` names the columns each experiment is allowed to see. Experiment A is price history and the season. Experiment B adds origin weather, diesel, the USD/LKR rate, inflation, and lagged finished-season extent. Experiment C adds the national as-of ratio on the matched rows.

`src/htfe/models` holds the LightGBM and XGBoost quantile trees, and the fixed SARIMAX order. LSTM is not here.

`src/htfe/training` builds the forecast table, fits Experiments A, B, and C, and writes the scores the studio shows.

`src/htfe/api` scores a model that is already saved. `runtime.py` is the dashboard price. It does not fit a new model.

`app` is the studio. `app/app.py` is the page list. `app/views` is one file per page. `app/lib` loads the saved scores and draws the shared headings. `app/data` is the JSON the pages read. That JSON is in git. The fitted files are not.

`data/member1` is the drop folder for Member 1. Right now it only has `.gitkeep`. The four contract files, when they are all present, are `harti_prices_weekly_long.csv`, `weather_weekly_open_meteo.csv`, `macros_published.csv`, and `dcs_highland_seasonal.csv`. `cultivation_monthly.csv` is absent. The DOA file keeps the name `doa_cultivation_progress.csv`. It is not renamed. The scored Experiment C run read it from the processed data folder.

`outputs/experiment_a` holds the Experiment A joblib files under `models/`, plus `predictions.csv`. SARIMAX stays in that prediction file. It has no tree file.

`outputs/experiment_b` holds the Experiment B joblib files under `models/`, plus `predictions.csv` and the A-against-B tables.

`outputs/experiment_c` holds the Experiment C joblib files under `models/`, the matched-row B* joblib files under `models_b_star/`, `predictions_c.csv`, `predictions_b_star.csv`, and `b_star_vs_c.csv`. The handoff file is `outputs/experiment_c/inference/member3_forecast_c.json`. That is the only file Member 3 needs from you. It has 12 horizon-1 forecasts.

`outputs/evaluation` holds `best_model_by_horizon.csv`. Which model reads that file.

`outputs/figures` holds the pictures the pages show.

`outputs/panel` holds `training_table.parquet`. That is the forecast table.

`outputs/runtime_model` holds the saved model behind the dashboard price.

`outputs/notes` holds written notes, including this guide. The other notes stay where they are.

`outputs/fixture` is a small as-of check. It is not the scored test.

`tests` checks the as-of cultivation rule, the national DOA join, the price fallback, the split protocol, and the saved-model checks.

`app/Notebooks` has two notebooks: `01_prepare_train_evaluate_RESEARCH_LOCKED_FINAL_FIX1 (1).ipynb` and `02_use_saved_models_RESEARCH_LOCKED_FINAL (1).ipynb`. They are the written method. The app does not import them.

`README.md` is the short description of the forecast, the split, and how to start the studio.

Joblib files and prediction CSVs live under `outputs/`. That whole folder is gitignored. They are on this machine because the run was already done. They are not in the git history.

## The five questions

Say these in plain words. Do not recite Python names unless someone asks where the code is.

### 1. Two key functions

The first job is to build the forecast table. I make one row for each crop, each market, each week I issue the forecast, and each horizon. A column is allowed into that row only when it was already public on that Monday.

The second job is to fit and score those forecasts. Experiments A and B, and SARIMAX, are scored on the full sample. Experiment C is scored only on the matched national rows. I then say which model is closest at each horizon. I do not name one model for every horizon.

### 2. What each one takes in

The table takes the wholesale price, origin weather, and the published macros. The macros are diesel, the USD/LKR rate, and inflation. It also takes lagged finished-season extent. That extent is a finished season, not this week's planting. The national cultivation ratio goes in only when the report date is on or before that Monday and the official season matches the season of that Monday. The ratio is achieved hectares divided by target hectares, taken from the report. I do not fill in hectares that were not published. District rows stay in the audit. They are not model inputs.

The score takes that same table, split by the week the forecast is issued. Training issue weeks run through 2022-12-31. The year 2023 is used only to set the interval adjustment. Test issue weeks are 2024 and 2025. Rows are not shuffled.

Experiment C was fit on those matched national rows. It was not fit on a synthetic calendar. The synthetic calendar belongs to the runtime model on the dashboard, and only there.

### 3. What each one produces

The table is written to `outputs/panel/training_table.parquet`.

The score writes the fitted trees as joblib files under `outputs/experiment_a/models`, `outputs/experiment_b/models`, and `outputs/experiment_c/models`. It also writes `outputs/evaluation/best_model_by_horizon.csv`. The studio pages read those scores.

Member 3 receives one file: `outputs/experiment_c/inference/member3_forecast_c.json`. It contains 12 horizon-1 forecasts, three crops by four markets. Member 3 does not need `predictions_c.csv`.

### 4. How each one serves the objective

The objective is a weekly wholesale price, in rupees per kilo, for carrot, leeks, and tomato, at 1, 2, 4, 8, and 12 weeks, with a range around the point.

The table keeps future weather, future macros, future prices, and later planting reports out of the past. The forecast for a Monday can use only what was already public that Monday.

The score says which added information changes the error, and which model is closest at each horizon. It does not crown one winner. It also does not treat 90 percent coverage as something we reached. Ninety percent is the level the interval is aimed at. The measured share is a different number, and on the full-sample 4-week LightGBM row it is below 90.

### 5. Where to point

Point at the page and, if they ask, at the file.

The two jobs are on What we did, in the box Two functions. The table is built in `src/htfe/training/build_training_table.py`, function `build`. The as-of cultivation rule is in `src/htfe/data/build/doa_national_progress.py`.

The fit is in `src/htfe/training/run_experiment_a.py` (`fit_quantile_trees` and `_train_experiment_a`), `src/htfe/training/run_experiment_b.py` (`_train_experiment_b`), and `src/htfe/training/run_experiment_c.py` (`_train`). SARIMAX is `src/htfe/models/sarimax.py`, function `predict`. The trees are `src/htfe/models/trees.py`. The studio JSON is refreshed by `src/htfe/training/export_evaluation.py`, function `export_this_run_studio`.

The decision sentence is on Which model. That page reads `outputs/evaluation/best_model_by_horizon.csv`. The 4-week full-sample numbers are on Results, from `app/data/evaluation.json`. The matched-row test and the Member 3 packet are on Cultivation proof.

## Numbers you can say

Say a number only when you are pointing at the place that shows it.

The protocol is one calendar. Fit through 2021. Early-stop on 2022, patience 50, cap 500. Refit through 2022. Use 2023 only for the interval adjustment, q-hat. Seed 42. Each horizon has its own model. The trees predict the log of the target price divided by the last observed price, then that is turned back into rupees per kilo. The quantiles are 0.05, 0.50, and 0.95. SARIMAX is order (1, 1, 1) times (1, 0, 0, 52). It is not AutoARIMA. LSTM was not run.

On the full 2024–2025 sample:

- 1 week: SARIMAX, MAE 60.96, MAPE 22.94 percent, n=1200. SARIMAX has no interval.
- 2 weeks: SARIMAX, MAE 83.45, n=1188.
- 4 weeks: XGBoost B, MAE 107.56, MAPE 42.38 percent, n=1164. LightGBM B, MAE 110.14, MAPE 43.96 percent, n=1164. Calibrated PICP for that LightGBM row is 83.42 percent, which is below 90. Raw PICP on the Results page is 75.34 percent. LightGBM A at 4 weeks is MAE 126.60. Moving from A to B for LightGBM at 4 weeks is about a 13 percent drop in MAE, from 126.60 to 110.14. Last-value naive is MAE 142.91, MAPE 64.65 percent, n=1164.
- 8 weeks: LightGBM, MAE 110.05, n=1116.
- 12 weeks: LightGBM, MAE 110.09, n=1071. That 12-week line is the full Experiment B set against Experiment A. It is not a weather-only test. Do not rank the columns as weather, then macro, then extent.

Seasonal naive has two counts, and they are not the same count. On Which model, the caption under 4-week full sample says seasonal naive uses 1,137 of those rows. The table in that same box shows seasonal naive MAE 161.7 and MAPE 73.05 percent. If you open the expander Seasonal naive on the rows that have a 52-week lag, horizon 4 in that file is n=1149, MAE 159.20, MAPE 73.52 percent. Point at one table. Quote that table. Do not treat 1,137 and 1,149 as one number.

Experiment C against B-star, LightGBM, matched national rows:

- 1 week, n=462: C MAE 59.75, B-star MAE 57.88. C is worse, by 3.23 percent. C calibrated PICP is 85.71 percent, below 90. Do not say C helps at horizon 1.
- 4 weeks, n=453: C MAE 80.04, B-star MAE 83.58, about 4.24 percent lower for C. C calibrated PICP is 91.17 percent. That is the matched-row interval, not the full-sample 83.42 percent.

The 2024 first-quarter shock note says the horizon-4 Experiment C window has zero rows, so that shock is not scored for C.

The Member 3 packet PICP, by horizon, is 85.71, 91.29, 91.17, 93.74, and 93.56. Horizon 1 is below 90. This is not an older set of 87.23.

The dashboard file for Carrot, Colombo stores a 1-week forecast for the week 2025-12-29. The saved prices in that file are about Rs 114.53, Rs 175.83, and Rs 292.58. The page shows whole rupees. The high number in the file is 292.58. If the three tiles on the screen are different, read the tiles. The slider is live on this runtime model, so the first load can rescore it. Either way, say it is the saved runtime model. It is not Experiment B and it is not Experiment C.

On Data, the missing-price chart is computed from the stored weeks. 2020 is 29.3 percent missing. 2017 is 21.3 percent missing. The axis rounds to a whole percent, so a bar can look like 29 or 21. Say the computed shares. The price-weeks percent in the table under that chart is a different thing. It is not PICP.

On Features, mean absolute SHAP for `cultivation_progress_ratio_real` on the horizon-4 model is 0, so that empty chart is hidden.

## Page by page

The sidebar names are the pages. The address is `http://localhost:8501/?page=` plus that name. Training, Comparison, and Novelty are old names. They open the renamed pages. They are not extra pages.

The large title on a page is the line under the studio header. The small label in `page_header` is not drawn.

### Dashboard

Sidebar: Dashboard. Address: `http://localhost:8501/?page=Dashboard`. Large title: Price forecast.

The saved row is Carrot, Colombo, horizon 1 week, forecast week 2025-12-29, issued 2025-12-22. The file stores low 114.53, expected 175.83, high 292.58. Read the three tiles you actually see, because the cultivation slider rescores this runtime model on load. The default slider is 85 percent. Moving it changes this runtime price only.

This price is the saved runtime model. The small line under the tiles says the cultivation column on this runtime model was trained so a new achieved-over-target record can be scored again. That sentence is about this dashboard model. The card beside it is what Member 3 would read. The button opens a note. It does not start Member 3.

The caption above the tiles is the full-sample test, and it says it is not this runtime price. The coverage line under the tiles is the saved Experiment B horizon-4 coverage, 83.42 percent, and the page says that coverage is not this forecast.

If they ask where it is built, open `app/views/dashboard.py`, function `render`, and `app/lib/live_match.py`, function `match_forecast`. The scorer is `src/htfe/api/runtime.py`, function `score_progress`. The stored row is `app/data/live_forecast.json`.

Leave this out. Do not call this tile Experiment B or Experiment C. Do not say Experiment C was trained on a synthetic calendar. Do not call the slider's 85 and 100 a published hectare report. Do not say this interval reached 90 percent coverage.

### What we did

Sidebar: What we did. Address: `http://localhost:8501/?page=What%20we%20did`. Large title: Question, data, split, and the models that were saved.

I forecast the weekly wholesale price, in rupees per kilo, for carrot, leeks, and tomato. The score markets are Dambulla, Colombo, Meegoda, and Nuwara Eliya. The horizons are 1, 2, 4, 8, and 12 weeks. The series is not shuffled. Training issue weeks run through the end of 2022. The year 2023 only sets the interval adjustment. The test is issue weeks in 2024 and 2025. Experiment A sees price lags and the season. Experiment B adds origin weather, diesel, the exchange rate, inflation, and lagged finished-season extent. Experiment C is a separate matched-row test on national reports. SARIMAX is the fixed order (1, 1, 1) times (1, 0, 0, 52). LSTM was not run.

Point at Two functions and read the two blocks: the issue-week row, then the horizon score. Point at Question and design for the split. Point at Feature sets for A, B, and C. Point at Models saved after training for the joblib counts. Point at Tree calendar for the 2021, 2022, 2023 protocol, seed 42, and the three quantiles.

If they ask where it is built, open `app/views/training.py`, function `render`. The protocol sentence is `render_protocol_note` in `app/lib/experiment_c.py`.

Leave this out. Do not say one model wins all horizons. Do not say Experiment C sits inside the full-sample table. Do not say Optuna chose the trees.

### Results

Sidebar: Results. Address: `http://localhost:8501/?page=Results`. Large title: 2024–2025 full sample, observed prices only.

These lines are the full sample, observed prices only. LightGBM B at 4 weeks has MAE 110.14, MAPE 43.96 percent, n=1164. Raw coverage is 75.34 percent. Calibrated coverage is 83.42 percent. LightGBM A at 4 weeks has MAE 126.60. SARIMAX at 1 week has MAE 60.96, MAPE 22.94 percent, n=1200, and no interval. XGBoost B at 4 weeks has MAE 107.56 and MAPE 42.38 percent. MAE is rupees per kilo. MAPE is a percent. They are separate. A share below 90 percent is below the level the interval is aimed at.

The Experiment B table is the same full-sample run, one row per model and horizon. The A-against-B table is the same rows. A negative MAE change means B is closer. For LightGBM at 4 weeks that change is about minus 13 percent, from 126.60 to 110.14. Experiment C is not in this table. SARIMAX is not in the A-against-B comparison, because SARIMAX does not use the extra columns. Which model is closest at each horizon is on the next page.

If they ask where it is built, open `app/views/results.py`, function `render`. The rows come from `app/data/evaluation.json`. The coverage sentence is `b4_coverage_sentence` in `app/lib/scores.py`.

Leave this out. Do not put the matched-row C numbers in this table. Do not say the 12-week improvement is a weather-only result. Do not say calibrated coverage reached 90 percent. On this page the raw 75.34 and the calibrated 83.42 are both for the same LightGBM 4-week row. Say which one you are pointing at.

### Which model

Sidebar: Which model. Address: `http://localhost:8501/?page=Which%20model`. Large title: One horizon at a time.

The opening line is the decision. SARIMAX is closest at 1 week. XGBoost is closest at 4 weeks. Cultivation is a separate test. The cards under Final decision read the saved file: 1 week SARIMAX, MAE 60.96, n=1200; 2 weeks SARIMAX, MAE 83.45, n=1188; 4 weeks XGBoost, MAE 107.56, n=1164; 8 weeks LightGBM, MAE 110.05, n=1116; 12 weeks LightGBM, MAE 110.09, n=1071. The lowest model changes with the horizon.

On the 4-week box, last-value uses the same 1,164 rows as Experiment B LightGBM. MAE 142.91, MAPE 64.65 percent. The caption says seasonal naive uses 1,137 of those rows. The table beside that caption shows seasonal naive MAE 161.7 and MAPE 73.05 percent. LSTM was not run. PICP in this table is the calibrated interval. LightGBM shows 83.42 percent. That is below 90. Naive forecasts have no interval.

The crop and market slices are Experiment B LightGBM at 4 weeks, on the same scored rows. The common-key box further down is a shared-key subset. It does not replace the cards above. If you open the seasonal-naive expander, quote that expander on its own: horizon 4 there is n=1149, MAE 159.20, MAPE 73.52 percent. That is not the 1,137 in the caption.

If they ask where it is built, open `app/views/comparison.py`, function `render`. The decision cards use `outputs/evaluation/best_model_by_horizon.csv`, loaded in `app/lib/experiment_c.py`, function `parity`.

Leave this out. Do not say one model wins every horizon. Do not say 1,137 and 1,149 are the same count. Do not say the 83.42 percent interval reached 90 percent.

### Cultivation proof

Sidebar: Cultivation proof. Address: `http://localhost:8501/?page=Cultivation%20proof`. Large title: Matched national rows, separate from the full-sample table.

This page is a different test from the full sample. B-star and C share the same eligible rows. B-star does not see current progress. C adds the national as-of ratio. District rows were not features. The dashboard price is not this result.

Start at Given to Member 3. That box is the packet in `member3_forecast_c.json`. The measured PICP by horizon is 85.71, 91.29, 91.17, 93.74, and 93.56. Horizon 1 is 85.71, below the intended level of 90. The twelve rows under that table are horizon-1 forecasts. They are not the dashboard price. Progress on those rows is national achieved hectares divided by target hectares.

Then point at Experiment C. LightGBM at 1 week: B-star MAE 57.88, C MAE 59.75, n=462. C is higher by 3.23 percent, so C is worse at 1 week. Calibrated PICP is 85.71 percent, below 90. LightGBM at 4 weeks: B-star MAE 83.58, C MAE 80.04, n=453. C is lower by about 4.24 percent. Calibrated PICP is 91.17 percent. A positive MAE change means C is less accurate. A negative change means C is closer on that horizon. LightGBM is the comparison I talk through.

The shock caption says the 2024 first-quarter shock at horizon 4 is not evaluable, because this run has zero Experiment C horizon-4 rows in that window.

The ablation box is A and B only, at the stored 4-week numbers. C is not in that table. PICP there is the raw interval. The monthly re-forecast list on this page is empty. The dashboard slider is not Experiment C.

If they ask where it is built, open `app/views/novelty.py`, function `render`, and `app/lib/experiment_c.py`, functions `render_member3_handoff` and `render_experiment_c_scores`. The section title Given to Member 3 is in that second file.

Leave this out. Do not say C helps at horizon 1. Do not say C is the full-sample winner. Do not say Member 3 needs `predictions_c.csv`. Do not say the packet PICP is 87.23. Do not invent hectares. Do not say this test was trained on a synthetic calendar.

### Uncertainty

Sidebar: Uncertainty. Address: `http://localhost:8501/?page=Uncertainty`. Large title: 4-week test interval.

This chart is Experiment B LightGBM on the 4-week test. The quantiles were adjusted on 2023 only. The 1-week adjustment is 6.0922. The 4-week adjustment is 25.5959. Measured calibrated coverage is 83.42 percent, below the 90 percent intended level. The width card rounds to 439 rupees per kilo. The band on the chart is the 5 to 95 percent interval after the 2023 adjustment. The amber line is the realised wholesale price. The green line is the median forecast.

If they ask where it is built, open `app/views/uncertainty.py`, function `render`. The 83.42 percent comes from the LightGBM row in `app/data/comparison.json`. The adjustments come from `app/data/models.json`.

Leave this out. Do not say 83.42 percent means the interval reached 90 percent. Do not call this chart Experiment C. The Experiment C coverage pictures, if they are on the page, are labelled as the matched rows.

### Data

Sidebar: Data. Address: `http://localhost:8501/?page=Data`. Large title: Wholesale weeks that were scored.

This is the scored wholesale panel. There are 5,761 weekly rows after the filters. Missing prices stay missing. Only weeks with a published wholesale price are scored. The three cards are the issue-week split. Train is 4,380 rows, from 2016-01-04 to 2022-12-26. Calibration is 624 rows, from 2023-01-02 to 2023-12-25. Test is 1,260 rows, from 2024-01-01 to 2025-12-29. That 1,260 is the panel window. The scored 1-week count on Results is 1,200. They are not the same count.

The price chart is the real HARTI series for the crop and market you pick. Spikes in 2020 and 2021 stay as spikes. They are not filled in. The money plot is a finished season's extent ratio against the next season's average price. It is not this week's achieved hectares divided by target hectares.

The coverage table's price-weeks percent is how many weeks in that series have a published wholesale price. Carrot at Dambulla is 93.7 percent. Some series sit just above 90 percent on that column. That percent is not the forecast interval, and it is not the 90 percent coverage target.

The bar chart is the share of weeks with a blank price. The page computes 29.3 percent for 2020 and 21.3 percent for 2017. The axis rounds to a whole percent.

If they ask where it is built, open `app/views/data.py`, function `render`. The missing-share bars are `_missing_share_chart` in that file. The series are `app/data/series.json`. The split cards are `app/data/split.json`.

Leave this out. Do not say price-weeks percent is PICP. Do not say coverage reached 90 percent because a series shows 90.6 or 93.7. Do not say the money plot is current cultivation progress.

### Features

Sidebar: Features. Address: `http://localhost:8501/?page=Features`. Large title: Weather comes from the supply district.

A market price is not joined to weather at the market town. Carrot in Colombo uses Nuwara Eliya and Badulla. Tomato also uses Matale. A cultivation report is used only when its date is on or before the forecast Monday, and only when the crop and the official season match.

The as-of ratio is national achieved hectares already known by that week, divided by national target hectares. The file keeps the name `doa_cultivation_progress.csv`. District rows stay in the audit. The dashboard is the runtime model. It is not this Experiment C score.

The bars are a stored importance file, scaled so the one-week price lag is 1. The first bars are the recent price, the week inside the season, diesel, a rainfall sum, and the exchange rate. The line under the page says mean absolute SHAP for the cultivation-progress column on the horizon-4 model is 0, so that chart has no bar and is hidden. SHAP does not replace the matched-row MAE test.

If they ask where it is built, open `app/views/features.py`, function `render`. The hidden-chart sentence is `_cultivation_shap_sentence` in that file. The origin lists are `app/data/origin_map.json`.

Leave this out. Do not say the bar order proves weather, then macro, then extent. Do not say the cultivation column drives the horizon-4 model. Its mean absolute SHAP is 0.

### Member 1

This is a handoff, not a second forecast. Click Data fusion, or open `http://localhost:8501/?member=1`. Large title: Weekly panel passed to the forecast.

Member 1 prepares the weekly panel. Member 2 scores observed wholesale weeks and does not re-clean those files. On this branch `data/v1` is not present, and `data/member1` does not yet hold the four contract files, so the counters on this page stay at zero. Say that plainly. The scored panel you actually talk through is on the Data page.

The handoff rules in the box are the grain, crop by market by week, wholesale only, missing prices left missing, and weather joined through the origin map.

If they ask where it is built, open `app/views/member1.py`, function `render`.

Leave this out. Do not invent an extract that is not on disk. Do not say Member 1's own dashboard is running here. It stays on their branch.

### Member 3 and Member 4

Click Crop viability, or open `http://localhost:8501/?member=3`. Large title: Member 3 · Crop viability. The note says this branch is Member 2 only, and crop viability and the season simulator stay on the team branch.

Click Season simulation, or open `http://localhost:8501/?member=4`. Large title: Member 4 · Season simulation. Same note.

Say one sentence. These buttons show a note. They do not start the other apps. The file I give Member 3 is the packet on Cultivation proof, and it has to be copied by hand because `outputs/` is not in git.

If they ask where the note is built, open `app/lib/nav.py`, function `render_external_member`.

Leave this out. Do not demo a planting recommendation or a season simulator from this branch.

### Old names

Training opens What we did. Comparison opens Which model. Novelty opens Cultivation proof. If someone uses the old word, click the new name and keep talking.

## Sixty seconds

Open the studio before you walk in, so the first wait is already over.

1. Open Dashboard. Say this price is the saved runtime model, for Carrot in Colombo, one week ahead, week 2025-12-29. Read the three tiles on the screen. Say it is not the Experiment B score and not the Experiment C score.
2. Open Which model. Read the decision line. SARIMAX is closest at 1 week. XGBoost is closest at 4 weeks. The closest model changes with the horizon.
3. Open Results. Point at the 4-week lines. XGBoost B MAE 107.56. LightGBM B MAE 110.14, MAPE 43.96 percent, n=1164. LightGBM A was 126.60.
4. Open Cultivation proof. Point at B-star against C. At 1 week C is worse, 59.75 against 57.88, n=462. At 4 weeks C is closer, 80.04 against 83.58, n=453. Then point at Given to Member 3 and the packet PICP, starting at 85.71 for horizon 1.
5. Open Uncertainty. One sentence: calibrated coverage at 4 weeks is 83.42 percent, and that is below 90 percent.
