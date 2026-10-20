# PP1 — two functions

Garusingarachchi Y.B. IT23415836. Member 2. J26-DS-322.

Say this in your own pace. Short sentences. The slides stay quiet. The studio is the proof.

## Where the two functions come from

CobwebProj, one folder up from this repo, has the locked notebooks and a zip. It does not have the slides.

The Member 2 slides are in `proosal presentation\Yenu\`: `proposal_pres_memeber 2 content.pptx` and `Member2_Forecasting_Engine_Slides.pptx`. The proposal is `Proposal Report\IT23415836_Proposal_Report.docx`.

Those files do not name a pair of two key functions. The slide called Specific objectives lists five jobs: build the crop by market by week dataset, compare SARIMA, LightGBM and XGBoost, make intervals, test cultivation with A/B/C, and expose an API. Section 6.2 of the proposal lists many "the system shall" lines. The work breakdown starts with a leak-safe crop by market by week panel, then spreads the modelling across later tasks.

Because no document names exactly two functions, these two are the ones this codebase actually implements. They line up with that panel task and with the model comparison.

1. Build the leakage-safe weekly forecast table.
2. Fit and score the direct multi-horizon forecast, and show which model is best at which horizon.

## 1. What are my two key functions?

The first function builds the weekly forecast table. A column gets in only if it was already public on the forecast week.

The second function fits and scores the direct multi-horizon forecast. That is Experiment A, Experiment B, SARIMAX, and a separate matched-row Experiment C. It then says which model is best at which horizon.

## 2. What input does each function receive?

Weekly forecast table. It takes the wholesale price, origin weather, published macros, lagged finished-season extent, and the national cultivation as-of ratio. The ratio is achieved hectares divided by target hectares. District cultivation rows stay out of the model. I read those columns on `outputs/panel/training_table.parquet`: price lags, `origin_rainfall_1w_mm`, `origin_mean_temp_4w_c`, `diesel_price_lkr_litre`, `usd_lkr_rate`, `inflation_rate_pct`, `previous_season_extent`, and `cultivation_progress_ratio`. The file has 54,243 rows and 62 columns.

Multi-horizon score. It takes that table, split by the week the forecast is issued. Train issue weeks run through 2022-12-31. Tree size is chosen on 2022 target weeks, then the trees are refit through 2022. 2023 only sets the interval adjustment. The test is issue weeks in 2024 and 2025. Experiment A sees price lags and the season. Experiment B adds origin weather, diesel, the USD/LKR rate, inflation, and lagged extent. SARIMAX is order (1, 1, 1) times (1, 0, 0, 52) on the price series. Experiment C is not mixed into that full sample. B* uses the Experiment B features on rows that have a national cultivation report. C adds the as-of ratio on those same rows.

## 3. What output does each function produce?

Weekly forecast table. One row per crop, market, issue week, and horizon. Nothing from a later week is in that row. The file is `outputs/panel/training_table.parquet`. The Data page and the Features page show the panel and the columns.

Multi-horizon score. Point error and interval scores for Experiment A, Experiment B, and SARIMAX on the full 2024–2025 sample, plus B* against C on the matched rows. The saved ranking is `outputs/evaluation/best_model_by_horizon.csv`. The Which model page reads that file. On the file I just opened, the lowest error at 1 week is SARIMAX and the lowest error at 4 weeks is XGBoost. The same file also names a best model at 2, 8, and 12 weeks. It is not the same name every time.

## 4. How does each function support the research objective?

The cobweb pattern is farmers planting into a high price, then a crash when that crop hits the market together. The objective is a weekly wholesale price forecast, in LKR per kg, for carrot, leeks, and tomato, in time for a planting decision, with a range around the point, and a fair test of whether this season's planting changes the forecast.

The table is what makes that forecast honest. Weather, macros, and the planting report cannot leak in from the future.

The score is what answers the objective. It says which model is closest at 1 week and at 4 weeks on the full sample. The matched-row test then asks whether the national as-of ratio helps, on the weeks that actually have a report. Those two questions stay separate.

## 5. What evidence proves I built it and understood it?

Open the studio and walk it. Do not read a result off the Dashboard. The Dashboard is the runtime model. It is not the accuracy test.

What we did. The block called What I built states the two functions, with one line of input and one line of output. Below that, the page shows the split, the feature sets, and the saved joblib files.

Results. Full-sample 2024–2025 table. From `app/data/evaluation.json`: SARIMAX at 1 week has MAE 60.96, n=1200. XGBoost B at 4 weeks has MAE 107.56, n=1164. LightGBM B at 4 weeks has MAE 110.14, MAPE 43.96%, and calibrated coverage 83.42%. The interval is aimed at 90%. 83.42% is the share we measured. It is below 90%.

Which model. The first sentence is read from `outputs/evaluation/best_model_by_horizon.csv`, not typed in by hand. On that file: 1 week SARIMAX, MAE 60.9553, n=1200. 2 weeks SARIMAX. 4 weeks XGBoost, MAE 107.5566, n=1164. 8 weeks LightGBM. 12 weeks LightGBM. Say that list if someone asks whether one model wins. One model does not win every horizon.

Cultivation proof. From `outputs/experiment_c/b_star_vs_c.csv`, LightGBM only: at 1 week C MAE is 59.75 and B* MAE is 57.88, n=462. C is worse at 1 week. At 4 weeks C MAE is 80.04 and B* MAE is 83.58, n=453. C is a separate test. It is not in the full-sample ranking. `outputs/figures/shock_note.json` says the 2024 Q1 shock is not evaluable for Experiment C, because that window has zero horizon-4 rows.

The table build is `src/htfe/training/build_training_table.py`. The fits are `run_experiment_a.py`, `run_experiment_b.py`, `run_experiment_c.py`, and the SARIMAX path stored with Experiment A.

## Slides, speech, and the studio

Slides are not the place for the proof. Use one or two quiet slides. Slide one: the two function names. Slide two: for each function, one line of input and one line of output. No MAE table. No claim that coverage hit 90%. No sentence that one model wins.

Say the limits out loud.

- Calibrated coverage for LightGBM B at 4 weeks is 83.42%. The target level is 90%. 83.42% is below that.
- Experiment C does not help at 1 week. 59.75 against 57.88, n=462.
- At 4 weeks on those matched rows, C is lower than B*, 80.04 against 83.58, n=453. That does not make C the full-sample winner. The full-sample 4-week winner is XGBoost B.
- The Dashboard number is the runtime model. It is not this test.
- The 2024 Q1 shock is not evaluable for C.
- SARIMAX, XGBoost, and LightGBM each win some horizon on the common-key table. None of them wins all five.

The studio is the evidence. Start on What we did, then Results, then Which model, then Cultivation proof. Point at the decision sentence on Which model, then at the table under it. If they ask about the live price, open the Dashboard and say it is a different thing from the test.
