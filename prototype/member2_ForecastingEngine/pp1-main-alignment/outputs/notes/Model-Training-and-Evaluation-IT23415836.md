# How the models were trained, what they output, and how they were scored

Member 2. Carrot, leeks, and tomato. Score markets: Dambulla, Colombo, Meegoda, Nuwara Eliya. The label is the wholesale price, in LKR per kg, at the target week.

## How training was done

Each row is one crop, one market, one forecast Monday, and one horizon. Horizons are 1, 2, 4, 8, and 12 weeks. A forecast Monday may use only information already published by that Monday. The target price is the HARTI wholesale price at the later week. Missing prices are not filled with zero, and a row is scored only when both the last observed price and the target price were originally published.

The split is chronological. Issue weeks through 2022-12-31 are train. Issue weeks in 2023 are calibration. Issue weeks in 2024 and 2025 are test. The series is not shuffled.

The model predicts the log change from the last observed price, then converts it back with `last price × exp(prediction)`.

- Experiment A uses price lags and season only.
- Experiment B adds origin-district weather, diesel, USD/LKR, inflation, and the cultivated extent of seasons that have already finished.
- Experiment C adds the national as-of cultivation ratio on the rows that have a real DOA report. It is a matched-row test against B* and is not mixed into the full-sample A/B table. District rows are audit-only. LSTM was not run.

SARIMAX is a univariate seasonal baseline, order (1,1,1)×(1,0,0,52). It does not take the weather or macro columns, so its error is the same in A and B.

LightGBM and XGBoost are quantile models at 0.05, 0.50, and 0.95. Hyperparameters are the notebook defaults, seed 42. Optuna is not used. The tree count is chosen by early stopping on 2022 target weeks after fitting through 2021, then the trees are refit through 2022. 2023 is used only for the interval.

Intervals are the 0.05 and 0.95 forecasts. Split conformal calibration adds a constant, fit only on 2023, to both tails. The 2024–2025 prices are not used to choose that constant.

## What the output is

For each scored row the saved prediction file has the issue week, the target week, the horizon, the true price, the median forecast, and the lower and upper forecast. The studio price is the horizon-1 Experiment B LightGBM median for the latest test week of that crop and market, with the 2023 conformal adjustment on the interval. Sending cultivation hectares does not change it, because that column was not in the model.

## How performance was measured

On the 2024–2025 test rows, for each model and horizon:

- MAE and RMSE in LKR per kg
- MAPE in percent
- Pinball loss of the median forecast
- PICP, the share of true prices inside the 5–95% interval, before and after conformal calibration
- Mean interval width, before and after calibration

There is no accuracy, F1, or confusion matrix. The target is a price, not a class.

## Experiment B test scores

| Model | Horizon | n | MAE | RMSE | MAPE | Pinball | PICP raw | PICP calibrated | Width calibrated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lightgbm | 1 | 1200 | 77.40 | 131.42 | 29.19 | 38.70 | 85.83 | 88.75 | 347.26 |
| lightgbm | 2 | 1188 | 93.48 | 161.57 | 32.46 | 46.74 | 86.20 | 87.21 | 405.16 |
| lightgbm | 4 | 1164 | 110.14 | 172.73 | 43.96 | 55.07 | 75.34 | 83.42 | 439.50 |
| lightgbm | 8 | 1116 | 110.05 | 163.22 | 54.53 | 55.02 | 71.95 | 81.99 | 426.59 |
| lightgbm | 12 | 1071 | 110.09 | 163.40 | 51.97 | 55.04 | 67.69 | 87.68 | 602.07 |
| sarimax | 1 | 1200 | 60.96 | 107.81 | 22.94 |  |  |  |  |
| sarimax | 2 | 1188 | 83.45 | 139.32 | 32.44 |  |  |  |  |
| sarimax | 4 | 1164 | 121.48 | 188.44 | 51.80 |  |  |  |  |
| sarimax | 8 | 1116 | 165.98 | 241.34 | 92.28 |  |  |  |  |
| sarimax | 12 | 1071 | 180.98 | 258.63 | 111.69 |  |  |  |  |
| xgboost | 1 | 1200 | 76.80 | 131.80 | 28.68 | 38.40 | 82.92 | 86.00 | 323.02 |
| xgboost | 2 | 1188 | 88.96 | 158.08 | 31.68 | 44.48 | 86.20 | 84.60 | 389.65 |
| xgboost | 4 | 1164 | 107.56 | 170.78 | 42.38 | 53.78 | 82.47 | 81.53 | 481.67 |
| xgboost | 8 | 1116 | 112.11 | 167.50 | 52.57 | 56.06 | 74.64 | 78.58 | 401.27 |
| xgboost | 12 | 1071 | 114.36 | 165.79 | 57.60 | 57.18 | 70.03 | 82.17 | 493.00 |

## What the scores say

At 1 week, SARIMAX MAE is 60.96, LightGBM B is 77.40, and XGBoost B is 76.80. At 4 weeks, LightGBM B is 110.14, XGBoost B is 107.56, and SARIMAX is 121.48. At 12 weeks, LightGBM B is 110.09 and SARIMAX is 180.98. LightGBM MAE at 4 weeks changes by -13.00% from A (126.60) to B (110.14). The 4-week LightGBM calibrated coverage is 83.42%, which is below the intended 90%. Read the table above for the full set. Do not quote a number that is not in that table.

Experiment C is not in this full-sample table. The nominal interval is 90%. Report the measured calibrated PICP. Do not describe that measured share as 90% unless it is at least 90.
