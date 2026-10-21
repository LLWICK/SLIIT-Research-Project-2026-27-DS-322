# Experiment A vs B

Test issue weeks are 2024-2025. Markets are Dambulla, Colombo, Meegoda, and Nuwara Eliya. Horizons are 1, 2, 4, 8, and 12 weeks. Scores use the same observed-price mask as Experiment A.

Experiment A is price lags and seasonal features. Experiment B adds origin-aligned historical weather, real as-of diesel, USD/LKR, and CCPI inflation, and lagged finished-season extent (`previous_season_extent`, `same_season_previous_year_extent`, `historical_mean_season_extent`, `extent_change_vs_previous_season`). Lagged extent is historical production, not current cultivation progress. `cultivation_progress_ratio` is null and is not a feature. The synthetic calendar is not a feature. Experiment C is a separate matched-row run and is not in this table. S was not run.

MAE and RMSE percent change is (B − A) / A × 100. A negative percent is a lower error for B. PICP and width are the raw 0.05/0.95 intervals. PICP CQR and width CQR apply the 2023 split-conformal adjustment to the 2024-2025 test. SARIMAX interval columns stay blank.

reused 8811 Experiment A SARIMAX rows. SARIMAX was not refit. Order is (1, 1, 1) x (1, 0, 0, 52).

## 4-week MAE

B improves 4-week MAE for LightGBM: B is 110.14 and A is 126.60 (-13.00% MAE).

B improves 4-week MAE for XGBoost: B is 107.56 and A is 121.12 (-11.20% MAE).

## 2024-2025 test

| Model | H | MAE A | MAE B | MAE % | RMSE A | RMSE B | RMSE % | MAPE A | MAPE B | Pinball A | Pinball B | PICP A | PICP B | Width A | Width B | PICP CQR A | PICP CQR B | Width CQR A | Width CQR B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lightgbm | 1 | 82.28 | 77.40 | -5.93 | 136.89 | 131.42 | -4.00 | 31.76 | 29.19 | 41.14 | 38.70 | 84.17 | 85.83 | 330.23 | 335.07 | 87.25 | 88.75 | 344.32 | 347.26 |
| lightgbm | 2 | 100.86 | 93.48 | -7.32 | 165.69 | 161.57 | -2.49 | 39.66 | 32.46 | 50.43 | 46.74 | 86.11 | 86.20 | 424.64 | 394.85 | 85.77 | 87.21 | 421.67 | 405.16 |
| lightgbm | 4 | 126.60 | 110.14 | -13.00 | 191.37 | 172.73 | -9.74 | 56.03 | 43.96 | 63.30 | 55.07 | 79.38 | 75.34 | 508.24 | 387.99 | 80.58 | 83.42 | 516.41 | 439.50 |
| lightgbm | 8 | 153.99 | 110.05 | -28.53 | 213.78 | 163.22 | -23.65 | 87.42 | 54.53 | 77.00 | 55.02 | 79.57 | 71.95 | 546.47 | 287.79 | 82.80 | 81.99 | 589.77 | 426.59 |
| lightgbm | 12 | 159.26 | 110.09 | -30.87 | 219.06 | 163.40 | -25.41 | 92.84 | 51.97 | 79.63 | 55.04 | 64.80 | 67.69 | 525.24 | 398.11 | 78.62 | 87.68 | 651.07 | 602.07 |
| sarimax | 1 | 60.96 | 60.96 | 0.00 | 107.81 | 107.81 | 0.00 | 22.94 | 22.94 |  |  |  |  |  |  |  |  |  |  |
| sarimax | 2 | 83.45 | 83.45 | 0.00 | 139.32 | 139.32 | 0.00 | 32.44 | 32.44 |  |  |  |  |  |  |  |  |  |  |
| sarimax | 4 | 121.48 | 121.48 | 0.00 | 188.44 | 188.44 | 0.00 | 51.80 | 51.80 |  |  |  |  |  |  |  |  |  |  |
| sarimax | 8 | 165.98 | 165.98 | 0.00 | 241.34 | 241.34 | 0.00 | 92.28 | 92.28 |  |  |  |  |  |  |  |  |  |  |
| sarimax | 12 | 180.98 | 180.98 | 0.00 | 258.63 | 258.63 | 0.00 | 111.69 | 111.69 |  |  |  |  |  |  |  |  |  |  |
| xgboost | 1 | 79.17 | 76.80 | -2.99 | 133.22 | 131.80 | -1.07 | 30.44 | 28.68 | 39.59 | 38.40 | 81.67 | 82.92 | 305.71 | 310.50 | 86.00 | 86.00 | 327.85 | 323.02 |
| xgboost | 2 | 96.44 | 88.96 | -7.76 | 162.23 | 158.08 | -2.56 | 36.77 | 31.68 | 48.22 | 44.48 | 82.41 | 86.20 | 401.78 | 397.70 | 82.91 | 84.60 | 403.10 | 389.65 |
| xgboost | 4 | 121.12 | 107.56 | -11.20 | 186.65 | 170.78 | -8.50 | 53.09 | 42.38 | 60.56 | 53.78 | 76.80 | 82.47 | 511.57 | 485.34 | 77.23 | 81.53 | 514.37 | 481.67 |
| xgboost | 8 | 150.54 | 112.11 | -25.53 | 205.89 | 167.50 | -18.65 | 84.24 | 52.57 | 75.27 | 56.06 | 79.84 | 74.64 | 544.29 | 352.50 | 81.63 | 78.58 | 563.86 | 401.27 |
| xgboost | 12 | 162.75 | 114.36 | -29.73 | 220.92 | 165.79 | -24.95 | 97.19 | 57.60 | 81.38 | 57.18 | 62.75 | 70.03 | 482.88 | 349.68 | 76.84 | 82.17 | 587.03 | 493.00 |

## Crop and market, best point model

The model at each of horizons 1, 4, and 12 is the one with the lowest pooled test MAE in Experiment B.

| Horizon | Model | Crop | Market | MAE A | MAE B | MAPE A | MAPE B |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | sarimax | carrot | Colombo | 75.61 | 75.61 | 23.05 | 23.05 |
| 1 | sarimax | carrot | Dambulla | 86.87 | 86.87 | 23.77 | 23.77 |
| 1 | sarimax | carrot | Meegoda | 80.29 | 80.29 | 19.21 | 19.21 |
| 1 | sarimax | carrot | Nuwara Eliya | 81.80 | 81.80 | 22.21 | 22.21 |
| 1 | sarimax | leeks | Colombo | 37.87 | 37.87 | 18.82 | 18.82 |
| 1 | sarimax | leeks | Dambulla | 36.85 | 36.85 | 17.04 | 17.04 |
| 1 | sarimax | leeks | Meegoda | 33.75 | 33.75 | 13.96 | 13.96 |
| 1 | sarimax | leeks | Nuwara Eliya | 29.36 | 29.36 | 13.68 | 13.68 |
| 1 | sarimax | tomato | Colombo | 62.20 | 62.20 | 25.69 | 25.69 |
| 1 | sarimax | tomato | Dambulla | 68.55 | 68.55 | 35.89 | 35.89 |
| 1 | sarimax | tomato | Meegoda | 70.36 | 70.36 | 29.23 | 29.23 |
| 1 | sarimax | tomato | Nuwara Eliya | 67.82 | 67.82 | 32.79 | 32.79 |
| 4 | xgboost | carrot | Colombo | 144.24 | 127.75 | 51.25 | 41.37 |
| 4 | xgboost | carrot | Dambulla | 151.05 | 138.46 | 51.51 | 43.81 |
| 4 | xgboost | carrot | Meegoda | 162.69 | 145.48 | 46.86 | 36.88 |
| 4 | xgboost | carrot | Nuwara Eliya | 160.69 | 142.21 | 55.14 | 42.99 |
| 4 | xgboost | leeks | Colombo | 69.12 | 60.14 | 33.00 | 26.05 |
| 4 | xgboost | leeks | Dambulla | 63.51 | 56.34 | 29.50 | 23.13 |
| 4 | xgboost | leeks | Meegoda | 71.99 | 66.16 | 29.84 | 24.21 |
| 4 | xgboost | leeks | Nuwara Eliya | 72.12 | 58.51 | 33.61 | 24.64 |
| 4 | xgboost | tomato | Colombo | 141.68 | 129.88 | 68.57 | 55.07 |
| 4 | xgboost | tomato | Dambulla | 134.50 | 113.43 | 83.75 | 65.29 |
| 4 | xgboost | tomato | Meegoda | 140.74 | 128.56 | 60.68 | 48.05 |
| 4 | xgboost | tomato | Nuwara Eliya | 141.92 | 123.81 | 95.51 | 78.76 |
| 12 | lightgbm | carrot | Colombo | 189.69 | 137.85 | 96.16 | 54.47 |
| 12 | lightgbm | carrot | Dambulla | 204.96 | 141.74 | 96.84 | 53.14 |
| 12 | lightgbm | carrot | Meegoda | 212.25 | 151.88 | 87.22 | 49.87 |
| 12 | lightgbm | carrot | Nuwara Eliya | 213.19 | 155.42 | 105.34 | 61.24 |
| 12 | lightgbm | leeks | Colombo | 96.53 | 66.07 | 52.57 | 33.45 |
| 12 | lightgbm | leeks | Dambulla | 92.21 | 64.43 | 50.33 | 31.43 |
| 12 | lightgbm | leeks | Meegoda | 102.17 | 67.90 | 47.79 | 27.83 |
| 12 | lightgbm | leeks | Nuwara Eliya | 101.03 | 60.70 | 55.71 | 29.17 |
| 12 | lightgbm | tomato | Colombo | 187.83 | 130.05 | 120.04 | 63.50 |
| 12 | lightgbm | tomato | Dambulla | 162.96 | 105.27 | 143.15 | 74.27 |
| 12 | lightgbm | tomato | Meegoda | 183.91 | 128.48 | 103.72 | 59.13 |
| 12 | lightgbm | tomato | Nuwara Eliya | 164.25 | 110.74 | 158.60 | 87.98 |

## Data caveats

Published CCPI headline year-on-year uses base 2013=100 through January 2023 and base 2021=100 from February 2023. The index was not rescaled from one base onto the other. This is a break in the published inflation series, not a model failure.

Inflation available dates use reference month-end plus 21 days. The reference month is not treated as the publication date. A forecast issued in the first three weeks of a month does not see that month's CCPI, and it does not see the previous month's CCPI until the 21-day lag has elapsed. Diesel uses the CPC Lanka Auto Diesel revision date. USD/LKR uses the CBSL observation date.
