# Feature coverage audit

Primary table: `processed/latest/training_table.parquet`.
Shares are non-null fractions of forecast rows (crop, market, issue week, horizon).
Weather coverage in the origin audit is computed once per issue week, not once per horizon.

Diesel, USD/LKR, and inflation are joined as-of from `processed/macros_published.csv`. Annual World Bank series were not used.

Observed cultivation data unavailable. `cultivation_progress_ratio` is null. Experiment C is not trained.

`origin_rainfall_1w_mm` was checked against the panel: it is `weather_precip_mm` shifted by one week inside each crop-market. It is the previous complete week, not the target week. Missing rain stays missing.
`origin_weight_method` is `extent_weighted_weather` on every supply-origin row. `supply_weight` is null.

| Group | Column | Non-null share |
| --- | --- | ---: |
| price_lag | `price_lag_1w` | 0.8891 |
| price_lag | `price_lag_2w` | 0.8873 |
| price_lag | `price_lag_4w` | 0.8838 |
| price_lag | `price_lag_8w` | 0.8771 |
| price_lag | `price_lag_52w` | 0.7962 |
| price_lag | `price_mean_4w` | 0.9143 |
| price_lag | `price_mean_8w` | 0.9195 |
| price_lag | `price_change_4w_pct` | 0.8348 |
| season | `season` | 1.0 |
| season | `season_transition` | 1.0 |
| season | `forecast_week_of_year` | 1.0 |
| season | `week_within_season` | 1.0 |
| origin_rainfall_1w | `origin_rainfall_1w_mm` | 0.9981 |
| origin_rainfall_4w | `origin_rainfall_sum_4w_mm` | 0.9942 |
| origin_temp | `origin_mean_temp_4w_c` | 0.9942 |
| diesel | `diesel_price_lkr_litre` | 1.0 |
| usd_lkr | `usd_lkr_rate` | 1.0 |
| inflation | `inflation_rate_pct` | 1.0 |
| lagged_extent | `previous_season_extent` | 0.9776 |
| lagged_extent | `same_season_previous_year_extent` | 0.9801 |
| lagged_extent | `historical_mean_season_extent` | 1.0 |
| lagged_extent | `extent_change_vs_previous_season` | 0.9728 |
| cultivation_progress_ratio | `cultivation_progress_ratio` | 0.0 |

## Origin weather

| Crop | Market | Origin districts | Weight method | Rain 1w | Rain 4w | Temp 4w |
| --- | --- | --- | --- | ---: | ---: | ---: |
| carrot | Nuwara Eliya | Nuwara Eliya | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| carrot | Dambulla | Nuwara Eliya; Badulla | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| carrot | Colombo | Nuwara Eliya; Badulla | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| carrot | Meegoda | Nuwara Eliya; Badulla | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| carrot | Kandy | Kandy; Nuwara Eliya | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| carrot | Keppetipola | Badulla | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| carrot | Thambuttegama | Anuradhapura | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| leeks | Nuwara Eliya | Nuwara Eliya | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| leeks | Dambulla | Nuwara Eliya; Badulla | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| leeks | Colombo | Nuwara Eliya; Badulla | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| leeks | Meegoda | Nuwara Eliya; Badulla | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| leeks | Kandy | Nuwara Eliya; Kandy | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| leeks | Keppetipola | Badulla; Nuwara Eliya | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| leeks | Thambuttegama | Nuwara Eliya | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| tomato | Nuwara Eliya | Nuwara Eliya; Matale | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| tomato | Dambulla | Matale; Nuwara Eliya; Badulla | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| tomato | Colombo | Matale; Badulla; Nuwara Eliya | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| tomato | Meegoda | Matale; Badulla; Nuwara Eliya | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| tomato | Kandy | Kandy; Matale | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| tomato | Keppetipola | Badulla | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |
| tomato | Thambuttegama | Anuradhapura; Matale | extent_weighted_weather | 0.9981 | 0.9942 | 0.9942 |

Equal weight is only the fallback when lagged finished-season extent is missing. It is not a measured 0.60/0.30/0.10 trade share.
