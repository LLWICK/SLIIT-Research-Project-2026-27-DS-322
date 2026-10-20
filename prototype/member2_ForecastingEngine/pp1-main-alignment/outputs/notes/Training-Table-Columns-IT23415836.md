# Training table columns

Each row is one forecast: one crop, one wholesale market, one Monday, one horizon.
Rows: 54243. Horizons: [1, 2, 4, 8, 12].
Cultivation: missing_no_district_monthly_file. Economics: joined:macros_published.csv.
Share of rows with an observed future price: 0.89.
Share of rows missing the 4-week origin rainfall sum: 0.006.

The future price is `target_price_lkr_kg`. It is not an input.
Seasonal Census extent is not copied into `cultivation_progress_ratio`.
Lagged extent columns use only seasons that have already finished.
Synthetic calendar progress is not on this table. It is in `cultivation_synthetic.parquet` and is not evidence about observed cultivation progress.

| Column | Non-null share | Role |
| --- | ---: | --- |
| `crop_id` | 1.0 | A identity. Not a future value. |
| `market_id` | 1.0 | A identity. Wholesale market, not the growing district. |
| `forecast_date` | 1.0 | Monday on which inputs are frozen. Same value as forecast_issue_week. |
| `forecast_issue_week` | 1.0 | Monday the forecast is issued. Every time-varying input is known on or before this Monday. |
| `horizon_weeks` | 1.0 | How many weeks ahead the label sits. Same value as forecast_horizon_weeks. |
| `forecast_horizon_weeks` | 1.0 | Weeks from forecast_issue_week to target_week. The label is the price at target_week. |
| `target_week` | 1.0 | Future Monday. Known as a calendar date. The label is the price of this week, not an input. |
| `target_price_lkr_kg` | 0.89 | Label only. Blank when that future week was not published. |
| `target_is_observed` | 1.0 | Score only rows where this is true. |
| `target_week_of_year` | 1.0 |  |
| `last_observed_price_lkr_kg` | 0.998 | Latest published price strictly before forecast_date. |
| `price_lag_1w` | 0.889 | Published price one week before forecast_date. Blank if that week was not published. |
| `price_lag_2w` | 0.887 |  |
| `price_lag_4w` | 0.884 |  |
| `price_lag_8w` | 0.877 |  |
| `price_lag_52w` | 0.796 |  |
| `price_mean_4w` | 0.914 |  |
| `price_mean_8w` | 0.92 |  |
| `price_change_4w_pct` | 0.835 | Change from price_lag_4w to price_lag_1w. Both must be published. |
| `season` | 1.0 |  |
| `season_transition` | 1.0 |  |
| `forecast_week_of_year` | 1.0 |  |
| `week_within_season` | 1.0 |  |
| `origin_rainfall_1w_mm` | 0.998 | Origin-district rain in the one complete week before forecast_date. Missing rain is left missing. |
| `origin_rainfall_sum_4w_mm` | 0.994 | Origin-district rain over the four complete weeks before forecast_date. The sum starts after shift(1). |
| `origin_rainfall_sum_8w_mm` | 0.988 |  |
| `origin_mean_temp_4w_c` | 0.994 |  |
| `origin_rainfall_anomaly_4w_mm` | 0.688 | That four-week sum minus the median of earlier years at the same week. Blank until three earlier years exist. |
| `diesel_price_lkr_litre` | 1.0 | Empty until a dated diesel series is on disk. Not invented. |
| `usd_lkr_rate` | 1.0 | Empty until a dated exchange-rate series is on disk. |
| `inflation_rate_pct` | 1.0 | Empty until a dated published inflation series is on disk. |
| `origin_target_hectares` | 0.0 | Empty until district-month target hectares exist. Not copied from seasonal extent. |
| `origin_achieved_hectares` | 0.0 | Empty until district-month achieved hectares exist. Not copied from seasonal extent. |
| `cultivation_progress_ratio` | 0.0 | Sum of achieved divided by sum of target, not clipped at 1. Empty when the monthly file is absent. Not the synthetic logistic. |
| `cultivation_target_exceeded` | 0.0 | True only when the raw achieved/target ratio is above 1. Empty when progress is empty. |
| `previous_season_extent` | 0.978 | Supply-district extent of the last Yala or Maha that has already finished. Not the current season. |
| `same_season_previous_year_extent` | 0.98 | Extent of the same season one year earlier, included only after that season has finished. |
| `historical_mean_season_extent` | 1.0 | Mean of finished Yala and Maha supply extents before forecast_date. The unfinished season is excluded. |
| `extent_change_vs_previous_season` | 0.973 | previous_season_extent minus the finished season immediately before it, in hectares. |
| `cultivation_source` | 1.0 | observed_monthly, lagged_seasonal_extent, or none. synthetic_calendar is not used on this table. |
| `cultivation_is_synthetic` | 1.0 | False on this table. Synthetic rows are only in cultivation_synthetic.parquet. |
| `cultivation_scenario` | 0.0 | Null on this table. early, typical, and late exist only on the synthetic file. |
| `achieved_hectares_change` | 0.0 |  |
| `cultivation_report_age_days` | 0.0 | forecast_date minus the report availability date. |
| `cultivation_report_month` | 0.0 |  |
| `cultivation_available_at` | 0.0 |  |
| `observation_date` | 1.0 |  |
| `price_available_at` | 1.0 | Day before forecast_date. The current week's average is not treated as known on Monday. |
| `source_name` | 1.0 |  |
| `source_record_id` | 1.0 |  |
| `cultivation_source_name` | 0.0 |  |
| `price_type` | 1.0 |  |
| `price_unit` | 1.0 |  |
| `supply_origin_mapping_version` | 1.0 |  |
| `price_missing_flag` | 1.0 |  |
| `weather_missing_flag` | 1.0 |  |
| `cultivation_missing_flag` | 1.0 |  |
| `diesel_missing_flag` | 1.0 |  |
| `usd_missing_flag` | 1.0 |  |
| `inflation_missing_flag` | 1.0 |  |
| `split` | 1.0 |  |
| `is_shock` | 1.0 |  |
