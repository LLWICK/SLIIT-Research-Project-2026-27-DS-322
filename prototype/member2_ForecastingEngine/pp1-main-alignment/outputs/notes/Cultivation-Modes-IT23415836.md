# Cultivation modes

Member 2 keeps two cultivation providers on one schema. They are not interchangeable, and they are not both inputs to the primary forecasting table.

## Observed monthly progress

A district-month file with year, month, crop, district, target_ha, and achieved_ha was not found.
Observed provider rows written: 0.
When that file is absent the observed provider writes an empty frame. `cultivation_progress_ratio`, `origin_target_hectares`, and `origin_achieved_hectares` stay null. Nothing is invented from price, rainfall, temperature, or zeros.
If a file is present, progress is summed achieved hectares divided by summed target hectares. The ratio is not clipped at 1. `cultivation_target_exceeded` is true only when that raw ratio is above 1. A forecast Monday uses a report only when its available date is on or before that Monday. Later months are not interpolated.

## Lagged seasonal extent on the primary table

These columns are real DCS extent for seasons that have already finished. They are not cultivation progress.

- `previous_season_extent`: last finished Yala or Maha, summed over the market's supply districts. Yala of year S is finished only after HARTI week 39 of S. Maha of year S is finished only after week 13 of S+1.
- `same_season_previous_year_extent`: the same season one year earlier, and only when that season has finished.
- `historical_mean_season_extent`: arithmetic mean of finished Yala and Maha extents before the forecast week. The current unfinished season is excluded.
- `extent_change_vs_previous_season`: `previous_season_extent` minus the finished season immediately before it, in hectares.

On the rebuilt primary table, `cultivation_progress_ratio` non-null rows: 0.
`previous_season_extent` non-null share of forecasts: 0.9774.
`same_season_previous_year_extent` non-null share: 0.9799.
`historical_mean_season_extent` non-null share: 1.0.
`extent_change_vs_previous_season` non-null share: 0.9727.
Crop-markets with no lagged extent in this table: none.
Series whose last finished season is missing for some weeks: carrot/thambuttegama (0.526). A blank DCS season stays null. A recorded 0 ha stays 0 ha and is not used to fill a blank season.
`cultivation_source` is `lagged_seasonal_extent` where any finished-extent feature is present, and `none` where none of them are. `cultivation_is_synthetic` is false. `cultivation_scenario` is null. `cultivation_source_name` stays empty when no monthly report exists.
The same season's final extent is not an input feature on this table.

## Synthetic calendar (prototype and sensitivity only)

Synthetic rows use the real final DCS `extent_ha` for that crop, district, and season, multiplied by a logistic planting curve. The curve is a research assumption. It is not an observed planting date and it is not achieved hectares divided by target hectares.

The planting windows below are configuration in `src/htfe/data/maps/crop_calendar.yaml`. They sit inside the existing HARTI season rule (Yala weeks 14-39, Maha the other weeks). They are not a record of when farmers planted.

- carrot Yala: planting weeks 14-22
- carrot Maha: planting weeks 40-48
- leeks Yala: planting weeks 14-22
- leeks Maha: planting weeks 40-48
- tomato Yala: planting weeks 14-22
- tomato Maha: planting weeks 40-48

Normalized u runs from 0 to 1 across the planting window. F(u) = (raw(u) - raw(0)) / (raw(1) - raw(0)), where raw(u) = 1 / (1 + exp(-k (u - m))).

- early: midpoint 0.4, steepness 8
- typical: midpoint 0.5, steepness 8
- late: midpoint 0.6, steepness 8

The default scenario is `typical`. The parquet contains all three scenarios so a sensitivity check can compare them.
Synthetic rows: 15417. Scenarios: early, late, typical.
Every synthetic row has `cultivation_source = synthetic_calendar` and `cultivation_is_synthetic = true`. `cultivation_progress` in this file is F(u), not achieved/target. `cultivation_reference_area` is that same season's final extent. `data_available_date` is the first HARTI Monday after the season has finished, which is after `week_start` for every planting-window row.

The file is `processed/latest/cultivation_synthetic.parquet`. It is not joined to `processed/latest/training_table.parquet`.

## Experiment names

A, B, C, and S now use the proposal names.

- A is price lags and seasonal features only. It has no weather, no macros, no cultivation progress, and no lagged extent.
- B is A plus origin-aligned historical weather, real diesel, USD/LKR, and inflation, plus lagged finished-season extent. Lagged extent is a historical production feature. It is not current cultivation progress.
- C is B plus observed within-season cultivation progress, summed achieved hectares over summed target hectares across supply districts, joined as-of. Ratios above 1 stay above 1, and `cultivation_target_exceeded` is set when the raw ratio is above 1.
- S is the synthetic calendar curve (early, typical, late). It is a separate prototype and sensitivity experiment. Its scores are not stored in the A/B/C accuracy table.

## What this does not show

Synthetic calendar progress is experiment S. It is prototype/sensitivity only and must not be cited as proof that observed cultivation progress improves accuracy.
S scores are not written into the A/B/C accuracy table.

## Weather weights

`origin_weight_method` is `extent_weighted_weather`.
The weather join reads origin points from `origin_map` (`weather`), not the market town. Colombo and Meegoda are not given a weather point named after the town. Weights use the lagged finished season's district extent, split across weather points that share a district. If that extent is missing, the join falls back to equal weights. `supply_weight` stays null. Measured trade shares such as 0.60/0.30/0.10 are not used.
`origin_rainfall_1w_mm` is the origin rainfall of the one complete week before the forecast Monday (`shift(1)`). It is not filled with zero. The four-week and eight-week sums still start from that same shift. `weather_missing_flag` remains the missingness of the four-week sum.
