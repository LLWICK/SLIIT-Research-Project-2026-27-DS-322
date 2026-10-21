# Data layer

Shared warehouse for carrot, leeks, and tomato in Colombo, Meegoda, Nuwara Eliya, and Dambulla. The other members read one view. They do not join the source tables themselves.

The database is PostgreSQL on Neon, schema `curated`, database `neondb`. A local DuckDB file is the staging copy. Dated Parquet files under `data_layer/export/` are the backup. The HARTI workbook and the CBSL file used to rebuild those tables are in `data_layer/source/`.

## Connect

Use `NEON_READER_URL` from the project-root `.env`. That login is `cobweb_reader`. It can `SELECT` and cannot insert or change tables.

```sql
SELECT *
FROM curated.crop_market_week
WHERE year = 2025
  AND week = 1
  AND market_code = 'colombo'
  AND crop_code = 'carrot'
  AND price_type = 'wholesale';
```

That row is Friday 3 January 2025 through Thursday 9 January 2025. The wholesale carrot price is 244.44.

Neon sleeps after a few minutes with no queries. The first connection can take a short time to wake. Use the direct host in the connection string, not a pooled host. SSL is required.

Do not commit `.env`. Do not share `NEON_DATABASE_URL`. That owner login reloads tables. Running `python -m data_layer.publish` with no argument also rotates the reader password and breaks the connection string the others already have.

## Dashboard

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

The page reads `curated.crop_market_week` and `curated.cultivation_extent` with the reader login. It does not change data. The first load waits if Neon is asleep.

## The view

`curated.crop_market_week` has 15,024 rows: one row for each HARTI week, price type, market, and crop from 2015 through 2026. Every row has `row_kind = 'harti_week'`.

| Column | Meaning |
| --- | --- |
| `price_type` | `retail` or `wholesale` |
| `market_code`, `market_name` | `colombo`, `meegoda`, `nuwara_eliya`, `dambulla` |
| `crop_code`, `crop_name` | `carrot`, `leeks`, `tomato` |
| `year`, `week` | HARTI year and week number |
| `week_start`, `week_end` | Friday and the following Thursday |
| `season_code`, `season_label` | Maha or Yala when all seven days are in one season. Empty when the week crosses a season boundary |
| `harti_price` | Workbook weekly price. Empty when that cell is blank, when the week is 53, or when the year is 2026 |
| `harti_is_missing` | True when the workbook has the cell and it is blank. Empty when the workbook has no such week |
| `cbsl_price` | Average of the CBSL daily prices inside that Friday–Thursday week, using only days that have a price |
| `cbsl_is_missing` | True when the week has CBSL rows and every price is blank |
| `extent_report_date`, `extent_as_of_date`, `extent_season_label` | The national extent snapshot attached to this week |
| `extent_target_ha`, `extent_achievement_ha` | Sri Lanka target and achievement, hectares |
| `extent_target_missing`, `extent_achievement_missing` | True when the report printed a dash. A dash is missing, not zero |
| `nuwara_eliya_t2m_c`, `badulla_t2m_c`, `dambulla_t2m_c` | Mean daily temperature (°C) over observed days |
| `nuwara_eliya_precip_mm`, `badulla_precip_mm`, `dambulla_precip_mm` | Sum of daily precipitation (mm) over observed days |
| `nuwara_eliya_days`, `badulla_days`, `dambulla_days` | How many days in the week had a weather observation |

HARTI and CBSL stay in separate columns. A week can hold both, because the week of 26 December 2025 through 1 January 2026 contains a 2025 HARTI price and a 2026 CBSL day. Do not add them together or copy one into the other.

## Week rule

A HARTI week runs Friday through Thursday. It belongs to the calendar year that contains at least four of those seven days. Week 1 is the first such week of that year.

Checked against the bulletin covers:

| Year | Week | Dates |
| --- | --- | --- |
| 2024 | 1 | 29 December 2023 – 4 January 2024 |
| 2024 | 52 | 20–26 December 2024 |
| 2024 | 53 | 27 December 2024 – 2 January 2025 |
| 2025 | 1 | 3–9 January 2025 |
| 2025 | 2 | 10–16 January 2025 |
| 2025 | 52 | 26 December 2025 – 1 January 2026 |
| 2026 | 1 | 2–8 January 2026 |

2018 and 2024 have 53 weeks. The workbook columns stop at W52, so week 53 has no HARTI price. That empty price is not a blank cell: `harti_is_missing` is empty too.

These are not ISO weeks and not Monday–Sunday weeks.

## What each source covers

### HARTI weekly prices

`curated.harti_price_weekly` has 13,104 rows from the HARTI workbook (`Retail - W` and `Wholesale - W`). Retail runs 2015–2025. Wholesale runs 2016–2025. Weeks are 1–52 only. 915 cells are blank and kept as missing. Nothing is filled in.

The retail Colombo workbook value is not the same number as the bulletin’s “Colombo & Suburbs” average. The week identity was checked with the wholesale Colombo carrot averages (2025 week 1 is 244.44, week 2 is 179). Use the workbook value.

Sheet spellings stored under the standard codes: `N'ELIYA` and `NUWARA ELIYA` are `nuwara_eliya`; `MEEGODA(DEC)` is `meegoda`; `TOMATOES` is `tomato`.

### CBSL daily prices, 2026

`curated.cbsl_price_daily` has 1,544 rows from 1 January 2026 through 16 October 2026. Crops are carrot and tomato. Markets are Colombo and Dambulla. Price types are retail and wholesale. The unit is `Rs/kg`. Pettah is stored as Colombo.

Leeks, Meegoda, and Nuwara Eliya are not in this file, so their `cbsl_price` is always empty. There is no CBSL series before 2026.

### Cultivation extent

`curated.cultivation_extent` has 472 rows. The view uses only the Sri Lanka total (`region_level = 'national'`). The same national total is the supply signal for all four markets. District rows are still in the table if you need them. Mahaweli and inter-provincial areas stay as their own regions. Meegoda’s administrative district is Colombo and Dambulla’s is Matale. Those districts are where the markets sit, not the supply catchments.

A crop page is stored only when the area rows add up to the Sri Lanka row, and each group’s area rows add up to its subtotal when that subtotal is printed. A blank group heading is a label, not a failed total. A printed 0 is zero. A dash is missing.

National snapshots currently stored:

| Report date | Season | As at | Crop | Target (ha) | Achievement (ha) | File |
| --- | --- | --- | --- | --- | --- | --- |
| 2022-04-15 | 2021/22 Maha | 2022-03-31 | carrot | 1743 | 1866 | maha-april-2022.pdf |
| 2024-12-15 | 2024/25 Maha | 2024-11-30 | leeks | 1326 | 418 | maha-december-2024-25.pdf |
| 2025-02-15 | 2024/25 Maha | 2025-01-31 | leeks | 1329 | 906 | maha-february-2024-25.pdf |
| 2025-04-15 | 2024/25 Maha | 2025-03-31 | carrot | 2001 | 2178 | maha-2024-25-final.pdf |
| 2025-06-15 | 2025 Yala | 2025-05-31 | leeks | 1322 | 471 | yala-june-2025.pdf |
| 2025-07-15 | 2025 Yala | 2025-06-30 | leeks | 1333 | 772 | yala-july-2025.pdf |
| 2025-08-15 | 2025 Yala | 2025-07-31 | leeks | 1333 | 1120 | yala-august-2025.pdf |
| 2025-09-15 | 2025 Yala | 2025-08-31 | leeks | 1333 | 1346 | yala-september-2025.pdf |
| 2025-10-15 | 2025 Yala | 2025-09-30 | leeks | 1333 | 1557 | yala-october-2025-yala-final.pdf |
| 2026-07-15 | 2026 Yala | 2026-06-30 | carrot | 2285 | 1195 | yala-july-2026.pdf |
| 2026-07-15 | 2026 Yala | 2026-06-30 | leeks | 1341 | 761 | yala-july-2026.pdf |

The view attaches the latest national snapshot whose report date and as-of date are both on or before that Thursday. A later publication does not appear on an earlier week. The week that contains 15 July 2026 sees the July 2026 leeks snapshot (1341 / 761). The week before it still sees the 15 October 2025 report.

Tomato has no stored extent. 383 extent cells are missing because the report printed a dash. Most other DOA pages were left out because the district numbers did not add up to the Sri Lanka row, or the scan had no readable table. Those pages were not guessed.

### Weather

`curated.weather_daily` has 12,933 rows from 1 January 2015 through 20 October 2026, from NASA POWER. Nine daily values are missing. A missing day is left out of the weekly mean and sum. It is not treated as zero.

Points are the supply towns, not the Colombo and Meegoda markets:

| Code | Place | District | Requested point |
| --- | --- | --- | --- |
| `nuwara_eliya` | Nuwara Eliya | Nuwara Eliya | 6.97, 80.77 |
| `badulla` | Badulla | Badulla | 6.99, 81.06 |
| `dambulla` | Dambulla | Matale | 7.87, 80.65 |

NASA snaps each request to its own grid cell. The snapped cell is in `curated.dim_weather_point`.

### Season calendar

`curated.season_calendar` is one row per day from 1 January 2015 through 31 December 2026 (4,383 days). Maha is October–March, labelled `2025/26 Maha`. Yala is April–September, labelled `2026 Yala`. These are the Department of Agriculture seasons, not a shorter trading window.

## Markets and crops

| Code | Name | Administrative district |
| --- | --- | --- |
| `colombo` | Colombo | Colombo |
| `meegoda` | Meegoda | Colombo |
| `nuwara_eliya` | Nuwara Eliya | Nuwara Eliya |
| `dambulla` | Dambulla | Matale |

`supply_rule` on `curated.dim_market` is `national_extent` for every market.

## Other tables

Query these only when the view is not enough.

| Table | Rows | Contents |
| --- | --- | --- |
| `dim_market` | 4 | Market code, name, district, supply rule |
| `dim_crop` | 3 | Crop code and name |
| `harti_price_weekly` | 13,104 | Workbook prices, one cell per year, week, price type, market, and crop |
| `cbsl_price_daily` | 1,544 | CBSL daily prices and the baseline printed beside them |
| `cultivation_extent` | 472 | District, subtotal, and Sri Lanka rows for the snapshots above |
| `season_calendar` | 4,383 | One season label per day |
| `dim_weather_point` | 3 | Supply points and the NASA grid cell actually used |
| `weather_daily` | 12,933 | Daily temperature, precipitation, and humidity |

## Rebuild locally

Python 3.14 or newer. From the project root, with the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m data_layer.build
.\.venv\Scripts\python.exe -m data_layer.weather
.\.venv\Scripts\python.exe -m data_layer.season
.\.venv\Scripts\python.exe -m data_layer.cbsl
.\.venv\Scripts\python.exe -m data_layer.extent
.\.venv\Scripts\python.exe -m data_layer.market_week
```

`build` reads `data_layer/source/Vegetable Prices.xlsx`. `weather` calls NASA POWER. `cbsl` reads `data_layer/source/cbsl_2026_long.csv`. `extent` loads the checked snapshots. `market_week` rebuilds the view in local DuckDB at `data_layer/warehouse/cobweb.duckdb`.

Publishing one piece, without rotating the reader password:

```powershell
.\.venv\Scripts\python.exe -m data_layer.publish weather
.\.venv\Scripts\python.exe -m data_layer.publish season
.\.venv\Scripts\python.exe -m data_layer.publish cbsl
.\.venv\Scripts\python.exe -m data_layer.publish extent
.\.venv\Scripts\python.exe -m data_layer.publish view
```

`python -m data_layer.publish` with no argument reloads the price tables and rotates `cobweb_reader`. Do not run that unless the reader password is meant to change. `data_layer/warehouse/`, `data_layer/export/`, and `data_layer/raw/` are local only and are not committed.
