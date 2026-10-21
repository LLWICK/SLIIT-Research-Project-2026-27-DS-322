# Cultivation progress source — not found

Methodology.docx section 3.9 requires district-month cultivation progress = achieved hectares / target hectares.
Looked for data/member1/cultivation_monthly.csv, then processed/cultivation_monthly.csv. The data disk is not scanned.
That district-month contract file is not on disk.

What else was checked:
- `processed/dcs_highland_seasonal.csv` is Yala/Maha extent and production, not monthly target and achieved.
- `processed/doa_cultivation_progress.csv` is read only when it is at that known path. It is not renamed into cultivation_monthly.csv.

processed/doa_cultivation_progress.csv has 172 rows.
Columns: available_date, observation_date, season, crop, district, target_ha, achieved_ha, source_report, source_page, scope_level, notes.
target_ha and achieved_ha are in the file, so these are target and achieved hectares rather than a renamed unrelated table.
year and month are not columns. The dates in the file are available_date and observation_date.
72 rows are national totals. 100 rows are district rows.
District observation dates: 2018-06-30, 2018-07-31.
Tomato has no district rows.
70 district rows have a target of 0 hectares. 30 district rows have a target above 0.
The loader contract is year, month, crop, district, target_ha, and achieved_ha, joined to each market's supply districts. This file was not copied to cultivation_monthly.csv.
District rows stay in the audit. The scored cultivation signal is national, matches the official source season, and uses a report only when available_date is on or before the forecast Monday.

No hectares were invented. The district-month column on the main training panel stays missing until a file matches that contract.
National exact-season scoring is written under outputs/experiment_c.
