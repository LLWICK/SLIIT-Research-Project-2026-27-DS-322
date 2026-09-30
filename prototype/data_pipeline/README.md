# Member 1 data pipeline

Ingesters and audits for the Cobweb HTFE project. **Code lives in git; data lives outside the repo.**

Default data root (override with `COBWEB_DATA_ROOT`):

```
D:\SLIIT\4th Year\Reaserch\data sets\
  Vegetable Prices.xlsx
  raw\
  processed\
```

Run every command from the `prototype` folder so `python -m data_pipeline...` resolves:

```bash
cd prototype
```

## HARTI Excel prices

Week rule (do not change): **W1 = first Monday on or after 1 January** (verified ~97.3% against the published monthly sheet).

```bash
python -m data_pipeline.ingest.harti_excel_prices
python -m data_pipeline.ingest.harti_excel_prices --xlsx "D:\path\to\Vegetable Prices.xlsx"
python -m data_pipeline.audit.profile_harti_excel
```

Writes: `processed/harti_prices_weekly_long.csv`, `harti_prices_monthly_long.csv`, `harti_coverage_by_series.csv`, audit profile under `processed/audit/`.

## Open-Meteo weather

Fetches daily ERA5/ERA5-Land JSON, then builds daily / **HARTI-week** / ISO-week / monthly tables. Re-aggregate without re-download:

```bash
python -m data_pipeline.ingest.weather_open_meteo --build-only
python -m data_pipeline.ingest.weather_open_meteo --build-only --include-extra
python -m data_pipeline.ingest.weather_open_meteo                  # fetch missing + build
python -m data_pipeline.ingest.weather_open_meteo --refresh        # re-fetch stale
```

Attribute: *Weather data by Open-Meteo.com* (CC BY 4.0).

## NASA POWER weather (fallback)

```bash
python -m data_pipeline.ingest.weather_nasa_power --build-only
python -m data_pipeline.ingest.weather_nasa_power
```

Daily CSV only; for weekly joins use `weather_common.aggregate_periods(..., "harti_week")`.

## DCS agriculture

Highland extent/production (district × Yala/Maha), retail dashboard, abstracts, etc.:

```bash
python -m data_pipeline.ingest.dcs_agriculture
python -m data_pipeline.ingest.dcs_agriculture --sections highland_timeseries retail_dashboard
python -m data_pipeline.ingest.dcs_agriculture --crops carrot leeks tomato --bulletins recent
```

Output: `raw/dcs/` + `raw/dcs/manifest.csv`.

## CBSL Daily Price Report

Listing crawl (IPv4 forced by default), resumable PDF download, status. Parser module not shipped yet.

```bash
python -m data_pipeline.ingest.cbsl_price_report crawl --wayback
python -m data_pipeline.ingest.cbsl_price_report download --start 2024 --weekly
python -m data_pipeline.ingest.cbsl_price_report download --start 2019-01 --end 2019-03
python -m data_pipeline.ingest.cbsl_price_report status
```

Manifest: `raw/cbsl_price_report/manifest.csv`. PDFs: `raw/cbsl_price_report/pdf/<year>/`.

## Analysis-ready panel

Wholesale weekly panel for carrot, leeks, and tomato. Run from `prototype`:

```bash
python -m data_pipeline.build.parse_dcs_highland
python -m data_pipeline.build.build_analysis_ready_panel
python -m data_pipeline.audit.panel_audit
```

Writes `processed/latest/analysis_ready_panel.parquet` and `processed/audit/panel_audit.md`.

Member 2 training (from `prototype/member2_ForecastingEngine`):

```bash
python -m training.run_pp1
```

## Dependencies

Typical: `pandas`, `pyyaml`, `pyarrow`, `pandera`, `holidays`, `requests`, `beautifulsoup4`, `lxml`, `openpyxl`, `pymupdf`. Training also needs `lightgbm`, `xgboost`, `statsmodels`, `matplotlib`.
