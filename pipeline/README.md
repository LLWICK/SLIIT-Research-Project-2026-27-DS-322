# Member 1: Data Engineering & Geo-Spatial Decision Support Pipeline

**Project**: Data-Driven Decision Support System to Mitigate the Cobweb Effect in Sri Lankan Vegetable Markets  
**Component**: Member 1 — Data Engineering and Geo-Spatial Decision Support System  
**Branch**: `feature/real-data-pipeline-member01`

---

## 1. Architecture Overview

Member 1 delivers the clean data foundation and geo-spatial decision support platform for the research group. This pipeline replaces synthetic dummy data with automated, auditable, and scientifically sound ingestion from official sources:

```
[Official Public Data Sources]
        │
        ├── NASA POWER API (Daily Agroclimatology)
        ├── HARTI (Daily/Weekly Market Bulletins)
        ├── Central Bank of Sri Lanka (CBSL Daily Price Reports)
        ├── Department of Agriculture (DOA SEPC Statistics)
        └── Department of Census and Statistics (DCS Highland Crops)
        │
        ▼
[Extraction Layer: pipeline/extractors/]
  - Resilient HTTP Sessions with exponential backoff
  - Unmodified Raw Payload Storage: data/raw/
  - Cryptographic Provenance Manifests (SHA-256, extraction timestamp, request URL)
        │
        ▼
[Transformation Layer: pipeline/transformers/]
  - ISO Date formatting (YYYY-MM-DD)
  - Missing Value Preservation (-999.0 strictly mapped to NaN, never zero)
  - Parameter snake_case normalization & unit alignment
  - Cultivation Season Assignment (Yala / Maha)
        │
        ▼
[Validation Layer: pipeline/validation/]
  - Schema compliance & date continuity
  - Domain physical meteorological boundary verification
  - Completeness & anomaly ratio auditing
        │
        ▼
[Storage & Staging Layer: pipeline/storage/]
  - Processed clean files: data/processed/
  - Shared research group exports: data/exports/ (for Member 3 MSRS/DCVS & Member 4 ABM)
  - Relational warehouse staging (PostgreSQL / DuckDB schema)
```

---

## 2. Directory Layout

```text
SLIIT-Research-Project-2026-27-DS-322/
├── pipeline/
│   ├── __init__.py
│   ├── config.py                 # Target crops, 25 district coordinates, NASA parameters
│   ├── cli.py                    # Unified command-line runner
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseExtractor with retries, logging, manifest generation
│   │   └── weather_nasa.py       # NasaPowerWeatherExtractor
│   ├── transformers/
│   │   ├── __init__.py
│   │   └── weather_cleaner.py    # Standardizer, seasonal mapper, NaN preserver
│   ├── validation/
│   │   ├── __init__.py
│   │   └── validator.py          # Schema, duplicate, and meteorological bounds checks
│   └── storage/
│       ├── __init__.py
│       └── warehouse_schema.sql  # Production PostgreSQL DDL with PostGIS compatibility
├── data/
│   ├── v1/                       # Pre-existing prototype synthetic data (untouched)
│   ├── raw/
│   │   └── weather/              # Raw JSON responses and JSON provenance manifests
│   ├── processed/
│   │   └── weather/              # Cleaned standardized CSV / Parquet datasets
│   └── exports/                  # Cross-member consolidated datasets
├── docs/
│   └── source_availability_matrix.md # Detailed audit of 5 official data sources
└── tests/
    ├── test_weather_cleaner.py   # Tests for date parsing, fill values, seasonal rules
    └── test_validator.py         # Tests for bounds, duplicates, and temperature inversion
```

---

## 3. How to Run the Pipeline

### Run Unit Tests
Run the automated test suite using Python's built-in `unittest`:
```powershell
python -m unittest discover tests
```

### Run NASA POWER Weather Extraction
Extract, clean, validate, and export weather observations for priority vegetable districts (e.g., Nuwara Eliya and Badulla):
```powershell
python -m pipeline.cli weather --districts "Nuwara Eliya,Badulla" --start 2024-01-01 --end 2024-01-10
```

Extract all 25 Sri Lankan administrative districts:
```powershell
python -m pipeline.cli weather --districts all --start 2024-01-01 --end 2024-01-31
```

---

## 4. Scientific Data Integrity Rules

1. **Preservation of Missing Observations**:
   NASA POWER flags missing or out-of-bounds readings with fill code `-999.0`. This pipeline strictly maps `-999.0` to `np.nan` (`null`), preventing catastrophic statistical distortion in mean temperature, cumulative rainfall, and crop viability indices. Missing values are **never** imputed to `0.0`.
2. **Raw Data Immutability**:
   Original API responses are stored unmodified in `data/raw/weather/` accompanied by SHA-256 hashes and timestamped JSON manifests.
3. **Market-Level Distinction**:
   Wholesale market prices (e.g., Dambulla DEC, Pettah) are tagged to specific market dimensions (`dim_market`) and not falsely generalized as district-wide farmgate prices.
4. **Cultivation Season Assignment**:
   Calendar dates are mapped according to Sri Lankan agro-climatic norms:
   - **Yala**: April 1 – September 30 (assigned to calendar year).
   - **Maha**: October 1 – March 31 (assigned to the starting October year).
