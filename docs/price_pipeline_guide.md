# Member 1: Vegetable Price ETL Pipeline Guide

**Research Project:** Data-Driven Decision Support System to Mitigate the Cobweb Effect in Sri Lankan Vegetable Markets
**Component:** Member 1 — Data Engineering and Geo-Spatial Decision Support System
**Git Branch:** `feature/real-price-data-pipeline-member01`

---

## 1. Overview & Research Scope

The vegetable-price ETL pipeline automates the ingestion, transformation, validation, and storage of genuine wholesale and retail vegetable price records from official Sri Lankan statutory authorities.

This pipeline forms the factual economic observation layer for:
- **Member 3 (Spatial Crowding & Advisory DSS):** Dynamic Crop Valuation Scores (DCVS) and Market Sentiment Recommendation System (MSRS).
- **Member 4 (Agent-Based Simulation):** Farmer planting decisions and price expectation dynamics under the Cobweb Effect.

### Target Research Vegetables
1. **Carrot** (*Daucus carota*)
2. **Cabbage** (*Brassica oleracea*)
3. **Leek** (*Allium ampeloprasum*)
4. **Big onion** (*Allium cepa*) — both Local and Imported varieties
5. **Tomato** (*Solanum lycopersicum*)

---

## 2. Official Data Sources & Verified Findings

### A. Central Bank of Sri Lanka (CBSL) — Daily Price Reports
- **Portal:** [https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report](https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report)
- **URL Pattern:** `https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/statistics/pricerpt/price_report_{YYYYMMDD}_e.pdf`
- **Format:** 2-page PDF document. Page 2 contains a 30-commodity tabular price matrix.
- **Decompression:** Streams use standard `FlateDecode`. Parsed directly using Python standard library `zlib` and regex tokenization (zero third-party PDF dependencies).
- **Markets Covered:** Pettah (Wholesale, Retail), Dambulla DEC (Wholesale, Retail), Narahenpita DEC (Retail).
- **Reported Metrics:** Previous day and current day point prices in Rs./kg. Missing observations are published as `'n.a.'` and preserved as null (`NaN`).
- **Critical Limitation Confirmed:** **Leek is NOT published in CBSL Daily Price Reports**. CBSL omits Leek from this bulletin. The pipeline explicitly verifies and logs this absence, never fabricating or substituting synthetic Leek values.

### B. Hector Kobbekaduwa Agrarian Research & Training Institute (HARTI)
- **Portal:** [https://www.harti.gov.lk/daily-price.php](https://www.harti.gov.lk/daily-price.php)
- **Format:** 11-page Daily Food Price Bulletins covering terminal and regional dedicated economic centres.
- **Markets Covered:** Pettah, Dambulla, Kandy, Keppetipola, Norochcholai, Meegoda, Nuwara Eliya, etc.
- **Reported Metrics:** Price distributions including **Minimum**, **Maximum**, and **Modal** prices for wholesale and retail markets.
- **Target Crop Coverage:** All 5 research crops (Carrot, Cabbage, Leek, Big onion, Tomato) are covered.
- **Source Limitations Confirmed:**
  1. *Automated Crawler Fragility:* Many recent bulletin links (e.g. October 2026) on `harti.gov.lk` return `HTTP 404` or point to non-PDF filenames.
  2. *Font Subset Encoding:* HARTI PDF streams use embedded subset glyph IDs (`BCDGEE+Iskoola`, `BCDFEE+NirmalaUI-Bold`) with sparse `/ToUnicode` CMaps, causing direct text extraction to yield glyph indices.
  3. *Tested Ingestion Fallback:* The pipeline implements an explicit offline fallback (`ingest_offline_bulletin` and `ingest_tabular_records`) that computes cryptographic SHA-256 hashes and captures full provenance manifests for verified official bulletins.

---

## 3. Architecture & Data Flow

```
┌────────────────────────┐      ┌────────────────────────┐
│   CBSL Daily Reports   │      │  HARTI Daily Bulletins │
│      (PDF Stream)      │      │ (PDF / Offline Ingest) │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            ▼                               ▼
  [price_cbsl.py]                 [price_harti.py]
  - zlib decompression            - Portal scraper
  - Kerning token parsing         - Offline bulletin ingestion
  - SHA-256 provenance            - SHA-256 provenance
            │                               │
            └───────────────┬───────────────┘
                            ▼
              data/raw/prices/<source>/
              ├── <document>.pdf
              └── <document>.manifest.json
                            │
                            ▼
                  [price_cleaner.py]
                  - Crop name standardization
                  - Agricultural season assignment (Maha/Yala)
                  - Market coordinates & district enrichment
                  - Preservation of original labels and nulls
                            │
                            ▼
                 [price_validator.py]
                 - Mandatory schema adherence
                 - Domain bounds: 5.0 to 5000.0 Rs./kg
                 - Retail >= Wholesale margin consistency
                 - Duplicate observation checks
                            │
                            ▼
             data/processed/prices/<filename>.csv
                            │
                            ▼
             [warehouse_schema.sql]
             Star-schema: fact_market_price, dim_market, dim_crop
```

---

## 4. Directory Structure

```
├── .gitignore                          # Root ignore rules for raw data, processed CSVs, and caches
├── pipeline/
│   ├── __init__.py
│   ├── config.py                       # Crop aliases, market coordinates, URL templates
│   ├── cli.py                          # Unified CLI entrypoint
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseExtractor with resilient session & SHA-256 hashing
│   │   ├── price_cbsl.py               # CBSL PDF decompression & token parser
│   │   └── price_harti.py              # HARTI bulletin scraper & offline fallback ingester
│   ├── quality/
│   │   ├── __init__.py
│   │   └── dri_calculator.py           # Data Reliability Index (Completeness, Recency, Consistency)
│   ├── transformers/
│   │   ├── __init__.py
│   │   └── price_cleaner.py            # Crop normalizer, season assigner, district enricher
│   ├── validation/
│   │   ├── __init__.py
│   │   └── price_validator.py          # Schema, economic bounds, and margin sanity checks
│   └── storage/
│       └── warehouse_schema.sql        # Star schema DDL for PostgreSQL / SQLite
├── tests/
│   ├── fixtures/
│   │   └── sample_cbsl_report.pdf      # Captured 215 KB official CBSL report for offline tests
│   ├── test_cbsl_price_extractor.py    # Offline CBSL extraction & value verification
│   ├── test_dri_calculator.py          # Offline DRI calculation & tier scoring
│   ├── test_harti_price_extractor.py   # Offline HARTI ingestion & provenance verification
│   ├── test_price_cleaner.py           # Normalization, season assignment & null preservation
│   └── test_price_validator.py         # Bounds, duplicates, and anomaly checks
└── docs/
    └── price_pipeline_guide.md         # This technical documentation
```

---

## 5. CLI Usage & Pipeline Execution

### A. Extract CBSL Price Reports

#### Live Download & Processing:
```bash
python -m pipeline.cli price --source cbsl --date 2026-10-14
```

#### Offline Ingestion from Local File:
```bash
python -m pipeline.cli price --source cbsl --file tests/fixtures/sample_cbsl_report.pdf
```

### B. Ingest HARTI Price Bulletins

#### Discover Available Remote Links on HARTI Portal:
```bash
python -m pipeline.cli price --source harti --discover
```

#### Ingest Offline Bulletin File (with SHA-256 Provenance):
```bash
python -m pipeline.cli price --source harti --file path/to/daily_31-12-2024.pdf
```

#### Attempt Live Fetch (Subject to portal link availability):
```bash
python -m pipeline.cli price --source harti --date 2024-12-31
```

---

## 6. Processed Output Schema

Saved under `data/processed/prices/prices_{YYYYMMDD}_{agency}.csv`:

| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `obs_date` | Date (`YYYY-MM-DD`) | `2026-10-14` | Observation calendar date |
| `year` | Integer | `2026` | Calendar year |
| `month` | Integer | `10` | Calendar month (1-12) |
| `day` | Integer | `14` | Day of month |
| `season_type` | String | `Maha` | Agricultural cultivation season (`Maha` or `Yala`) |
| `season_year` | Integer | `2026` | Cultivation cycle year |
| `crop_name` | String | `Carrot` | Standardized research target crop name |
| `source_commodity_name` | String | `Carrot` | Unaltered original name from official source |
| `market_name` | String | `Pettah` | Standardized market center identity |
| `original_market_name` | String | `Pettah` | Unaltered original market label |
| `district` | String | `Colombo` | Administrative district of the market |
| `latitude` | Float | `6.9365` | Market centroid latitude |
| `longitude` | Float | `79.8528` | Market centroid longitude |
| `price_type` | String | `wholesale` | Price category (`wholesale` or `retail`) |
| `price_lkr_per_kg` | Float / Null | `220.0` | Primary price in LKR/kg (never 0 for missing) |
| `min_price_lkr_per_kg` | Float / Null | `200.0` | Minimum reported price (when present in HARTI) |
| `max_price_lkr_per_kg` | Float / Null | `240.0` | Maximum reported price (when present in HARTI) |
| `modal_price_lkr_per_kg`| Float / Null | `220.0` | Modal reported price |
| `unit` | String | `Rs./kg` | Original unit reported by source authority |
| `source_agency` | String | `CBSL` | Statutory authority (`CBSL` or `HARTI`) |
| `source_url` | String | `https://...` | Provenance source URL or local offline URI |
| `provenance_sha256` | String | `5b61...` | 64-character SHA-256 hash of raw input document |

---

## 7. Running Tests (100% Offline)

All tests execute entirely offline using captured fixture data:

```bash
python -m unittest discover tests
```

Expected output:
```text
Ran 40 tests in ~2.5s
OK
```

Test coverage includes:
- `tests/test_cbsl_price_extractor.py`: PDF token decompression, table parsing, market mappings, Leek absence.
- `tests/test_harti_price_extractor.py`: Scraper URL discovery, modern/archive date patterns, offline fallback ingestion.
- `tests/test_price_cleaner.py`: Crop aliases, district enrichment, coordinate lookups, missing-value null preservation.
- `tests/test_price_validator.py`: Schema enforcement, economic bounds (5-5000 Rs./kg), duplicate checks on variety identities.
- `tests/test_dri_calculator.py`: Completeness, recency decay, consistency, small-sample penalty (0.7x), ledger append/preservation.
- `tests/test_price_pipeline_e2e.py`: End-to-end execution of CBSL sample PDF through extraction, cleaning, validation, DRI scoring, and CLI execution.

---

## 8. Data Reliability Index (DRI) Methodology

The Data Reliability Index (`pipeline/quality/dri_calculator.py`) evaluates the fitness-for-use of incoming observational datasets before feeding downstream decision models (Member 3 MSRS/DCVS and Member 4 ABM).

### Composite Formula
$$\text{DRI} = 100 \times \left( w_c \cdot S_{\text{comp}} + w_t \cdot S_{\text{time}} + w_k \cdot S_{\text{cons}} \right)$$

### Pillar Normalizations
1. **Completeness ($S_{\text{comp}} \in [0.0, 1.0]$)**:
   $$S_{\text{comp}} = \frac{N_{\text{valid}}}{N_{\text{total}}}$$
   Proportion of observations with valid, non-null, strictly positive values. Missing prices (such as `'n.a.'` in source reports) are preserved as null and reduce completeness accordingly.
2. **Timeliness / Recency ($S_{\text{time}} \in [0.0, 1.0]$)**:
   Linear decay over acceptable window (default: 30 days):
   $$S_{\text{time}} = \max\left(0.0, 1.0 - \frac{\Delta t}{\text{max\_delay\_days}}\right)$$
   Where $\Delta t = \max(0, (\text{eval\_date} - \text{max\_obs\_date}).\text{days})$.
   If $\Delta t \le 1$, $S_{\text{time}} = 1.0$. If $\Delta t \ge 30$, $S_{\text{time}} = 0.0$.
3. **Consistency ($S_{\text{cons}} \in [0.0, 1.0]$)**:
   Penalizes out-of-bounds prices ($< 5$ or $> 5000$ Rs./kg), inverted bounds ($\text{min} > \text{max}$), inverted margins ($\text{retail} < \text{wholesale}$), and duplicate observations.

### Configurable Weights
Parameters are defined in `DRIConfig` and strictly enforced:
- **Completeness Weight ($w_c$)**: `0.40`
- **Timeliness Weight ($w_t$)**: `0.30`
- **Consistency Weight ($w_k$)**: `0.30`
- Constraint: $w_c + w_t + w_k = 1.0$ (raises `ValueError` if sum $\ne 1.0$).

### Small-Sample Penalty (0.7x Multiplier)
Statistical estimators derived from sparse observations exhibit high sampling variance. To prevent small samples from generating falsely confident reliability scores:
- If observation count $N < 5$ (`min_observations_required`), an exact **0.7 multiplier** is applied:
  $$\text{DRI}_{\text{penalized}} = \text{round}(\text{DRI}_{\text{raw}} \times 0.7, 2)$$
- An audit note is automatically recorded:
  `"Small sample penalty applied: {N} rows < 5 required."`

### Classification Tiers
- **HIGH ($\text{DRI} \ge 80.0$)**: Ready for production automated advisory modeling.
- **MODERATE ($60.0 \le \text{DRI} < 80.0$)**: Usable with confidence alerts / fallback checks.
- **LOW ($\text{DRI} < 60.0$)**: High risk; triggers manual verification or fallback ingestion.
- **INSUFFICIENT_DATA ($\text{DRI} = 0.0$)**: Empty or zero-observation dataset.

### Validation Guard: No Silent Overrides for Invalid Data
If a dataset fails structural or domain validation (`is_valid is False`):
- Validation status is recorded as `"FAILED"`.
- The dataset is **never silently treated as reliable**.
- If the unconstrained score falls in the HIGH tier, it is automatically downgraded to **MODERATE** (or LOW).
- An explicit error audit note is logged: `"Validation status is FAILED with {n} error(s)."`.

---

## 9. Local Audit Ledger vs. Relational Data Warehouse

Understanding the architectural distinction between local-file persistence and the relational warehouse is essential:

```
┌─────────────────────────────────┐
│   Price CLI / ETL Execution     │
└───────────────┬─────────────────┘
                │
        ┌───────┴─────────────────────────┐
        ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Local File JSONL Ledger       │ │ Relational Data Warehouse     │
│ (Implemented for Member 1)    │ │ (Schema DDL Ready)            │
├───────────────────────────────┤ ├───────────────────────────────┤
│ Path: data/processed/prices/  │ │ Target: PostgreSQL / Timescale│
│       dri_audit_ledger.jsonl  │ │ Table: fact_data_reliability_ │
│ Format: JSON Lines (1 rec/ln) │ │        audit                  │
│ Purpose: Immutable append-only│ │ Purpose: Multi-tenant DSS     │
│ log for offline ETL auditing  │ │ analytical queries & BI joins │
│ Dependencies: Zero (pure std) │ │ Dependencies: DB connection,  │
│ Safe: Never overwrites        │ │ credentials, server process   │
└───────────────────────────────┘ └───────────────────────────────┘
```

1. **Local JSON Lines Audit Ledger (`data/processed/prices/dri_audit_ledger.jsonl`)**:
   - Every execution of the price pipeline appends a single complete JSON audit record to this file.
   - Preserves all historical records (append mode `"a"`).
   - Requires zero external database servers, connection pools, or credentials.
   - Safely creates parent directories on demand and logs write failures transparently.
   - Enables offline reproduction of quality metrics for any historical ingestion.

2. **Relational Warehouse Schema (`pipeline/storage/warehouse_schema.sql`)**:
   - The production star-schema defines the `fact_data_reliability_audit` table with foreign keys to `dim_district` and `dim_crop`.
   - Populating this table is an enterprise batch-load responsibility once the centralized PostgreSQL/TimescaleDB cluster is provisioned.
   - Member 1's local ETL pipeline writes to the JSONL ledger first to guarantee zero dependency on external server availability.

---

## 10. Troubleshooting & Common Pitfalls

1. **CBSL HTTP 404 Error:**
   - CBSL publishes daily price reports strictly on Central Bank business days. No reports are issued on weekends or Sri Lankan mercantile holidays.
2. **Missing Leek in CBSL Reports:**
   - This is intentional and expected. Do not fabricate Leek prices from CBSL; use HARTI bulletins for Leek observations.
3. **HARTI Portal 404 Links & Filename Variations:**
   - Recent links on `harti.gov.lk` frequently use modern names like `Vegetable Pricenew ex1(2026.10.07).pdf`. The pipeline supports both modern and historical naming conventions.
4. **Audit Ledger Location:**
   - The audit ledger is located at `data/processed/prices/dri_audit_ledger.jsonl`. It is automatically appended to on each CLI price run.
5. **Git Safety Check:**
   - Never run `git add .` or stage `data/raw/` or `data/processed/` files. The root `.gitignore` automatically excludes them.