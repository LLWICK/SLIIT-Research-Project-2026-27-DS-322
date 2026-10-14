-- ==============================================================================
-- Member 1: Data Engineering & Geo-Spatial Decision Support System
-- Target Schema: PostgreSQL (with optional PostGIS spatial extension)
-- 
-- Compatible with local PostgreSQL 14+, DuckDB, and SQLite (via standard types).
-- Coordinates and geometry columns support spatial joins and choropleth rendering.
-- ==============================================================================

-- Optional PostGIS extension (if enabled in PostgreSQL)
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- ------------------------------------------------------------------------------
-- 1. Dimension Tables
-- ------------------------------------------------------------------------------

-- Administrative District Dimension
CREATE TABLE IF NOT EXISTS dim_district (
    district_id SERIAL PRIMARY KEY,
    district_name VARCHAR(50) NOT NULL UNIQUE,
    province VARCHAR(50) NOT NULL,
    latitude NUMERIC(8, 4) NOT NULL,
    longitude NUMERIC(8, 4) NOT NULL,
    elevation_m NUMERIC(6, 1),
    agro_ecological_zone VARCHAR(50), -- Up Country Wet Zone, Dry Zone, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Center Dimension (preserves market-level identities e.g. Pettah, Dambulla)
CREATE TABLE IF NOT EXISTS dim_market (
    market_id SERIAL PRIMARY KEY,
    market_name VARCHAR(100) NOT NULL UNIQUE,
    market_type VARCHAR(50) NOT NULL, -- Dedicated Economic Centre, Wholesale, Retail
    district_id INT NOT NULL REFERENCES dim_district(district_id),
    latitude NUMERIC(8, 4),
    longitude NUMERIC(8, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Crop Dimension (5 target vegetables + varieties)
CREATE TABLE IF NOT EXISTS dim_crop (
    crop_id SERIAL PRIMARY KEY,
    crop_name VARCHAR(50) NOT NULL UNIQUE,
    botanical_name VARCHAR(100),
    crop_category VARCHAR(50) NOT NULL, -- Up-country vegetable, Low-country, OFC
    standard_unit VARCHAR(20) DEFAULT 'kg',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------------------------
-- 2. Fact Tables
-- ------------------------------------------------------------------------------

-- Daily Weather Observations (NASA POWER MERRA-2 Agroclimatology)
CREATE TABLE IF NOT EXISTS fact_daily_weather (
    weather_id BIGSERIAL PRIMARY KEY,
    district_id INT NOT NULL REFERENCES dim_district(district_id),
    obs_date DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    season_type VARCHAR(10) NOT NULL CHECK (season_type IN ('Yala', 'Maha')),
    season_year INT NOT NULL,
    precipitation_mm NUMERIC(8, 2), -- PRECTOTCORR: Corrected precipitation (mm/day)
    temp_mean_c NUMERIC(5, 2),      -- T2M: Mean temperature at 2m (°C)
    temp_max_c NUMERIC(5, 2),       -- T2M_MAX: Max temperature at 2m (°C)
    temp_min_c NUMERIC(5, 2),       -- T2M_MIN: Min temperature at 2m (°C)
    relative_humidity_pct NUMERIC(5, 2), -- RH2M: Relative humidity at 2m (%)
    wind_speed_ms NUMERIC(5, 2),    -- WS2M: Wind speed at 2m (m/s)
    source_provenance VARCHAR(50) DEFAULT 'NASA_POWER_MERRA2',
    raw_sha256 VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_weather_district_date UNIQUE (district_id, obs_date)
);

CREATE INDEX IF NOT EXISTS idx_weather_date ON fact_daily_weather (obs_date);
CREATE INDEX IF NOT EXISTS idx_weather_district_season ON fact_daily_weather (district_id, season_year, season_type);

-- Vegetable Market Prices (HARTI / CBSL daily & weekly records)
CREATE TABLE IF NOT EXISTS fact_market_price (
    price_id BIGSERIAL PRIMARY KEY,
    market_id INT REFERENCES dim_market(market_id),
    district_id INT REFERENCES dim_district(district_id),
    crop_id INT NOT NULL REFERENCES dim_crop(crop_id),
    price_date DATE NOT NULL,
    price_type VARCHAR(20) NOT NULL CHECK (price_type IN ('wholesale', 'retail', 'producer')),
    granularity VARCHAR(20) NOT NULL, -- 'daily', 'weekly_average'
    price_lkr_per_kg NUMERIC(10, 2) NOT NULL,
    source_agency VARCHAR(50) NOT NULL, -- 'HARTI', 'CBSL'
    raw_sha256 VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_price_record UNIQUE (market_id, district_id, crop_id, price_date, price_type)
);

CREATE INDEX IF NOT EXISTS idx_price_crop_date ON fact_market_price (crop_id, price_date);
CREATE INDEX IF NOT EXISTS idx_price_market ON fact_market_price (market_id, price_type);

-- Seasonal Cultivation & Targets (DOA SEPC / Crop Forecast / Agrarian Development)
CREATE TABLE IF NOT EXISTS fact_cultivation_commitment (
    commitment_id BIGSERIAL PRIMARY KEY,
    district_id INT NOT NULL REFERENCES dim_district(district_id),
    crop_id INT NOT NULL REFERENCES dim_crop(crop_id),
    season_year INT NOT NULL,
    season_type VARCHAR(10) NOT NULL CHECK (season_type IN ('Yala', 'Maha')),
    target_hectares NUMERIC(10, 2),
    committed_hectares NUMERIC(10, 2),
    cultivation_intensity NUMERIC(6, 3), -- committed / target
    source_agency VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_commitment_district_crop_season UNIQUE (district_id, crop_id, season_year, season_type)
);

-- Official Harvest & Production (DCS / DOA final statistics)
CREATE TABLE IF NOT EXISTS fact_crop_production (
    production_id BIGSERIAL PRIMARY KEY,
    district_id INT NOT NULL REFERENCES dim_district(district_id),
    crop_id INT NOT NULL REFERENCES dim_crop(crop_id),
    season_year INT NOT NULL,
    season_type VARCHAR(10) NOT NULL CHECK (season_type IN ('Yala', 'Maha')),
    asv_extent_ha NUMERIC(10, 2),      -- Extent harvested (ha)
    production_mt NUMERIC(12, 2),       -- Production (Metric Tons)
    average_yield_kgha NUMERIC(10, 2),  -- Yield (kg/ha)
    source_agency VARCHAR(50) NOT NULL, -- 'DCS', 'DOA'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_production_district_crop_season UNIQUE (district_id, crop_id, season_year, season_type)
);

-- ------------------------------------------------------------------------------
-- 3. Data Reliability Index (DRI) Audit Ledger
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_data_reliability_audit (
    audit_id BIGSERIAL PRIMARY KEY,
    district_id INT NOT NULL REFERENCES dim_district(district_id),
    crop_id INT NOT NULL REFERENCES dim_crop(crop_id),
    season_year INT NOT NULL,
    season_type VARCHAR(10) NOT NULL,
    price_completeness_ratio NUMERIC(5, 4),
    price_source_reliability NUMERIC(5, 4),
    weather_completeness_ratio NUMERIC(5, 4),
    weather_source_reliability NUMERIC(5, 4),
    commitment_completeness_ratio NUMERIC(5, 4),
    commitment_source_reliability NUMERIC(5, 4),
    calculated_dri NUMERIC(5, 2) NOT NULL, -- Composite 0-100 DRI score
    audit_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
