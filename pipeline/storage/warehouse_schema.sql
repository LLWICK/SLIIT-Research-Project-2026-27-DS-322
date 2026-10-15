-- ==============================================================================
-- Member 1: Data Engineering & Geo-Spatial Decision Support System
-- Target Schema: PostgreSQL (with PostGIS compatibility) & SQLite/DuckDB
--
-- Preserves verified market identities, crop varieties, original reported units,
-- min/max/modal prices, and cryptographic provenance hashes (SHA-256).
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Dimension Tables
-- ------------------------------------------------------------------------------

-- Administrative District Dimension (25 Districts of Sri Lanka)
CREATE TABLE IF NOT EXISTS dim_district (
    district_id SERIAL PRIMARY KEY,
    district_name VARCHAR(50) NOT NULL UNIQUE,
    province VARCHAR(50) NOT NULL,
    latitude NUMERIC(8, 4) NOT NULL,
    longitude NUMERIC(8, 4) NOT NULL,
    elevation_m NUMERIC(6, 1),
    agro_ecological_zone VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Center Dimension (preserves market-level identities e.g. Pettah, Dambulla)
CREATE TABLE IF NOT EXISTS dim_market (
    market_id SERIAL PRIMARY KEY,
    market_name VARCHAR(100) NOT NULL UNIQUE,
    market_type VARCHAR(50) NOT NULL, -- Dedicated Economic Centre, Terminal Wholesale, Retail
    district_id INT REFERENCES dim_district(district_id),
    latitude NUMERIC(8, 4),
    longitude NUMERIC(8, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Crop Dimension (5 target research vegetables + varieties)
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

-- Vegetable Market Prices (CBSL and HARTI daily records)
CREATE TABLE IF NOT EXISTS fact_market_price (
    price_id BIGSERIAL PRIMARY KEY,
    market_id INT REFERENCES dim_market(market_id),
    district_id INT REFERENCES dim_district(district_id),
    crop_id INT NOT NULL REFERENCES dim_crop(crop_id),
    price_date DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,
    season_type VARCHAR(10) NOT NULL CHECK (season_type IN ('Yala', 'Maha')),
    season_year INT NOT NULL,
    price_type VARCHAR(20) NOT NULL CHECK (price_type IN ('wholesale', 'retail', 'producer')),
    price_lkr_per_kg NUMERIC(10, 2),
    min_price_lkr_per_kg NUMERIC(10, 2),
    max_price_lkr_per_kg NUMERIC(10, 2),
    modal_price_lkr_per_kg NUMERIC(10, 2),
    unit VARCHAR(20) DEFAULT 'Rs./kg',
    original_market_name VARCHAR(100),
    original_crop_name VARCHAR(100),
    source_agency VARCHAR(50) NOT NULL, -- 'CBSL', 'HARTI'
    source_url TEXT,
    provenance_sha256 VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_price_record UNIQUE (market_id, crop_id, price_date, price_type)
);

CREATE INDEX IF NOT EXISTS idx_price_crop_date ON fact_market_price (crop_id, price_date);
CREATE INDEX IF NOT EXISTS idx_price_market_type ON fact_market_price (market_id, price_type);
CREATE INDEX IF NOT EXISTS idx_price_season ON fact_market_price (crop_id, season_year, season_type);

-- Daily Weather Observations (NASA POWER MERRA-2 Agroclimatology)
CREATE TABLE IF NOT EXISTS fact_daily_weather (
    weather_id BIGSERIAL PRIMARY KEY,
    district_id INT NOT NULL REFERENCES dim_district(district_id),
    obs_date DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    season_type VARCHAR(10) NOT NULL CHECK (season_type IN ('Yala', 'Maha')),
    season_year INT NOT NULL,
    precipitation_mm NUMERIC(8, 2),
    temp_mean_c NUMERIC(5, 2),
    temp_max_c NUMERIC(5, 2),
    temp_min_c NUMERIC(5, 2),
    relative_humidity_pct NUMERIC(5, 2),
    wind_speed_ms NUMERIC(5, 2),
    source_provenance VARCHAR(50) DEFAULT 'NASA_POWER_MERRA2',
    raw_sha256 VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_weather_district_date UNIQUE (district_id, obs_date)
);

-- Seasonal Cultivation Targets & Commitments (DOA SEPC / Crop Forecast)
CREATE TABLE IF NOT EXISTS fact_cultivation_commitment (
    commitment_id BIGSERIAL PRIMARY KEY,
    district_id INT NOT NULL REFERENCES dim_district(district_id),
    crop_id INT NOT NULL REFERENCES dim_crop(crop_id),
    season_year INT NOT NULL,
    season_type VARCHAR(10) NOT NULL CHECK (season_type IN ('Yala', 'Maha')),
    target_hectares NUMERIC(10, 2),
    committed_hectares NUMERIC(10, 2),
    cultivation_intensity NUMERIC(6, 3),
    source_agency VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_commitment_district_crop_season UNIQUE (district_id, crop_id, season_year, season_type)
);

-- Official Harvest & Production (DCS / DOA final agricultural statistics)
CREATE TABLE IF NOT EXISTS fact_crop_production (
    production_id BIGSERIAL PRIMARY KEY,
    district_id INT NOT NULL REFERENCES dim_district(district_id),
    crop_id INT NOT NULL REFERENCES dim_crop(crop_id),
    season_year INT NOT NULL,
    season_type VARCHAR(10) NOT NULL CHECK (season_type IN ('Yala', 'Maha')),
    asv_extent_ha NUMERIC(10, 2),
    production_mt NUMERIC(12, 2),
    average_yield_kgha NUMERIC(10, 2),
    source_agency VARCHAR(50) NOT NULL,
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
    calculated_dri NUMERIC(5, 2) NOT NULL,
    audit_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
