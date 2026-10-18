"""
Weekly Aligner: The main orchestrator for the weekly temporal alignment pipeline.

Harmonizes heterogeneous daily market prices and daily weather observations into
standardized Monday-to-Sunday ISO weekly features, enriches them with seasonal DOA
and annual DCS macro-context without data fabrication, and provides dual-layer outputs
(granular market-level series and prototype-compatible district matrices).
"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np

from pipeline.alignment.schemas import (
    CLEAN_PRICE_REQUIRED_COLUMNS,
    CLEAN_WEATHER_REQUIRED_COLUMNS,
    CLEAN_DOA_REQUIRED_COLUMNS,
    CLEAN_DCS_REQUIRED_COLUMNS,
    validate_dataframe,
)
from pipeline.alignment.calendar_utils import get_agricultural_season
from pipeline.alignment.weather_aggregator import WeatherAggregator
from pipeline.alignment.price_aggregator import PriceAggregator
from pipeline.alignment.crop_vocab import CropVocabularyEnforcer, CANONICAL_TARGET_CROPS
from pipeline.alignment.adapters import (
    export_to_prototype_v1_format,
    compute_weekly_dri_status,
)


class WeeklyAligner:
    """
    Orchestrates end-to-end temporal alignment of agricultural market, weather,
    and administrative datasets on a standardized Monday-to-Sunday weekly grid.
    """

    def __init__(
        self,
        decimals: int = 2,
        quarantine_sink_path: Optional[str] = None,
    ):
        """
        Initialize the weekly aligner.

        Args:
            decimals: Number of decimal places to round computed statistics to.
            quarantine_sink_path: Optional file path to append quarantined ambiguous records.
        """
        self.decimals = decimals
        self.quarantine_sink_path = quarantine_sink_path
        self.weather_aggregator = WeatherAggregator(decimals=decimals)
        self.price_aggregator = PriceAggregator(
            decimals=decimals,
            quarantine_sink_path=quarantine_sink_path,
        )

    def align_price_and_weather(
        self,
        daily_price_df: pd.DataFrame,
        daily_weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align daily price observations with daily weather observations into a weekly
        market-level panel dataset.

        Steps:
        1. Aggregate daily weather by district and ISO week (sum rainfall, average temp/humidity).
        2. Aggregate daily prices by market, crop, and ISO week (filter 5 target crops, isolate Big Onion origin).
        3. Merge on (district, iso_year, iso_week, week_start_date, week_end_date).
        4. Attach Sri Lankan agricultural season markers (Maha / Yala) based on week_start_date.

        Args:
            daily_price_df: Daily cleaned price observations.
            daily_weather_df: Daily cleaned agro-meteorological observations.

        Returns:
            Granular weekly market-weather DataFrame.
        """
        # Step 1: Aggregate high-frequency streams
        weekly_weather = self.weather_aggregator.aggregate(daily_weather_df)
        weekly_price = self.price_aggregator.aggregate(daily_price_df)

        if weekly_price.empty:
            return pd.DataFrame()

        join_keys = [
            "district",
            "iso_year",
            "iso_week",
            "week_start_date",
            "week_end_date",
        ]

        # Step 2: Left-join district weather to market-level prices
        if not weekly_weather.empty:
            merged_df = pd.merge(
                weekly_price,
                weekly_weather,
                on=join_keys,
                how="left",
                suffixes=("", "_weather"),
            )
        else:
            merged_df = weekly_price.copy()

        # Step 3: Attach agricultural season metadata
        seasons = []
        season_years = []
        season_labels = []

        for w_start in merged_df["week_start_date"]:
            s_meta = get_agricultural_season(w_start)
            seasons.append(s_meta["season_type"])
            season_years.append(s_meta["season_year"])
            season_labels.append(s_meta["season_label"])

        merged_df["active_season"] = seasons
        merged_df["season_year"] = season_years
        merged_df["season_label"] = season_labels

        return merged_df

    def link_contextual_statistics(
        self,
        weekly_df: pd.DataFrame,
        clean_doa_df: Optional[pd.DataFrame] = None,
        clean_dcs_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Link seasonal DOA cultivation targets and annual DCS baseline statistics
        as macro-contextual attributes without data smearing or weekly fabrication.

        CRITICAL INTEGRITY RULE:
        Under no circumstances are annual or seasonal totals divided by 52 or distributed
        into artificial weekly production values. They are attached as static contextual steps
        representing the active operational planning environment.

        Args:
            weekly_df: Weekly aligned DataFrame (from align_price_and_weather).
            clean_doa_df: Optional DataFrame of seasonal DOA cultivation targets.
            clean_dcs_df: Optional DataFrame of annual/baseline DCS production and yield metrics.

        Returns:
            Enriched weekly DataFrame with contextual attributes.
        """
        if weekly_df.empty:
            return weekly_df.copy()

        res_df = weekly_df.copy()

        # Link DOA Seasonal Cultivation Targets
        if clean_doa_df is not None and not clean_doa_df.empty:
            validate_dataframe(clean_doa_df, CLEAN_DOA_REQUIRED_COLUMNS, "CleanDOA", allow_empty=True)
            
            # Standardize crop names in DOA dataset to match canonical vocabulary
            doa_copy = clean_doa_df.copy()
            doa_enforcer = CropVocabularyEnforcer()
            doa_copy = doa_enforcer.filter_and_standardize(doa_copy, crop_col="crop")

            # Join keys: (district, crop, active_season == season, season_year == year)
            doa_cols_to_merge = [
                "district", "crop", "season", "year",
                "target_extent_ha", "cultivated_extent_ha", "expected_production_mt"
            ]
            available_doa_cols = [c for c in doa_cols_to_merge if c in doa_copy.columns]
            
            doa_subset = doa_copy[available_doa_cols].rename(columns={
                "crop": "crop_name",
                "season": "active_season",
                "year": "season_year",
                "target_extent_ha": "doa_target_extent_ha",
                "cultivated_extent_ha": "doa_cultivated_extent_ha",
                "expected_production_mt": "doa_expected_production_mt",
            })

            res_df = pd.merge(
                res_df,
                doa_subset,
                on=["district", "crop_name", "active_season", "season_year"],
                how="left",
            )
        else:
            res_df["doa_target_extent_ha"] = np.nan
            res_df["doa_cultivated_extent_ha"] = np.nan
            res_df["doa_expected_production_mt"] = np.nan

        # Link DCS Annual Baseline Statistics (e.g. historical yield)
        if clean_dcs_df is not None and not clean_dcs_df.empty:
            validate_dataframe(clean_dcs_df, CLEAN_DCS_REQUIRED_COLUMNS, "CleanDCS", allow_empty=True)
            
            dcs_copy = clean_dcs_df.copy()
            dcs_enforcer = CropVocabularyEnforcer()
            dcs_copy = dcs_enforcer.filter_and_standardize(dcs_copy, crop_col="crop")

            # Filter for yield metric if present
            yield_mask = dcs_copy["metric"].str.lower().str.contains("yield")
            if yield_mask.any():
                yield_df = dcs_copy[yield_mask].copy()
                # If multiple years exist, take latest available per (district, crop)
                if "year" in yield_df.columns:
                    yield_df = yield_df.sort_values(by="year").groupby(["district", "crop"], as_index=False).last()
                else:
                    yield_df = yield_df.groupby(["district", "crop"], as_index=False).first()

                yield_subset = yield_df[["district", "crop", "value"]].rename(columns={
                    "crop": "crop_name",
                    "value": "dcs_baseline_yield_kg_ha",
                })

                res_df = pd.merge(
                    res_df,
                    yield_subset,
                    on=["district", "crop_name"],
                    how="left",
                )
            else:
                res_df["dcs_baseline_yield_kg_ha"] = np.nan
        else:
            res_df["dcs_baseline_yield_kg_ha"] = np.nan

        return res_df

    def create_district_rollup(self, granular_weekly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Roll up granular market-level weekly records to the district level.

        Preserves district weather, computes district-average prices across markets,
        counts reporting markets (market_count), and sums total observations (price_obs_count).

        Args:
            granular_weekly_df: Weekly granular market-level DataFrame.

        Returns:
            District-level weekly DataFrame.
        """
        if granular_weekly_df.empty:
            return pd.DataFrame()

        group_cols = [
            "district",
            "crop_name",
            "origin_variety",
            "price_type",
            "iso_year",
            "iso_week",
            "week_start_date",
            "week_end_date",
            "active_season",
            "season_year",
        ]

        # Aggregate across markets within each district
        rows = []
        for group_keys, group_df in granular_weekly_df.groupby(group_cols, as_index=False, dropna=False):
            (
                district,
                crop_name,
                origin_variety,
                price_type,
                iso_year,
                iso_week,
                week_start,
                week_end,
                active_season,
                season_year,
            ) = group_keys

            mean_p = float(group_df["weekly_price_mean"].mean())
            min_p = float(group_df["weekly_price_min"].min())
            max_p = float(group_df["weekly_price_max"].max())
            total_obs = int(group_df["price_obs_count"].sum())
            markets = list(group_df["market_name"].dropna().unique())
            market_count = len(markets)

            row = {
                "district": district,
                "crop_name": crop_name,
                "origin_variety": origin_variety,
                "price_type": price_type,
                "iso_year": iso_year,
                "iso_week": iso_week,
                "week_start_date": week_start,
                "week_end_date": week_end,
                "active_season": active_season,
                "season_year": season_year,
                "weekly_price_mean": round(mean_p, self.decimals),
                "weekly_price_min": round(min_p, self.decimals),
                "weekly_price_max": round(max_p, self.decimals),
                "price_obs_count": total_obs,
                "market_count": market_count,
                "reporting_markets": ", ".join(markets),
                "unit": "Rs./kg",
            }

            # Retain weather features if present
            for w_col in [
                "weekly_precip_sum_mm", "weekly_temp_mean_c", "weekly_temp_min_c",
                "weekly_temp_max_c", "weekly_humidity_mean_pct", "weekly_wind_speed_mean_ms",
                "weather_obs_count"
            ]:
                if w_col in group_df.columns:
                    val = group_df[w_col].iloc[0]
                    row[w_col] = val if pd.notna(val) else None

            # Retain contextual statistics if present
            for c_col in [
                "doa_target_extent_ha", "doa_cultivated_extent_ha", "doa_expected_production_mt",
                "dcs_baseline_yield_kg_ha"
            ]:
                if c_col in group_df.columns:
                    val = group_df[c_col].iloc[0]
                    row[c_col] = val if pd.notna(val) else None

            rows.append(row)

        rollup_df = pd.DataFrame(rows)
        if not rollup_df.empty:
            rollup_df = rollup_df.sort_values(
                by=["district", "crop_name", "iso_year", "iso_week"]
            ).reset_index(drop=True)

        return rollup_df

    def export_prototype_dataset(
        self,
        district_weekly_df: pd.DataFrame,
        price_type: str = "retail",
    ) -> pd.DataFrame:
        """
        Format district-level weekly dataset for 100% backward compatibility
        with data/v1/analysis_ready_price.csv.

        Args:
            district_weekly_df: District-level weekly aggregated DataFrame.
            price_type: 'retail' or 'wholesale'.

        Returns:
            DataFrame formatted to match the analysis_ready_price.csv schema.
        """
        return export_to_prototype_v1_format(
            district_weekly_df,
            price_type=price_type,
            source_label="aligned_pipeline",
        )
