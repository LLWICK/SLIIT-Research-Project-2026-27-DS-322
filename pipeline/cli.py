"""Command-line interface for Member 1 Data Engineering Pipeline.

Provides automated extraction, cleaning, and validation commands.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pipeline.config import (
    DISTRICT_COORDINATES,
    EXPORTS_DIR,
    PRIORITY_VEGETABLE_DISTRICTS,
    PROCESSED_WEATHER_DIR,
    RAW_WEATHER_DIR,
)
from pipeline.extractors.weather_nasa import NasaPowerWeatherExtractor
from pipeline.transformers.weather_cleaner import WeatherCleaner
from pipeline.validation.validator import WeatherValidator


def run_weather_pipeline(
    districts: list[str],
    start_date: str,
    end_date: str,
    save_export: bool = True,
) -> int:
    """Run extraction -> cleaning -> validation -> export for weather."""
    print("=" * 70)
    print("MEMBER 1 DATA ENGINEERING PIPELINE: NASA POWER WEATHER")
    print("=" * 70)
    print(f"Districts:  {districts}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Raw Dir:    {RAW_WEATHER_DIR}")
    print(f"Clean Dir:  {PROCESSED_WEATHER_DIR}")
    print("=" * 70)

    extractor = NasaPowerWeatherExtractor(output_dir=RAW_WEATHER_DIR)
    cleaner = WeatherCleaner(output_dir=PROCESSED_WEATHER_DIR)
    validator = WeatherValidator()

    all_cleaned_dfs: list[pd.DataFrame] = []
    success_count = 0

    for district in districts:
        print(f"\n>>> Processing District: {district} ...")
        try:
            # 1. Extraction (Raw NASA POWER JSON + Manifest)
            extraction_res = extractor.extract_point(
                district=district,
                start_date=start_date,
                end_date=end_date,
            )
            raw_json = extraction_res["raw_json"]
            print(f"  [1/3] Extracted raw JSON to: {extraction_res['raw_path']}")
            print(f"        Provenance manifest:   {extraction_res['manifest_path']}")

            # 2. Cleaning & Standardization (Dates ISO, Null preservation, snake_case)
            clean_df = cleaner.clean_raw_payload(
                raw_json=raw_json,
                district=district,
                latitude=extraction_res["latitude"],
                longitude=extraction_res["longitude"],
            )
            clean_csv_path = cleaner.save_clean(
                df=clean_df,
                district=district,
                start_date=start_date,
                end_date=end_date,
                file_format="csv",
            )
            print(f"  [2/3] Cleaned dataset saved to: {clean_csv_path}")

            # 3. Validation
            report = validator.validate(clean_df)
            status_text = "[PASS]" if report["is_valid"] else "[FAIL]"
            print(f"  [3/3] Validation {status_text}: {'PASSED' if report['is_valid'] else 'FAILED'}")
            print(f"        Observations: {report['total_rows']} daily records")
            for col, stats in report["completeness"].items():
                print(
                    f"        - {col:22s}: {stats['valid_observations']}/{stats['total_observations']} "
                    f"({stats['completeness_pct']:.1f}% complete) | Mean: {stats['mean_observed']}"
                )

            if report["warnings"]:
                for w in report["warnings"]:
                    print(f"        [!] Warning: {w}")

            all_cleaned_dfs.append(clean_df)
            success_count += 1

        except Exception as e:
            print(f"  [ERROR] Failed processing {district}: {e}", file=sys.stderr)

    # 4. Export consolidated dataset for Member 3 & Member 4 if requested
    if save_export and all_cleaned_dfs:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        consolidated_df = pd.concat(all_cleaned_dfs, ignore_index=True)
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        export_csv = EXPORTS_DIR / f"sri_lanka_vegetable_weather_{start_fmt}_{end_fmt}.csv"
        consolidated_df.to_csv(export_csv, index=False)
        print("\n" + "=" * 70)
        print(f"CONSOLIDATED EXPORT: {export_csv} ({len(consolidated_df):,} rows)")
        print("Ready for consumption by Member 3 (MSRS/DCVS) & Member 4 (Agent Simulation).")
        print("=" * 70)

    print(f"\nPipeline finished. Successfully processed {success_count}/{len(districts)} districts.")
    return 0 if success_count == len(districts) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Member 1 Data Engineering Pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Weather pipeline parser
    weather_parser = subparsers.add_parser("weather", help="Run NASA POWER weather pipeline")
    weather_parser.add_argument(
        "--districts",
        type=str,
        default=",".join(PRIORITY_VEGETABLE_DISTRICTS[:2]),
        help="Comma-separated district names or 'all' for all 25 districts",
    )
    weather_parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    weather_parser.add_argument(
        "--end",
        type=str,
        default="2024-01-10",
        help="End date (YYYY-MM-DD)",
    )
    weather_parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip consolidated export",
    )

    args = parser.parse_args()

    if args.command == "weather":
        if args.districts.strip().lower() == "all":
            dist_list = sorted(DISTRICT_COORDINATES.keys())
        else:
            dist_list = [d.strip() for d in args.districts.split(",") if d.strip()]

        code = run_weather_pipeline(
            districts=dist_list,
            start_date=args.start,
            end_date=args.end,
            save_export=not args.no_export,
        )
        sys.exit(code)


if __name__ == "__main__":
    main()
