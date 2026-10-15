"""Member 1 Data Engineering CLI.

Command line interface for managing, executing, and validating real data ETL pipelines
for Sri Lankan vegetable market prices (CBSL and HARTI) and agroclimatology.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.extractors.price_cbsl import CbslPriceExtractor
from pipeline.extractors.price_harti import HartiPriceExtractor
from pipeline.quality.dri_calculator import DRICalculator
from pipeline.transformers.price_cleaner import PriceCleaner
from pipeline.validation.price_validator import PriceValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline.cli")


def run_price_pipeline(args: argparse.Namespace) -> int:
    """Execute vegetable price ETL workflow."""
    source = args.source.lower().strip()
    cleaner = PriceCleaner()
    validator = PriceValidator()

    raw_records = []
    source_agency = ""
    date_token = args.date or "unspecified"

    if source == "cbsl":
        source_agency = "CBSL"
        extractor = CbslPriceExtractor()
        if args.file:
            logger.info(f"Extracting CBSL prices from local file: {args.file}")
            raw_records = extractor.extract_from_file(args.file, report_date=args.date)
        elif args.date:
            logger.info(f"Fetching CBSL daily price report for date: {args.date}")
            pdf_bytes, manifest = extractor.fetch_report(args.date, save_raw=True)
            raw_records = extractor.extract_from_bytes(
                pdf_bytes=pdf_bytes,
                report_date=extractor.format_date_str(args.date)[0],
                source_url=manifest.get("source_url", ""),
                sha256_hash=manifest.get("raw_sha256", ""),
            )
        else:
            logger.error("Must specify either --date YYYY-MM-DD or --file <path> for CBSL price extraction.")
            return 1

    elif source == "harti":
        source_agency = "HARTI"
        extractor = HartiPriceExtractor()

        if args.discover:
            logger.info("Discovering available bulletin links on HARTI portal...")
            links = extractor.scrape_available_bulletin_links()
            print(f"\nFound {len(links)} daily bulletins on HARTI portal. Sample (latest 5):")
            for link in links[:5]:
                print(f"  {link['date']}: {link['full_url']}")
            return 0

        include_prev = getattr(args, "include_prev", False)
        if args.file:
            logger.info(f"Extracting HARTI prices from local bulletin: {args.file}")
            raw_records = extractor.extract_from_file(
                file_path=args.file,
                report_date=args.date,
                include_previous_day=include_prev,
            )
            if not raw_records:
                # If automated table extraction is unavailable (e.g. non-standard font encoding), record offline manifest
                pdf_bytes, manifest = extractor.ingest_offline_bulletin(args.file, report_date=args.date)
                print(f"Recorded offline HARTI manifest with SHA-256: {manifest.get('raw_sha256')}")
                print("Notice: No tabular DEC vegetable records could be extracted from this PDF format. Fallback manifest recorded.")
                return 0

        elif args.date:
            logger.info(f"Attempting live fetch for HARTI bulletin on date: {args.date}")
            try:
                pdf_bytes, manifest = extractor.fetch_bulletin(args.date, save_raw=True)
                print(f"Successfully downloaded HARTI bulletin. SHA-256: {manifest.get('raw_sha256')}")
                raw_records = extractor.extract_from_bytes(
                    pdf_bytes=pdf_bytes,
                    report_date=args.date,
                    source_url=manifest.get("source_url", ""),
                    sha256_hash=manifest.get("raw_sha256", ""),
                    include_previous_day=include_prev,
                )
            except Exception as exc:
                logger.error(f"Live HARTI fetch failed: {exc}")
                logger.info(
                    "Reminder: HARTI portal has known 404 links for recent months. "
                    "Use --file <path_to_bulletin.pdf> to ingest verified offline bulletins."
                )
                return 1
        else:
            logger.error("Must specify --date, --file, or --discover for HARTI.")
            return 1

    else:
        logger.error(f"Unsupported source agency '{args.source}'. Supported: 'cbsl', 'harti'.")
        return 1

    if not raw_records:
        logger.warning("No price records extracted.")
        return 1

    logger.info(f"Extracted {len(raw_records)} raw records from {source_agency}.")

    # Clean and standardize
    cleaned_df = cleaner.clean_records(raw_records, filter_target_crops=not args.all_crops)
    logger.info(f"Cleaned dataset contains {len(cleaned_df)} target vegetable price observations.")

    # Validate
    report = validator.validate(cleaned_df)
    validation_status = "PASSED" if report["is_valid"] else "FAILED"

    print("\n" + "=" * 60)
    print(f"PRICE VALIDATION REPORT — {source_agency}")
    print("=" * 60)
    print(f"Validation Status: {validation_status}")
    print(f"Total Observations: {report['total_rows']}")
    print(f"Date Range: {report['date_range']}")
    if report["errors"]:
        print("\nErrors:")
        for err in report["errors"]:
            print(f"  - {err}")
    if report["warnings"]:
        print("\nWarnings:")
        for warn in report["warnings"]:
            print(f"  - {warn}")

    print("\nTarget Crop Price Summaries (Rs./kg):")
    for crop, metrics in report["crop_summaries"].items():
        print(f"  - {crop:<15}: Obs={metrics['valid_prices']}/{metrics['total_records']}, "
              f"Mean={metrics['mean_price']}, Min={metrics['min_price']}, Max={metrics['max_price']}")

    # Compute Data Reliability Index (DRI)
    dri_calc = DRICalculator()
    source_meta = {
        "source_agency": source_agency,
        "input_date": args.date,
        "input_file": str(args.file) if args.file else None,
        "filter_target_crops": not args.all_crops,
        "force_save": args.force_save,
        "validation_is_valid": report["is_valid"],
        "validation_error_count": len(report["errors"]),
        "validation_warning_count": len(report["warnings"]),
    }
    dri_report = dri_calc.evaluate_price_dataset(
        df=cleaned_df,
        validation_report=report,
        evaluation_date=args.date,
        source_agency=source_agency,
        source_metadata=source_meta,
    )

    print("\n" + "=" * 60)
    print(f"DATA RELIABILITY INDEX (DRI) REPORT — {source_agency}")
    print("=" * 60)
    print(f"DRI Score:          {dri_report['calculated_dri']} / 100.0")
    print(f"Reliability Tier:   {dri_report['reliability_tier']}")
    print(f"Validation Status:  {validation_status}")
    print(f"Completeness Score: {dri_report['completeness_score']}")
    print(f"Timeliness Score:   {dri_report['timeliness_score']}")
    print(f"Consistency Score:  {dri_report['consistency_score']}")
    print(f"Audit Notes:        {dri_report['audit_notes']}")

    # Append audit record to local JSON Lines ledger
    ledger_path = getattr(args, "ledger_path", None)
    try:
        saved_ledger_path = dri_calc.save_audit_ledger_record(dri_report, output_path=ledger_path)
        print(f"\nDRI Audit Record appended to: {saved_ledger_path}")
    except Exception as exc:
        logger.error(f"Failed to record DRI audit to ledger: {exc}")
        print(f"\nWarning: Could not save DRI audit record to ledger: {exc}")

    # Save processed dataset if validation passed (or force_save enabled)
    if report["is_valid"] or args.force_save:
        if date_token == "unspecified" and raw_records:
            date_token = raw_records[0]["price_date"]
        clean_date_token = date_token.replace("-", "")
        out_path = cleaner.save_processed(
            cleaned_df,
            filename_prefix=f"prices_{clean_date_token}",
            source_agency=source_agency,
        )
        print(f"Processed data saved to: {out_path}")
    else:
        logger.warning("Dataset failed validation. Processed data was NOT saved (use --force-save to override).")

    return 0 if report["is_valid"] else 1


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Member 1: Data Engineering & Geo-Spatial DSS CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # Price pipeline parser
    price_parser = subparsers.add_parser("price", help="Run vegetable price ETL pipeline")
    price_parser.add_argument("--source", type=str, required=True, choices=["cbsl", "harti"], help="Price source agency")
    price_parser.add_argument("--date", type=str, default=None, help="Report date (YYYY-MM-DD or YYYYMMDD)")
    price_parser.add_argument("--file", type=str, default=None, help="Local PDF/bulletin file path for offline processing")
    price_parser.add_argument("--discover", action="store_true", help="Scrape portal to list available bulletin links (HARTI)")
    price_parser.add_argument("--all-crops", action="store_true", help="Retain all food commodities rather than filtering to 5 target crops")
    price_parser.add_argument("--include-prev", action="store_true", help="Include observations flagged from previous business day")
    price_parser.add_argument("--force-save", action="store_true", help="Save processed output even if warnings occur")
    price_parser.add_argument("--ledger-path", type=str, default=None, help="Custom destination path for the JSON Lines audit ledger")

    args = parser.parse_args()

    if args.command == "price":
        sys.exit(run_price_pipeline(args))


if __name__ == "__main__":
    main()
