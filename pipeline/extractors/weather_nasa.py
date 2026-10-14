"""NASA POWER Daily Weather Extractor.

Queries the official NASA POWER Daily Point API for Agroclimatology (AG),
persisting raw immutable JSON responses alongside provenance manifests.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from pipeline.config import (
    DISTRICT_COORDINATES,
    NASA_PARAMETER_METADATA,
    NASA_POWER_API_URL,
    NASA_POWER_COMMUNITY,
    NASA_POWER_PARAMETERS,
    RAW_WEATHER_DIR,
)
from pipeline.extractors.base import BaseExtractor


class NasaPowerWeatherExtractor(BaseExtractor):
    """Extractor for NASA POWER Daily Point Agroclimatology API."""

    def __init__(
        self,
        output_dir: Path = RAW_WEATHER_DIR,
        timeout: tuple[int, int] = (10, 60),
    ) -> None:
        super().__init__(source_name="nasa_power", timeout=timeout)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_date(date_str: str) -> str:
        """Convert 'YYYY-MM-DD' or 'YYYYMMDD' to 'YYYYMMDD'."""
        cleaned = date_str.replace("-", "").strip()
        if len(cleaned) != 8 or not cleaned.isdigit():
            raise ValueError(
                f"Invalid date '{date_str}'. Expected format YYYY-MM-DD or YYYYMMDD."
            )
        # Validate calendar validity
        datetime.strptime(cleaned, "%Y%m%d")
        return cleaned

    def extract_point(
        self,
        district: str,
        start_date: str,
        end_date: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        parameters: Optional[list[str]] = None,
        save_raw: bool = True,
    ) -> dict[str, Any]:
        """Extract daily weather observations for a district over a date range.

        Args:
            district: Name of the Sri Lankan district.
            start_date: Start date string (YYYY-MM-DD or YYYYMMDD).
            end_date: End date string (YYYY-MM-DD or YYYYMMDD).
            latitude: Optional latitude override. If None, looked up from DISTRICT_COORDINATES.
            longitude: Optional longitude override. If None, looked up from DISTRICT_COORDINATES.
            parameters: List of NASA POWER parameter codes (default: PRECTOTCORR, T2M, etc.)
            save_raw: Whether to write the unmodified raw JSON and manifest to disk.

        Returns:
            Dictionary containing the raw JSON response and extraction metadata.
        """
        # Resolve coordinates
        if latitude is None or longitude is None:
            if district not in DISTRICT_COORDINATES:
                raise ValueError(
                    f"District '{district}' not recognized in verified coordinates. "
                    f"Available: {sorted(DISTRICT_COORDINATES.keys())}"
                )
            latitude, longitude = DISTRICT_COORDINATES[district]

        start_fmt = self._format_date(start_date)
        end_fmt = self._format_date(end_date)
        if int(start_fmt) > int(end_fmt):
            raise ValueError(f"start_date ({start_date}) must be before or equal to end_date ({end_date}).")

        param_list = parameters or NASA_POWER_PARAMETERS
        query_params = {
            "parameters": ",".join(param_list),
            "community": NASA_POWER_COMMUNITY,
            "longitude": f"{longitude:.4f}",
            "latitude": f"{latitude:.4f}",
            "start": start_fmt,
            "end": end_fmt,
            "format": "JSON",
        }

        self.logger.info(
            f"Querying NASA POWER for {district} (lat: {latitude:.4f}, lon: {longitude:.4f}) "
            f"from {start_fmt} to {end_fmt}"
        )

        try:
            response = self.session.get(
                NASA_POWER_API_URL,
                params=query_params,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error querying NASA POWER: {e}")
            raise

        if response.status_code != 200:
            error_msg = (
                f"NASA POWER API returned HTTP {response.status_code}: {response.text[:300]}"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        raw_bytes = response.content
        try:
            raw_json = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode NASA POWER JSON response: {e}")
            raise

        # Check for NASA POWER error messages inside 200 responses
        messages = raw_json.get("messages", [])
        if messages:
            self.logger.warning(f"NASA POWER API returned messages: {messages}")

        district_slug = district.lower().replace(" ", "_")
        raw_file_name = f"{district_slug}_{start_fmt}_{end_fmt}.json"
        manifest_file_name = f"{district_slug}_{start_fmt}_{end_fmt}_manifest.json"

        raw_path = self.output_dir / raw_file_name
        manifest_path = self.output_dir / manifest_file_name

        if save_raw:
            # Save raw payload byte-for-byte unchanged
            with open(raw_path, "wb") as f:
                f.write(raw_bytes)
            self.logger.info(f"Saved raw JSON to {raw_path} ({len(raw_bytes):,} bytes)")

            # Record provenance manifest
            extra_meta = {
                "district": district,
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_fmt,
                "end_date": end_fmt,
                "raw_file": raw_file_name,
                "api_header": raw_json.get("header", {}),
                "parameter_units": {
                    p: NASA_PARAMETER_METADATA.get(p, {}).get("units", "unknown")
                    for p in param_list
                },
            }
            self.record_manifest(
                manifest_path=manifest_path,
                source_url=response.url,
                parameters=query_params,
                raw_payload_bytes=raw_bytes,
                extra_metadata=extra_meta,
            )

        return {
            "district": district,
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_fmt,
            "end_date": end_fmt,
            "raw_json": raw_json,
            "raw_path": str(raw_path) if save_raw else None,
            "manifest_path": str(manifest_path) if save_raw else None,
        }

    def extract_districts(
        self,
        districts: list[str],
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Batch extract observations across multiple districts sequentially."""
        results = []
        for dist in districts:
            try:
                res = self.extract_point(
                    district=dist,
                    start_date=start_date,
                    end_date=end_date,
                )
                results.append(res)
            except Exception as e:
                self.logger.error(f"Failed extraction for district '{dist}': {e}")
                results.append({"district": dist, "error": str(e)})
        return results
