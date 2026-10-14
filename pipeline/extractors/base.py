"""Base Extractor providing resilient HTTP session management, logging, and provenance recording."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure standard logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BaseExtractor:
    """Base class for all data extractors."""

    def __init__(
        self,
        source_name: str,
        timeout: tuple[int, int] = (10, 60),
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        self.source_name = source_name
        self.timeout = timeout
        self.logger = logging.getLogger(f"extractor.{source_name}")

        # Setup resilient requests session with exponential backoff
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {"User-Agent": "SLIIT-Research-Project-DS322-DataPipeline/1.0"}
        )

    def compute_sha256(self, content: bytes | str) -> str:
        """Compute SHA256 checksum of content for data provenance tracking."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def record_manifest(
        self,
        manifest_path: Path,
        source_url: str,
        parameters: dict[str, Any],
        raw_payload_bytes: bytes,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Save an immutable JSON metadata manifest accompanying the raw payload."""
        manifest = {
            "source_name": self.source_name,
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": source_url,
            "parameters": parameters,
            "raw_sha256": self.compute_sha256(raw_payload_bytes),
            "payload_size_bytes": len(raw_payload_bytes),
            **(extra_metadata or {}),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self.logger.info(f"Recorded extraction manifest to {manifest_path}")
        return manifest
