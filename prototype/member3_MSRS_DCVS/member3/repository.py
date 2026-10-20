"""SQLite-backed confirmed-commitment repository for the prototype."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class CommitmentRepository:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS commitments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    farmer_reference TEXT NOT NULL,
                    district TEXT NOT NULL,
                    market TEXT NOT NULL,
                    crop TEXT NOT NULL,
                    season TEXT NOT NULL,
                    planting_date TEXT NOT NULL,
                    harvest_date TEXT NOT NULL,
                    area_ha REAL NOT NULL CHECK(area_ha > 0),
                    status TEXT NOT NULL CHECK(status IN ('confirmed', 'cancelled')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        fields = ("idempotency_key", "farmer_reference", "district", "market", "crop", "season", "planting_date", "harvest_date", "area_ha", "status")
        with self.connection() as conn:
            existing = conn.execute("SELECT * FROM commitments WHERE idempotency_key = ?", (data["idempotency_key"],)).fetchone()
            if existing:
                return dict(existing)
            conn.execute(
                f"INSERT INTO commitments ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})",
                tuple(data[field] for field in fields),
            )
            return dict(conn.execute("SELECT * FROM commitments WHERE id = last_insert_rowid()").fetchone())

    def aggregate_area(self, district: str, market: str, crop: str, season: str) -> float:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT COALESCE(SUM(area_ha), 0) AS area FROM commitments
                   WHERE district = ? AND market = ? AND crop = ? AND season = ? AND status = 'confirmed'""",
                (district, market, crop, season),
            ).fetchone()
        return float(row["area"])

    def update_area(self, commitment_id: int, area_ha: float) -> dict[str, Any] | None:
        with self.connection() as conn:
            conn.execute("UPDATE commitments SET area_ha = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND status = 'confirmed'", (area_ha, commitment_id))
            row = conn.execute("SELECT * FROM commitments WHERE id = ?", (commitment_id,)).fetchone()
        return dict(row) if row else None

    def cancel(self, commitment_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            conn.execute("UPDATE commitments SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (commitment_id,))
            row = conn.execute("SELECT * FROM commitments WHERE id = ?", (commitment_id,)).fetchone()
        return dict(row) if row else None
