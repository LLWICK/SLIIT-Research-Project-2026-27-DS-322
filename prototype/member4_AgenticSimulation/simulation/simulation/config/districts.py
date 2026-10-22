"""District targets used as MSRS fair shares.

Carrot hectares are the area targets on the Maha 2024/25 final report
(2025-04-15). Leeks hectares are the area targets on the 2025 Yala final
report (2025-10-15). Names are the spellings stored in the warehouse.
Tomato has no checked district table, so it is not given a share.
"""

from __future__ import annotations

from dataclasses import dataclass

# crop_code, district, target hectares
DISTRICT_TARGETS: tuple[tuple[str, str, float], ...] = (
    ("carrot", "Nuwaraeliya", 1224.0),
    ("carrot", "Badulla", 614.0),
    ("carrot", "Kandy", 65.0),
    ("carrot", "Jaffna", 38.0),
    ("carrot", "Kilinochchi", 12.0),
    ("carrot", "Ampara", 10.0),
    ("carrot", "Vavuniya", 8.0),
    ("carrot", "Trincomalee", 7.0),
    ("carrot", "Mannar", 6.0),
    ("carrot", "Mullathivu", 6.0),
    ("carrot", "Ratnapura", 6.0),
    ("carrot", "Moneragala", 4.0),
    ("carrot", "Kurunegala", 1.0),
    ("leeks", "Nuwaraeliya", 1080.0),
    ("leeks", "Badulla", 190.0),
    ("leeks", "Kandy", 53.0),
    ("leeks", "Jaffna", 8.0),
    ("leeks", "Ratnapura", 2.0),
)

ADVISORY_CROPS = ("carrot", "leeks")


@dataclass(frozen=True)
class DistrictTarget:
    crop_code: str
    district: str
    target_ha: float


def district_targets(crop_code: str | None = None) -> tuple[DistrictTarget, ...]:
    rows = tuple(
        DistrictTarget(crop, district, hectares)
        for crop, district, hectares in DISTRICT_TARGETS
        if crop_code is None or crop == crop_code
    )
    if crop_code is not None and not rows:
        raise KeyError(crop_code)
    return rows


def fair_share_ha(crop_code: str, district: str) -> float:
    """Checked target for this district. Zero means the crop is not grown there."""

    try:
        targets = district_targets(crop_code)
    except KeyError:
        return 0.0
    for target in targets:
        if target.district == district:
            return target.target_ha
    return 0.0


def district_land_ha() -> dict[str, float]:
    """Sum of checked carrot and leeks targets in each district."""

    land: dict[str, float] = {}
    for target in district_targets():
        land[target.district] = land.get(target.district, 0.0) + target.target_ha
    return land
