"""Market saturation score from Member 3's calculate_pressure_risk.

The formula is theirs: score rises from 0 at 80% of capacity to 1 at 120%
of capacity. Their demo capacities (carrot 1,000 ha, leek 800 ha, tomato
900 ha) are not used. Their README says those numbers are placeholders.
The capacity passed here is the checked district target.
"""

from __future__ import annotations

import sys
from pathlib import Path

from simulation.config.districts import fair_share_ha

_MEMBER3_ROOT = Path(__file__).resolve().parents[2] / "member3_MSRS_DCVS"
if str(_MEMBER3_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMBER3_ROOT))

from member3.config import ScoringConfig
from member3.domain import calculate_pressure_risk

MEMBER3_CONFIG = ScoringConfig()


def saturation_score(committed_ha: float, fair_share: float) -> float:
    """Member 3 pressure score for one district and crop."""

    if committed_ha < 0:
        raise ValueError("committed_ha must be zero or positive")
    if fair_share < 0:
        raise ValueError("fair_share must be zero or positive")
    if committed_ha == 0:
        return 0.0
    if fair_share == 0:
        return 1.0
    return calculate_pressure_risk(committed_ha, fair_share, MEMBER3_CONFIG).score


def district_scores(committed: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Score each (district, crop) pair present in the commitments."""

    scores: dict[tuple[str, str], float] = {}
    for (district, crop_code), hectares in committed.items():
        scores[(district, crop_code)] = saturation_score(
            hectares, fair_share_ha(crop_code, district)
        )
    return scores
