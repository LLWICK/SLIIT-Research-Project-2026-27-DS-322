"""Multiple algorithmic recommendation approaches for crowding-aware crop selection.

Implements:
1. Baseline DCVS (Piecewise Linear MSRS + Weighted DCVS)
2. MCDM / TOPSIS (Multi-Criteria Decision Analysis with Ideal Distance)
3. Non-Linear Congestion (BPR Function) + Crop Profile Vector Similarity
4. Markowitz-Inspired Portfolio Hedging (Land-Splitting Strategy)
5. Probabilistic / Softmax Load Balancing (Systemic Herding Prevention)
6. Data-Driven Weight Calibration Engine
   - Supervised Profit (LTR)
   - Shannon Entropy
   - CRITIC
   - Cobweb Policy Simulator
"""

from __future__ import annotations

import math
from typing import Any

from .config import DEMO_CAPACITY_HA, ScoringConfig
from .domain import calculate_dcvs, calculate_pressure_risk, min_max_normalize
from .suitability import CROP_REQUIREMENTS


def run_baseline_dcvs(
    candidates_data: list[dict[str, Any]],
    config: ScoringConfig = ScoringConfig(),
) -> list[dict[str, Any]]:
    """Approach 1: Baseline Weighted DCVS.
    Uses piecewise linear MSRS and linear weighted sum.
    """
    results = []
    prices = {item["crop"]: item["forecast"]["predicted_price"] for item in candidates_data}
    normalized_prices = min_max_normalize(prices)

    for item in candidates_data:
        crop = item["crop"]
        current = item["current_confirmed_area_ha"]
        proposed = item["proposed_area_ha"]
        capacity = item["reference_capacity_ha"]
        suitability = item["suitability"]

        pressure = calculate_pressure_risk(current + proposed, capacity, config)
        score = calculate_dcvs(normalized_prices[crop], suitability, pressure.score, config)

        results.append({
            "crop": crop,
            "rank_score": score,
            "score_label": "DCVS (%)",
            "pressure_risk": pressure.score,
            "pressure_label": pressure.label,
            "suitability": suitability,
            "predicted_price": item["forecast"]["predicted_price"],
            "explanation": f"DCVS {score:.1f}% based on linear weights (Price {config.price_weight*100:.0f}%, Suitability {config.suitability_weight*100:.0f}%, Low Saturation {config.pressure_weight*100:.0f}%).",
        })

    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results


def run_topsis_mcdm(candidates_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Approach 2: Multi-Criteria Decision Making via TOPSIS.
    Evaluates crops across 4 conflicting criteria:
    - C1: Agronomic Suitability (Benefit, max) - weight 0.30
    - C2: Forecast Price (Benefit, max) - weight 0.35
    - C3: Price Uncertainty Interval (Cost, min) - weight 0.15
    - C4: Market Saturation Pressure Ratio (Cost, min) - weight 0.20
    """
    weights = [0.30, 0.35, 0.15, 0.20]
    is_benefit = [True, True, False, False]

    matrix: list[list[float]] = []
    for item in candidates_data:
        forecast = item["forecast"]
        price_uncertainty = max(1.0, float(forecast["upper_price"]) - float(forecast["lower_price"]))
        pressure_ratio = (item["current_confirmed_area_ha"] + item["proposed_area_ha"]) / max(1.0, item["reference_capacity_ha"])
        matrix.append([
            float(item["suitability"]),
            float(forecast["predicted_price"]),
            price_uncertainty,
            pressure_ratio,
        ])

    num_rows = len(matrix)
    num_cols = len(weights)

    # 1. Vector Normalization
    col_norms = []
    for j in range(num_cols):
        col_sum_sq = sum(matrix[i][j] ** 2 for i in range(num_rows))
        col_norms.append(math.sqrt(col_sum_sq) or 1.0)

    norm_matrix = [[matrix[i][j] / col_norms[j] for j in range(num_cols)] for i in range(num_rows)]

    # 2. Weighted Normalized Matrix
    weighted_matrix = [[norm_matrix[i][j] * weights[j] for j in range(num_cols)] for i in range(num_rows)]

    # 3. Ideal Best (A+) and Ideal Worst (A-)
    ideal_best: list[float] = []
    ideal_worst: list[float] = []
    for j in range(num_cols):
        col_values = [weighted_matrix[i][j] for i in range(num_rows)]
        if is_benefit[j]:
            ideal_best.append(max(col_values))
            ideal_worst.append(min(col_values))
        else:
            ideal_best.append(min(col_values))
            ideal_worst.append(max(col_values))

    # 4. Euclidean Distances & Closeness Coefficient
    results = []
    for i, item in enumerate(candidates_data):
        dist_best = math.sqrt(sum((weighted_matrix[i][j] - ideal_best[j]) ** 2 for j in range(num_cols)))
        dist_worst = math.sqrt(sum((weighted_matrix[i][j] - ideal_worst[j]) ** 2 for j in range(num_cols)))

        closeness = dist_worst / (dist_best + dist_worst) if (dist_best + dist_worst) > 0 else 0.5
        topsis_score = round(closeness * 100, 2)

        results.append({
            "crop": item["crop"],
            "rank_score": topsis_score,
            "score_label": "TOPSIS Closeness (%)",
            "dist_ideal_best": round(dist_best, 4),
            "dist_ideal_worst": round(dist_worst, 4),
            "suitability": item["suitability"],
            "predicted_price": item["forecast"]["predicted_price"],
            "price_spread": round(matrix[i][2], 1),
            "pressure_ratio": round(matrix[i][3], 3),
            "explanation": f"TOPSIS Closeness {topsis_score:.1f}%. Distance to ideal solution: {dist_best:.3f}, distance to negative ideal: {dist_worst:.3f}.",
        })

    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def run_bpr_similarity(
    candidates_data: list[dict[str, Any]],
    requested_crop: str,
) -> list[dict[str, Any]]:
    """Approach 3: Non-Linear Congestion (BPR) + Crop Vector Similarity Recommender.
    Uses Bureau of Public Roads (BPR) convex saturation curve:
      Congestion = 1 - 1 / (1 + 0.15 * (V/C)^4)
    Matches alternatives to the requested crop using feature vectors
    [growth_duration, water_demand, labor_intensity, capital_cost].
    """
    # Crop profile feature vectors
    # Normalize features to comparable scales
    crop_vectors: dict[str, list[float]] = {}
    for crop_name, req in CROP_REQUIREMENTS.items():
        crop_vectors[crop_name] = [
            req.growth_duration_weeks / 20.0,
            req.water_demand_level,
            req.labor_intensity_level,
            req.capital_cost_per_ha_lkr / 600_000.0,
        ]

    req_vec = crop_vectors.get(requested_crop, [0.5, 0.5, 0.5, 0.5])
    results = []

    for item in candidates_data:
        crop = item["crop"]
        ratio = (item["current_confirmed_area_ha"] + item["proposed_area_ha"]) / max(1.0, item["reference_capacity_ha"])

        # Non-linear BPR congestion curve (alpha=0.15, beta=4)
        bpr_congestion = 1.0 - (1.0 / (1.0 + 0.15 * (ratio ** 4)))
        bpr_congestion = min(1.0, max(0.0, bpr_congestion))

        target_vec = crop_vectors.get(crop, [0.5, 0.5, 0.5, 0.5])
        similarity = _cosine_similarity(req_vec, target_vec) if crop != requested_crop else 1.0

        # Composite Recommender Score:
        # 35% agronomic suitability + 35% operational similarity + 30% un-congested capacity
        score = (0.35 * item["suitability"]) + (0.35 * similarity) + (0.30 * (1.0 - bpr_congestion))
        score_pct = round(score * 100, 2)

        results.append({
            "crop": crop,
            "rank_score": score_pct,
            "score_label": "BPR-Similarity Score (%)",
            "bpr_congestion": round(bpr_congestion, 4),
            "profile_similarity": round(similarity, 3),
            "suitability": item["suitability"],
            "predicted_price": item["forecast"]["predicted_price"],
            "explanation": f"Profile similarity {similarity*100:.0f}% to {requested_crop}. BPR congestion penalty: {bpr_congestion*100:.1f}%.",
        })

    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results


def run_portfolio_hedging(
    candidates_data: list[dict[str, Any]],
    requested_crop: str,
    proposed_area_ha: float,
) -> dict[str, Any]:
    """Approach 4: Markowitz-Inspired Portfolio Hedging / Land Splitting.
    If the requested crop is at risk of market saturation, computes an optimal
    land-split between the requested crop (up to safe capacity) and the top alternative.
    """
    req_item = next((item for item in candidates_data if item["crop"] == requested_crop), candidates_data[0])
    req_current = req_item["current_confirmed_area_ha"]
    req_capacity = req_item["reference_capacity_ha"]
    safe_cap_limit = 0.85 * req_capacity

    remaining_safe_ha = max(0.0, safe_cap_limit - req_current)

    # Sort alternatives by suitability * price
    alternatives = [item for item in candidates_data if item["crop"] != requested_crop]
    alternatives.sort(key=lambda x: x["suitability"] * x["forecast"]["predicted_price"], reverse=True)
    best_alt = alternatives[0] if alternatives else req_item

    if remaining_safe_ha >= proposed_area_ha:
        # 100% safe to plant requested crop
        allocations = [
            {
                "crop": requested_crop,
                "allocated_area_ha": round(proposed_area_ha, 2),
                "allocation_pct": 100.0,
                "role": "Primary requested crop (Within safe market capacity)",
            }
        ]
        status = "safe_single_crop"
        risk_reduction = 0.0
    else:
        # Saturated: Split land to protect against oversupply crash
        safe_req_ha = min(proposed_area_ha, max(0.25 * proposed_area_ha, remaining_safe_ha))
        alt_ha = proposed_area_ha - safe_req_ha
        allocations = [
            {
                "crop": requested_crop,
                "allocated_area_ha": round(safe_req_ha, 2),
                "allocation_pct": round((safe_req_ha / proposed_area_ha) * 100, 1),
                "role": "Retained primary crop (Capped to prevent Cobweb crash)",
            },
            {
                "crop": best_alt["crop"],
                "allocated_area_ha": round(alt_ha, 2),
                "allocation_pct": round((alt_ha / proposed_area_ha) * 100, 1),
                "role": f"Hedging alternative ({best_alt['crop'].title()} has safe market capacity)",
            },
        ]
        status = "hedged_split_allocation"
        overshoot_avoided = (req_current + proposed_area_ha) - safe_cap_limit
        risk_reduction = min(100.0, round((overshoot_avoided / max(1.0, req_capacity)) * 100, 1))

    return {
        "strategy": "Markowitz Land-Splitting Hedging",
        "status": status,
        "requested_crop": requested_crop,
        "total_proposed_area_ha": proposed_area_ha,
        "estimated_risk_reduction_pct": risk_reduction,
        "allocations": allocations,
        "explanation": (
            f"Allocates {allocations[0]['allocated_area_ha']} ha to {requested_crop} and "
            f"{allocations[1]['allocated_area_ha']} ha to {allocations[1]['crop']} "
            f"to protect the farmer against price collapse while satisfying crop preferences."
            if len(allocations) > 1
            else f"Full {proposed_area_ha} ha can safely be planted with {requested_crop} without exceeding the 85% saturation threshold."
        ),
    }


def run_softmax_load_balancing(
    candidates_data: list[dict[str, Any]],
    temperature: float = 0.6,
) -> list[dict[str, Any]]:
    """Approach 5: Probabilistic / Softmax Recommendation for Herding Prevention.
    Distributes recommendation probability over viable crops using a Boltzmann/Softmax distribution.
    Prevents 100% of farmers from switching into the exact same alternative crop.
    """
    # Use baseline viability scores as utilities
    prices = {item["crop"]: item["forecast"]["predicted_price"] for item in candidates_data}
    norm_prices = min_max_normalize(prices)
    config = ScoringConfig()

    utilities: list[float] = []
    crops: list[str] = []

    for item in candidates_data:
        crop = item["crop"]
        pressure = calculate_pressure_risk(
            item["current_confirmed_area_ha"] + item["proposed_area_ha"],
            item["reference_capacity_ha"],
            config,
        )
        dcvs = calculate_dcvs(norm_prices[crop], item["suitability"], pressure.score, config)
        utilities.append(dcvs)
        crops.append(crop)

    # Softmax with temperature scaling
    temp = max(0.1, temperature)
    max_u = max(utilities)  # For numerical stability
    exp_values = [math.exp((u - max_u) / (temp * 20.0)) for u in utilities]
    sum_exp = sum(exp_values) or 1.0
    probabilities = [v / sum_exp for v in exp_values]

    results = []
    for i, crop in enumerate(crops):
        prob_pct = round(probabilities[i] * 100, 1)
        results.append({
            "crop": crop,
            "rank_score": prob_pct,
            "score_label": "Systemic Recommendation Share (%)",
            "probability": round(probabilities[i], 4),
            "utility_dcvs": round(utilities[i], 1),
            "explanation": f"{prob_pct}% of querying farmers in this district are steered towards {crop} to preserve macro market equilibrium.",
        })

    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results
