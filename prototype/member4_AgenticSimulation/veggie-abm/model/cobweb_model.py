"""
CobwebModel: the world the FarmerAgents act in. Each step() is one
growing season. Sequence per step:
  1. Agents look at last season's price (+ MSRS if enabled) and
     decide how much to plant -> register_commitment()
  2. Model aggregates all commitments per (district, crop)
  3. Model computes this season's MARKET price from aggregate supply
     (the actual cobweb mechanic: price is a function of how much
     got planted, decided BEFORE this price was known)
  4. Model computes MSRS for next season's agents to see
  5. DataCollector records the season's metrics
"""
from mesa import Model
from mesa.datacollection import DataCollector

from model.agent import FarmerAgent


class CobwebModel(Model):
    def __init__(self, n_farmers_per_district_crop: int, districts: list[str],
                 crops: list[str], base_price: dict, base_extent: dict,
                 msrs_enabled: bool, adoption_rate: float,
                 price_elasticity: float = 0.6, dampening_strength: float = 0.6,
                 msrs_smoothing_alpha: float = 0.6, rng=None):
        super().__init__(rng=rng)
        self.districts = districts
        self.crops = crops
        self.msrs_enabled = msrs_enabled
        self.adoption_rate = adoption_rate
        self.price_elasticity = price_elasticity
        self.dampening_strength = dampening_strength
        self.msrs_smoothing_alpha = msrs_smoothing_alpha

        self.base_price = dict(base_price)
        self.base_extent = dict(base_extent)

        self.current_price: dict[str, float] = dict(base_price)
        self.current_commitments: dict[tuple[str, str], float] = {}
        self.current_msrs: dict[tuple[str, str], float] = {
            (d, c): 0.0 for d in districts for c in crops
        }
        self.season_count = 0

        for district in districts:
            for crop in crops:
                per_farmer = base_extent[crop] / len(districts) / n_farmers_per_district_crop
                for _ in range(n_farmers_per_district_crop):
                    risk_tolerance = self.random.uniform(0.2, 0.9)
                    # Heterogeneous MSRS responsiveness beta_i ~ U(min_beta, max_beta)
                    min_beta = max(0.1, dampening_strength - 0.3)
                    max_beta = min(1.0, dampening_strength + 0.3)
                    msrs_sensitivity = self.random.uniform(min_beta, max_beta)
                    # Crop portfolio preferences: 1.0 for primary specialty, 0.4 - 0.7 for alternatives
                    candidate_crops = list(crops)
                    crop_preferences = {c: (1.0 if c == crop else self.random.uniform(0.4, 0.7)) for c in crops}
                    FarmerAgent(
                        self, district=district, primary_crop=crop,
                        candidate_crops=candidate_crops, crop_preferences=crop_preferences,
                        risk_tolerance=risk_tolerance, base_planting_hectares=per_farmer,
                        msrs_sensitivity=msrs_sensitivity
                    )

        self.datacollector = DataCollector(
            model_reporters={
                "season": lambda m: m.season_count,
                **{f"price_{c}": (lambda m, c=c: m.current_price.get(c)) for c in crops},
                **{f"extent_{c}": (lambda m, c=c: sum(
                    v for (d, cr), v in m.current_commitments.items() if cr == c)) for c in crops},
                **{f"equilibrium_{c}": (lambda m, c=c: max(
                    1.0 - abs(sum(v for (d, cr), v in m.current_commitments.items() if cr == c) - m.base_extent[c]) / m.base_extent[c],
                    0.0
                )) for c in crops},
                **{f"mean_profit_{c}": (lambda m, c=c: (
                    sum(a.last_profit for a in m.agents if getattr(a, "chosen_crop", getattr(a, "crop", None)) == c) /
                    max(sum(1 for a in m.agents if getattr(a, "chosen_crop", getattr(a, "crop", None)) == c), 1)
                )) for c in crops},
                **{f"loss_rate_{c}": (lambda m, c=c: (
                    sum(1 for a in m.agents if getattr(a, "chosen_crop", getattr(a, "crop", None)) == c and a.last_profit < 0) /
                    max(sum(1 for a in m.agents if getattr(a, "chosen_crop", getattr(a, "crop", None)) == c), 1)
                )) for c in crops},
            }
        )

    def get_last_price_signal(self, district: str, crop: str) -> float:
        """(price - base) / base — positive means last season was
        expensive (encourages planting more), negative means cheap."""
        price = self.current_price.get(crop, self.base_price[crop])
        return (price - self.base_price[crop]) / self.base_price[crop]

    def get_msrs(self, district: str, crop: str) -> float:
        return self.current_msrs.get((district, crop), 0.0)

    def register_commitment(self, district: str, crop: str, hectares: float):
        key = (district, crop)
        self.current_commitments[key] = self.current_commitments.get(key, 0.0) + hectares

    def step(self):
        self.current_commitments = {}
        self.agents.shuffle_do("step")  # each farmer decides + registers commitment

        # Price formation: aggregate oversupply relative to base
        # target pushes price down (the actual cobweb mechanic)
        for crop in self.crops:
            total_extent = sum(v for (d, c), v in self.current_commitments.items() if c == crop)
            oversupply = (total_extent - self.base_extent[crop]) / self.base_extent[crop]
            noise = self.rng.normal(1, 0.05)
            new_price = self.base_price[crop] * (1 - self.price_elasticity * oversupply) * noise
            self.current_price[crop] = max(new_price, self.base_price[crop] * 0.15)

        # Update each farmer agent's seasonal profit/revenue
        for agent in self.agents:
            if hasattr(agent, "update_financials"):
                chosen_c = getattr(agent, "chosen_crop", getattr(agent, "crop", None))
                realized_p = self.current_price.get(chosen_c, self.base_price.get(chosen_c, 100.0))
                agent.update_financials(realized_p)


        # MSRS for NEXT season: how saturated does this
        # district/crop look relative to its fair share of target?
        # Sensitivity note (found by testing): dividing by fair_share
        # alone meant MSRS only triggered when a district overshot
        # its ENTIRE fair share by 100%+ — essentially never happens
        # in practice, so MSRS sat near 0 and had no effect. Dividing
        # the overshoot by 0.3 means a 30% overshoot already maps to
        # MSRS=1, which is the range that actually matters for a
        # district-level crowding signal. Re-tune this if your real
        # calibration data suggests a different sensitivity.
        SATURATION_SENSITIVITY = 0.3
        for district in self.districts:
            for crop in self.crops:
                key = (district, crop)
                committed = self.current_commitments.get(key, 0.0)
                fair_share = self.base_extent[crop] / len(self.districts)
                if fair_share:
                    overshoot = (committed / fair_share) - 1
                    raw_msrs = min(max(overshoot / SATURATION_SENSITIVITY, 0.0), 1.0)
                else:
                    raw_msrs = 0.0

                # Exponential moving average smoothing across seasons
                prev_msrs = self.current_msrs.get(key, 0.0)
                self.current_msrs[key] = (
                    self.msrs_smoothing_alpha * raw_msrs +
                    (1.0 - self.msrs_smoothing_alpha) * prev_msrs
                )

        self.season_count += 1
        self.datacollector.collect(self)


