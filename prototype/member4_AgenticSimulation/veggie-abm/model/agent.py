"""
FarmerAgent: one virtual farmer making a per-season planting
decision. Two decision modes:
  - naive: react only to LAST season's own price (the classic
    cobweb behavior — plant more of what was expensive last time,
    regardless of what everyone else is about to do)
  - msrs_informed: also sees the Market Saturation Risk Score
    (MSRS) — a signal of how much OTHER farmers have already
    committed to this crop/season — and moderates the decision

Mesa 3.x API notes (checked directly against mesa==3.5.1, NOT the
older mesa.time.RandomActivation style):
  - Agent(model) auto-assigns self.unique_id — don't pass it yourself
  - No schedulers; the Model steps agents via self.agents.shuffle_do("step")
  - self.random is a per-agent RNG provided by Mesa, seeded from the model
"""
from mesa import Agent


class FarmerAgent(Agent):
    def __init__(self, model, district: str, crop: str = None,
                 primary_crop: str = None, candidate_crops: list[str] = None,
                 crop_preferences: dict[str, float] = None,
                 risk_tolerance: float = 0.5, base_planting_hectares: float = 1.0,
                 msrs_sensitivity: float = 0.6,
                 yield_per_hectare: float = 15000.0,
                 cost_fraction_of_base: float = 0.65):
        super().__init__(model)
        self.district = district
        self.primary_crop = primary_crop or crop
        self.crop = self.primary_crop
        self.chosen_crop = self.primary_crop
        self.candidate_crops = candidate_crops or [self.primary_crop]
        self.crop_preferences = crop_preferences or {
            c: (1.0 if c == self.primary_crop else 0.5) for c in self.candidate_crops
        }
        self.risk_tolerance = risk_tolerance          # 0 (cautious) - 1 (aggressive)
        self.msrs_sensitivity = msrs_sensitivity      # beta_i: heterogeneous responsiveness to MSRS
        self.base_planting_hectares = base_planting_hectares
        self.planned_hectares = base_planting_hectares
        self.yield_per_hectare = yield_per_hectare    # kg per hectare
        self.cost_fraction_of_base = cost_fraction_of_base
        self.last_revenue = 0.0
        self.last_profit = 0.0
        self.history: list[dict] = []

    def step(self):
        if self.model.msrs_enabled and self.random.random() < self.model.adoption_rate:
            self.chosen_crop, self.planned_hectares = self._decide_with_msrs()
        else:
            self.chosen_crop, self.planned_hectares = self._decide_naive()

        self.crop = self.chosen_crop
        self.model.register_commitment(self.district, self.chosen_crop, self.planned_hectares)

    def update_financials(self, realized_price: float):
        """Calculate realized revenue, production cost, and net seasonal profit for the chosen crop."""
        base_crop_price = self.model.base_price.get(self.chosen_crop, 100.0)
        cost_per_hectare = base_crop_price * self.yield_per_hectare * self.cost_fraction_of_base
        self.last_revenue = self.planned_hectares * self.yield_per_hectare * realized_price
        production_cost = self.planned_hectares * cost_per_hectare
        self.last_profit = self.last_revenue - production_cost
        self.history.append({
            "season": self.model.season_count,
            "crop": self.chosen_crop,
            "hectares": self.planned_hectares,
            "realized_price": realized_price,
            "revenue": self.last_revenue,
            "cost": production_cost,
            "profit": self.last_profit,
        })

    def _decide_naive(self) -> tuple[str, float]:
        """Classic cobweb behavior: evaluate candidate crops purely based on last
        season's price signals and historical crop preferences, ignoring market crowding."""
        best_crop = self.primary_crop
        best_score = -1.0

        for c in self.candidate_crops:
            last_price_signal = self.model.get_last_price_signal(self.district, c)
            pref = self.crop_preferences.get(c, 0.5)
            score = pref * max(1.0 + (self.risk_tolerance * last_price_signal), 0.05)
            if score > best_score:
                best_score = score
                best_crop = c

        price_signal = self.model.get_last_price_signal(self.district, best_crop)
        adjustment = 1.0 + (self.risk_tolerance * price_signal)
        planned = max(self.base_planting_hectares * adjustment, 0.0)
        return best_crop, planned

    def _decide_with_msrs(self) -> tuple[str, float]:
        """MSRS-informed behavior: evaluates candidates by penalizing crowded crops.
        If the primary crop is heavily saturated (high MSRS), the farmer dynamically
        switches land to a viable, uncrowded alternative crop."""
        best_crop = self.primary_crop
        best_score = -1.0

        for c in self.candidate_crops:
            last_price_signal = self.model.get_last_price_signal(self.district, c)
            msrs = self.model.get_msrs(self.district, c)
            dampening = max(1.0 - (msrs * self.msrs_sensitivity), 0.05)
            pref = self.crop_preferences.get(c, 0.5)
            score = pref * max(1.0 + (self.risk_tolerance * last_price_signal), 0.05) * dampening
            if score > best_score:
                best_score = score
                best_crop = c

        price_signal = self.model.get_last_price_signal(self.district, best_crop)
        msrs_val = self.model.get_msrs(self.district, best_crop)
        dampening = max(1.0 - (msrs_val * self.msrs_sensitivity), 0.05)
        adjustment = 1.0 + (self.risk_tolerance * price_signal)
        planned = max(self.base_planting_hectares * adjustment * dampening, 0.0)
        return best_crop, planned



