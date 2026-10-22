# Simulation

Member 4's planting simulation. It reads the local warehouse and does not write to it. Carrot and leeks can share land. Tomato runs only in the separate-crop baseline, because no tomato district table passed the extent sum check.

The data-layer guide is the project-root `README.md`. This file is only the simulation.

Prices are Colombo wholesale rupees per kilogram. Areas are hectares. A season is Maha (October–March) or Yala (April–September), using the same labels as `data_layer.season`. The first simulated season is 2015/16 Maha.

## Run

Python 3.14 or newer. From the project root, with the virtual environment. The local DuckDB warehouse must already exist (`data_layer/warehouse/cobweb.duckdb`). The calibration commands open it read-only.

```powershell
.\.venv\Scripts\python.exe -m simulation.scenarios.baseline
.\.venv\Scripts\python.exe -m simulation.calibration.compare
.\.venv\Scripts\python.exe -m simulation.calibration.baseline_path
.\.venv\Scripts\python.exe -m simulation.experiments.adoption
.\.venv\Scripts\python.exe -m unittest simulation.tests.test_baseline simulation.tests.test_calibration simulation.tests.test_advisory simulation.tests.test_price_process
```

Dashboard, on its own port so it does not take the data-layer page:

```powershell
.\.venv\Scripts\python.exe -m streamlit run simulation/dashboard.py --server.port 8502
```

The first dashboard load fits the price path and runs the adoption comparison. Later loads use a five-minute cache.

## What is in the package

| Path | Role |
| --- | --- |
| `config/settings.py` | Crop references, slopes, season sequence, crash rule |
| `config/districts.py` | Checked district targets for carrot and leeks |
| `agents/farmer.py` | Identical farmers. Each crop is its own market |
| `agents/population.py` | District farmers, fixed adopters, shared land |
| `market/clearing.py` | Price from planted hectares |
| `market/msrs.py` | Member 3 saturation score |
| `market/price_process.py` | Persistence plus a shock |
| `scenarios/baseline.py` | Separate-crop cobweb |
| `scenarios/advisory.py` | Carrot and leeks with a fixed adopter set |
| `calibration/compare.py` | One-step comparison with Colombo wholesale seasons |
| `calibration/baseline_path.py` | No-advisory check of the persistent price |
| `experiments/adoption.py` | Adoption rates at the reference prices, no shock |
| `inputs/harti.py` | Season means from `curated.crop_market_week` |
| `dashboard.py` | Streamlit view of one run, the adoption comparison, the wholesale check, and the districts |
| `tests/` | Bounds, calibration, district shares, and the price process |

`member3_MSRS_DCVS` supplies `calculate_pressure_risk`. This package calls that function. It does not copy the formula.

## Reference points

The warehouse has no quantity sold, so the supply and demand slopes are parameters. The reference point is the place where planted area equals the reference area and the cleared price equals the reference price.

| Crop | Reference area (ha) | Reference price (Rs/kg) | Where the area comes from |
| --- | --- | --- | --- |
| Carrot | 2001 | 149.09 | Maha 2024/25 final, report 2025-04-15. Price is the Colombo wholesale weekly median |
| Leeks | 1333 | 125.83 | 2025 Yala final, report 2025-10-15. Price is the Colombo wholesale weekly median |
| Tomato | 1333 | 120.00 | Area is an assumption, set equal to the leeks target so the separate-crop baseline can run. It is not a DOA figure. Price is the Colombo wholesale weekly median |

Twenty percent of the reference area responds to price at the reference price (`PRICE_RESPONSE_SHARE = 0.20`). The other eighty percent is the supply intercept, so the intercept stays positive.

```
supply slope (ha per Rs/kg) = 0.20 * reference area / reference price
area = intercept + supply slope * last price
demand slope (Rs/kg per ha) = slope ratio / supply slope
price = max(0, demand intercept - demand slope * area)
```

`slope ratio` is supply slope times demand slope. Below 1, a gap from the reference shrinks. At 1 it cycles. Above 1 it grows until the price or the area hits zero. The separate-crop baseline uses 0.8 as its default scenario. The HARTI comparison below does not support using a ratio above the bottom of the grid.

In the separate-crop baseline each farmer holds 10,000 ha, so the land cap does not bind. Carrot, leeks, and tomato do not compete for hectares in that run.

`python -m simulation.scenarios.baseline` starts every crop 20% above its reference price and prints carrot for slope ratios 0.5 and 1.5. Ratio 0.5 damps back toward 149.09. Ratio 1.5 grows, produces a crash, and then hits the zero floor.

## Crash

A crash is one season whose price is at least 25% below the previous season. The first season is never a crash. A later season that stays low is not a second crash unless it falls by 25% again. A week is not a crash. The 25% cut is the definition used by the experiments. It is not a fitted threshold.

## District farmers

Carrot and leeks share one stock of land. The land in a district is the sum of its checked carrot and leeks targets. National land is 3,334 ha. Spellings are the warehouse spellings.

| District | Carrot target (ha) | Leeks target (ha) |
| --- | --- | --- |
| Nuwaraeliya | 1224 | 1080 |
| Badulla | 614 | 190 |
| Kandy | 65 | 53 |
| Jaffna | 38 | 8 |
| Kilinochchi | 12 | |
| Ampara | 10 | |
| Vavuniya | 8 | |
| Trincomalee | 7 | |
| Mannar | 6 | |
| Mullathivu | 6 | |
| Ratnapura | 6 | 2 |
| Moneragala | 4 | |
| Kurunegala | 1 | |

A blank cell means that crop has no checked target in that district. The fair share is then zero. Planting a crop there still scores 1 if any hectares are committed.

Every district gets one farmer. Extra farmers are split by land share, largest remainder first. One hundred farmers is the default, and the count cannot be below 13, because there are 13 districts.

Adopters are chosen once. The same seed shuffles the farmer ids the same way and keeps the first `round(n * rate)` of them. A farmer who follows the score in season 1 follows it in every later season. Adoption is not redrawn.

At reference prices, a farmer's crop weights reproduce the district targets, so national area stays 2,001 ha of carrot and 1,333 ha of leeks. A higher relative price raises that crop's weight. An adopter multiplies a crop weight by `(1 - score)`. If every weight is zero, all of that farmer's land goes to the least saturated crop. A tie goes to the crop with the larger target.

The score seen at planting time is last season's commitments. Season 0 has no score yet.

## Member 3 score

`simulation/market/msrs.py` calls `member3.domain.calculate_pressure_risk` with Member 3's default `ScoringConfig`.

```
ratio = committed hectares / district target
score = clip((ratio - 0.80) / (1.20 - 0.80), 0, 1)
```

The score is 0 at 80% of the district target, 0.5 at the target, and 1 at 120% of the target. Committed hectares of 0 score 0. A zero target with a positive commitment scores 1.

Member 3's demo capacities (carrot 1,000 ha, leek 800 ha, tomato 900 ha) are not used. Their own readme marks those numbers as placeholders. The capacity here is the checked district target in the table above.

At the targets, both crops score 0.5. An adopter scales both weights by the same amount, so the crop mix does not move until a price shock pushes a district away from its target.

## Colombo wholesale comparison

`simulation/inputs/harti.py` reads `curated.crop_market_week` for Colombo wholesale. The season price is the mean of weeks that have a HARTI price and a single season label. A season with fewer than 15 priced weeks is left out. The current warehouse gives 18 seasons and 16 consecutive pairs for each crop. The interior gap is 2020/21 Maha, so 18 seasons do not produce 17 pairs.

The one-step comparison recenters on the mean of those season prices, not on the weekly median. Centers are carrot 196.65, leeks 141.93, and tomato 163.13 Rs/kg. It tries slope ratios from 0.05 to 2.00 in steps of 0.05 and scores each ratio by the root-mean-square error of the next season's price, in Rs/kg. Three rules are compared: flat (repeat the center), persist (repeat last season), and cobweb (the best ratio on the grid).

`python -m simulation.calibration.compare` prints:

| Crop | Seasons | Pairs | Crashes | Crash rate | Lag | Deseasoned lag | Change std | Best ratio | Flat RMSE | Persist RMSE | Cobweb RMSE | Closest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Carrot | 18 | 16 | 6 | 0.38 | 0.41 | 0.43 | 0.70 | 0.05 | 104.5 | 110.5 | 106.5 | flat |
| Leeks | 18 | 16 | 6 | 0.38 | 0.44 | 0.50 | 1.00 | 0.05 | 66.5 | 69.8 | 68.0 | flat |
| Tomato | 18 | 16 | 6 | 0.38 | 0.54 | 0.60 | 0.67 | 0.05 | 97.3 | 91.2 | 99.8 | persist |

Lag is the correlation of a season price with the previous season. Deseasoned lag removes the Maha and Yala means from the log price first. Both lags are positive. The sign-flipping cobweb is not the closest rule for any of the three crops. The best positive ratio is 0.05, which is the bottom of the grid. These numbers are from the current local warehouse. Re-run the command after a warehouse rebuild.

## Persistent price

The separate-crop cobweb at a ratio below 1 damps to the reference and then has no crashes. The season price used for the check is

```
price = max(0, persistence * last price + (1 - persistence) * cleared price + shock)
```

Persistence and the shock scale are an AR(1) fitted to the same Colombo wholesale season prices, around that crop's season-mean center. Persistence is clipped to the range 0 to 1. A negative estimate is not kept. The cleared price is the inverse-demand price at the planted area. When the area is the reference area, the cleared price is the season-mean center.

`python -m simulation.calibration.baseline_path` runs 40 seeds with no adopters. Carrot and leeks use the district farmers and slope ratio 0.05. Tomato has no district table, so its path is the price equation only. The printed result:

| Crop | Persistence | Shock std (Rs/kg) | Center | HARTI crash rate | Simulated crash rate | HARTI lag | Simulated lag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Carrot | 0.43 | 98.0 | 196.7 | 0.38 | 0.33 | 0.41 | 0.31 |
| Leeks | 0.45 | 61.6 | 141.9 | 0.38 | 0.30 | 0.44 | 0.30 |
| Tomato | 0.57 | 84.0 | 163.1 | 0.38 | 0.30 | 0.54 | 0.38 |

The simulated crash rate is lower than 0.38, and the simulated lag is lower than the HARTI lag. The shock was fitted on these same seasons, so this shows that the price rule can produce drops and a positive lag. It is not a separate confirmation. A slope ratio of 0.8 on this same path turns the lag negative, because planting then flips the next price. That ratio is not used in the check.

## Adoption experiment

`python -m simulation.experiments.adoption` sweeps adopter shares 0, 0.1, 0.3, 0.5, 0.7, 0.9, and 1.0. Ten seeds, 16 seasons, 100 farmers. It does not add the fitted shock. Prices start at the weekly-median references, both crops score 0.5, and the crop mix does not move. Crash rates print as 0.00. The command also prints the HARTI crash rate of 0.38 and says that a lower simulated rate is not evidence that the advisory prevents those crashes.

The same command prints a single 20% carrot price shock with no random shock. Season 0 cannot see a score, so every adoption rate plants the same carrot area. Season 1 is the first season that can use the score. Adopters plant less carrot than non-adopters in that season.

The dashboard runs the same adoption rates on the fitted persistent price, with the same shocks at every adoption rate. Compare a rate with the 0% row. The historical crash rate of 0.38 is drawn on that chart as context. It is not the control.

## Dashboard

`simulation/dashboard.py` has four tabs.

* **Season path.** One run from the sidebar: adopters, seed, seasons, farmers, and slope ratio. The price path is either the Colombo wholesale fit or the clearing price alone. A point is a crash.
* **Adoption.** Mean crash rate and mean planted area across the selected number of runs. The band is the lowest and highest run.
* **Colombo wholesale.** Ignores the sidebar. Season prices, the 40-run price check, and the one-step rule table.
* **Districts and score.** Checked targets, the Member 3 score curve, and which farmers are adopters for the sidebar seed.

Slope choices in the sidebar are 0.05, 0.5, 0.8, 1.0, and 1.5. The default is 0.05.

## Tests

The tests lock the behaviour above.

* Ratio 0.5 returns to the carrot reference. Ratio 1.5 grows.
* A pure two-season cycle selects slope ratio 1.0. A rising series selects the bottom of the grid, and persist is closer than the cobweb.
* Nuwaraeliya carrot is 1,224 ha, not an equal share of the national target.
* The Member 3 score is 0 at 80% of a target, 0.5 at the target, and 1 at 120%.
* Reference prices reproduce the district targets when nobody is an adopter.
* A 20% price shock makes season-0 area the same at every adoption rate, and season-1 area smaller when everyone adopts.
* Full persistence with no shock repeats last season's price. A negative fitted persistence is stored as 0.

The warehouse test in `test_calibration.py` is skipped when `cobweb.duckdb` is missing. It checks that the deseasoned lag is positive, the best ratio is 0.05, and the closest rules are flat for carrot and leeks and persist for tomato.

## What this simulation does not do

It does not write the warehouse, rotate the reader password, or read Neon. Calibration uses the local DuckDB file only.

It does not treat a lower crash rate among adopters as evidence about the historical HARTI crashes. The no-shock run has a crash rate of 0. The shocked run is quieter than Colombo wholesale, and its shock was fitted on those same seasons.

It does not raise the slope ratio to force a sign-flipping cobweb. The comparison says that rule is not the closest.

It does not use Member 3's demo capacities, and it does not invent tomato district targets, yield, or cost. Supply is in hectares because kilograms per hectare are not in the warehouse.

`member4_AgenticSimulation` is an earlier model. This package does not load it.
