# Multi-Agent Simulation — Wickramaarachchi's Sub-Objective

Mesa 3.x agent-based model for the counterfactual MSRS-adoption experiment
(TAF section 9). Built and tested end-to-end against dummy data — swap in
real data from Thamel's `data/exports/` when ready, same column shapes.

## Setup
```bash
pip install -r requirements.txt
python data/generate_dummy_data.py    # optional — only needed if you want the standalone dummy dataset
```

## Run it
```bash
streamlit run app.py                         # launches the interactive Sri Lankan visual simulator & map
python experiments/counterfactual_sweep.py   # runs the full MSRS on/off x adoption-rate sweep
python experiments/visualize_results.py       # plots crash frequency & volatility vs adoption rate
python experiments/sensitivity_analysis.py    # SALib Sobol sensitivity on key parameters
```


## What's actually built and tested (Mesa 3.5.1 — verified against the real
current API, not the older `mesa.time.RandomActivation` style)
- `model/agent.py` — `FarmerAgent`: naive (cobweb) vs MSRS-informed decision modes
- `model/cobweb_model.py` — season loop, price formation from aggregate supply, MSRS computation
- `experiments/counterfactual_sweep.py` — adoption-rate sweep with replications, runs and produces real CSV output
- `experiments/visualize_results.py` — plots, tested, produces `outputs/counterfactual_plot.png`
- `experiments/sensitivity_analysis.py` — SALib Sobol analysis, tested, produces real indices

## HONEST STATUS — read before presenting results anywhere

The mechanical pipeline works end-to-end: agents make decisions, MSRS is a
real signal that varies meaningfully season to season, the experiment
harness runs the full sweep and produces output. **But it is not yet
calibrated to show the expected effect.**

Two real findings from testing, worth including in your report as genuine
model-development observations rather than hiding:

1. **MSRS sensitivity bug found and fixed live**: the original saturation
   formula only triggered when a district's commitments exceeded its ENTIRE
   fair share by 100%+ — essentially never happened with realistic farmer
   counts, so MSRS sat near 0 and had no effect regardless of adoption rate.
   Fixed by rescaling so a 30% overshoot maps to MSRS=1
   (`SATURATION_SENSITIVITY` in `cobweb_model.py`) — tune this against real
   data once available.

2. **Volatility currently INCREASES with adoption rate, not decreases** —
   the opposite of the intended effect. Likely cause: every MSRS-aware
   agent reacts identically and simultaneously to the same signal, so they
   dampen together and overcorrect — a herding effect that amplifies
   oscillation instead of smoothing it. Confirmed via SALib: `price_elasticity`
   (S1=0.212) dominates volatility far more than `adoption_rate` (S1=0.002)
   at current settings.

## Next steps to actually calibrate this (in priority order)
1. Add per-agent heterogeneity in MSRS *sensitivity*, not just binary
   adopt/don't-adopt — right now every adopting agent dampens by the same
   fixed 0.8 factor (`FarmerAgent._decide_with_msrs`), which is what causes
   the herding effect. Give each agent its own dampening strength.
2. Smooth the MSRS signal (e.g. exponential moving average across seasons)
   instead of a raw last-season snapshot, so agents aren't all reacting to
   the same sharp spike simultaneously.
3. Once Thamel's real price data is available, validate crash frequency/timing
   against ACTUAL historical crashes (Big Onion, Carrot, Leek) instead of an
   arbitrary -25% threshold — this is also what your TAF's validation step
   (section 9) calls for.
4. Wire `dampening_strength` into `CobwebModel` as a real constructor
   parameter (currently a TODO in `sensitivity_analysis.py`) so the
   sensitivity analysis can actually test it.
5. Re-run `sensitivity_analysis.py` with a larger `n_samples` (64 was just
   to confirm it runs) for final, citable results.
