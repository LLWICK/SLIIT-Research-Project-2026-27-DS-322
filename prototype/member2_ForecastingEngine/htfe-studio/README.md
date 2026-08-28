# Hybrid Temporal Forecasting Engine — Member 2

Streamlit studio for Garusingarachchi Y.B (IT23415836). Same research screens as
the earlier React demo, packaged the same way as Member 4: one app under `prototype/`.

The header switches the research chain **Member 1 → 2 → 3 → 4**. Only Member 2
is implemented here. Clicking Member 3 or 4 opens **their existing Streamlit
apps as-is**. Those files are not edited and their UIs are not rebuilt.

Member 2 does **not** invent the Market Saturation Risk Score. It produces a
calibrated weekly price interval that Member 3 consumes, and that Member 4 can
reuse in batch.

## Chain

| Step | Owner | What this app does |
|---|---|---|
| 1 → 2 | Member 1 extract | Reads `data/v1/analysis_ready_price.csv` when present and reports coverage. Charts stay on frozen scored artifacts so metrics do not drift. |
| 2 | This studio | LightGBM quantile + CQR interval, intensity-aware re-forecast. |
| 2 → 3 | Header switch | Opens `prototype/member3_MSRS_DCVS/app.py` unchanged. Dashboard still writes `outputs/forecast_packet.json` for when they wire the API. |
| 3 → 4 | Header switch | Opens `prototype/member4_AgenticSimulation/veggie-abm/app.py` unchanged. |

## Setup

```bash
cd prototype/member2_ForecastingEngine/htfe-studio
pip install -r requirements.txt
```

## Run it

```bash
streamlit run app.py
```

Opens at http://localhost:8501

Member 3 and Member 4 are unchanged:

```bash
streamlit run prototype/member3_MSRS_DCVS/app.py
streamlit run prototype/member4_AgenticSimulation/veggie-abm/app.py
```

## Screens

Header: **01 Data Fusion · 02 Price Forecast · 03 Crop Viability · 04 Season Simulation**

Inside Price Forecast (sidebar):

- **Dashboard** — crop, market, planting-pressure slider, 90% interval, Member 3 packet
- **Results** — four models, same test weeks
- **Data** — panel split, series explorer, cobweb scatter, data rules
- **Features** — origin map, intensity formula, permutation importance
- **Training** — replay of the recorded four-model run
- **Comparison** — baseline vs LightGBM / XGBoost / LSTM-style
- **Uncertainty** — CQR fan chart on hold-out weeks
- **Novelty** — A/B/C ablation and week 2/5/8/12 re-forecast

## Honest status

Artifacts in `data/` are the scored demo registry (seed 42). This app does not
retrain LightGBM. Commitments are labelled **simulated** until a live HARTI
registration feed exists. LSTM-style MAE can look better; it is not shipped
because PICP falls to about 70%.
