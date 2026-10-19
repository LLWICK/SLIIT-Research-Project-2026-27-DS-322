# Member 3 - Crowding-Aware Crop Recommendation Prototype

This early-stage prototype demonstrates the required commitment workflow without
claiming a validated market-oversupply model.

## Scope

- Reads forecasts only from `Member3Data/member3_forecast_c.json`.
- Stores only confirmed commitments in SQLite.
- Separates `POST /recommendations/assess` from `POST /commitments`.
- Maps mock agro-climatic parameters (elevation, temperature, rainfall, soil type) to planting districts for suitability calculation.
- Evaluates Market Saturation Risk (MSRS) to warn of Cobweb price crashes.
- Computes and compares 5 recommendation paradigms:
  1. **Baseline DCVS**: Linear weighted multi-attribute score.
  2. **MCDM / TOPSIS**: Multi-criteria ranking minimizing distance to ideal solution while penalizing price forecast uncertainty intervals.
  3. **Non-Linear BPR & Crop Profile Similarity**: Traffic-routing congestion curves combined with operational feature vector matching.
  4. **Markowitz Portfolio Land-Splitting**: Optimal acreage division between primary and hedging crops.
  5. **Softmax Load Balancing**: Game-theoretic probabilistic allocation to prevent the Herding Effect.

## Important research limitations

`reference_capacity_ha` is a visible demo placeholder, not HARTI-validated demand.
MSRS/DCVS weights and pressure thresholds are experimental configuration. The response
labels this clearly. Replace these controls only after demand estimates and historical
validation are available.

Member 2 forecasts are market-level. This prototype asks for both district and market;
a validated district-to-market mapping is still required before deployment.

## Run

```bash
cd prototype/member3_MSRS_DCVS
python -m pip install -r requirements.txt

# Start FastAPI backend
uvicorn member3.api:app --reload

# Start Streamlit interactive comparison dashboard (in a separate terminal)
streamlit run app.py
```

Open `http://127.0.0.1:8000/docs` to test the API or `http://localhost:8501` to view the Streamlit UI. Run tests with:

```bash
PYTHONPATH=. pytest -v
```
