# Member 3 - Crowding-Aware Crop Recommendation Prototype

This early-stage prototype demonstrates the required commitment workflow without
claiming a validated market-oversupply model.

## Scope

- Reads forecasts only from `Member3Data/member3_forecast_c.json`.
- Stores only confirmed commitments in SQLite.
- Separates `POST /recommendations/assess` from `POST /commitments`.
- Recomputes cultivation-pressure risk from current confirmed commitments for every assessment.

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
uvicorn member3.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to demonstrate the API. Run tests with:

```bash
PYTHONPATH=. pytest -q
```
