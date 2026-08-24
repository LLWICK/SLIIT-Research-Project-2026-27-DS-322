# HTFE Studio

Frontend-only supervisor demo for Member 2 — Hybrid Temporal Forecasting Engine.

```bash
npm install
npm run dev
```

Open http://localhost:5173

All charts and metrics read JSON from `public/data/`, written by:

```bash
python forecasting-engine/scripts/run_demo_pipeline.py
```

There is no backend. The `/forecast` screen shows the frozen FastAPI contract Members 3 and 4 will call later.
