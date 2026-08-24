# Supervisor demo script — Member 2 (IT23415836)

**App:** http://localhost:5173  
**Time:** about 8–10 minutes  
**You:** Garusingarachchi Y.B · Hybrid Temporal Forecasting Engine

Start on **The system** (home). That is the group product view. Research pages are evidence, not the story. Click only when the script says **CLICK**.

---

## 0. Before they sit down (30 seconds)

```bash
cd apps/forecasting-studio
npm run dev
```

Open Chrome at **http://localhost:5173**. Full screen. This file on a second screen or phone.

If the page is blank: wait 5 seconds. If still blank, refresh once.

---

## Numbers to say without looking

| What they ask | What you say |
|---|---|
| Beat the baseline? | Yes. MAE **48.82** vs **126.21** — about **61%** lower |
| Is the interval calibrated? | Empirical PICP **89.2%** against a **90%** target |
| How wide? | Mean width **Rs. 203** — usable, not 20 to 2000 |
| How did you split? | Temporal only. Train to mid-2023, then calibration, test **2024–2025**. **1484** observed weeks |
| LSTM? | Optional. Lower MAE (**44**) but PICP only **70%**. That is why it is not the product |
| Commitments real? | **Labelled simulation** until HARTI registrations arrive |
| Dambulla? | Not in this extract. Colombo is the consumer-market series |

---

## 1. The system — 1.5 minutes (start here)

**Say:**

> This is one decision-support system, four members.  
> Member 1 cleans the data.  
> **I** am Member 2 — the forecasting engine. I do **not** create the Market Saturation Risk Score. Member 3 does.  
> I create the **price interval** they need: lower, point, upper.  
> Member 4 calls the same thing in batch for the simulation.

Point at the four chain cards. Yours is the highlighted one.

**DRAG** the intensity slider.

> When planting commitments change, my interval updates. That packet is what will appear on Member 3’s dashboard — the same idea as their upload appearing on the parent screen. Their MSRS box stays empty here on purpose. I will not invent their score.

**CLICK** My evidence only if they ask “how do we know this number is real?” Otherwise stay on the system, then jump to Comparison if short on time.

**If they want the lab:** **CLICK** Data.

---

## 2. Data — 1.5 minutes

Point at the three split boxes.

> Time is cut in order. Never random. Train, then a calibration window for conformal prediction, then test from January 2024. Panels ask this. The answer is one chronological splitter.

**CLICK** Carrot · Colombo. Then Leek · Badulla if there is time.

> These are real HARTI wholesale weekly prices. You can see the boom–bust cycle. That is the cobweb we are trying to warn farmers about **before** harvest.

Point at the yellow scatter (“Money plot”).

> X is planting intensity this season. Y is the next season’s price. This is the cobweb in **our** panel, not only in Ezekiel 1938.

Point at Exclusions.

> Local Big Onion is out — too incomplete in this source, and imported onion is a different mechanism.  
> Dambulla is a **market**, and it is not in this file. I did not invent it. Colombo is the consumer-market series. Badulla and Nuwara Eliya are the high-coverage origin-adjacent series.

**CLICK** Features.

---

## 3. Features — 1 minute

Point at the origin map.

> Prices are observed at **markets**. Crops are grown in **districts**. Weather and commitments never join on the market name. Colombo carrot weather comes from Nuwara Eliya, Badulla and Matale.

Point at the formula.

> Intensity is cumulative hectares registered in those supply districts **up to this week**, over a demand benchmark. A registration after the forecast week cannot enter. That is the as-of-safety rule.

Point at the importance bars.

> Last week’s price dominates, as expected. Cultivation intensity is in the **top four** features. That is supporting evidence. The proof is the next two experiments, not this chart alone.

**CLICK** ML Studio.

---

## 4. ML Studio — 1.5 minutes

**CLICK** Replay training.

While it runs:

> This is a replay of the real training run — same seed, same folds. Four models, one protocol.  
> First, a 52-week seasonal baseline — the number every later model is judged against.  
> Then LightGBM quantiles at 5%, 50%, 95%, wrapped with chronological CQR. That is the primary model.  
> Then XGBoost the same way, because Moratuwa also used boosting with Optuna.  
> Last, an optional LSTM-style network. The TAF said hybrid comparison. It is not required for the grade if the core holds.

When the last card finishes:

> Watch this. The neural net can look **better on MAE**. Its PICP is only about **70%**, and the band is too tight. That is a failed uncertainty product. We do not ship it. LightGBM plus CQR is the engine because coverage stays near 90% and the width is still usable.

**CLICK** Comparison.

---

## 5. Comparison — 1 minute

Point at the LightGBM row.

> Same 1484 observed test weeks for every model.  
> Baseline MAE **126**, MAPE **64%**, PICP **78%**, width **346**.  
> LightGBM MAE **49**, MAPE **20%**, PICP **89%**, width **203**.  
> XGBoost is close but a bit worse and a wider band.  
> So yes — we beat a simple baseline, and we did it with a measured interval, not a point guess.

Point at the crop slice if asked.

> Leek is easiest — MAE about **28**, PICP **96%**. Carrot is hardest — MAE about **69**. That is honest. High-volatility crops are the ones the cobweb hurts.

**CLICK** Uncertainty.

---

## 6. Uncertainty — 1 minute

**Say this sentence exactly:**

> We **target** a 90% calibrated interval and **evaluate** empirical coverage on chronologically held-out data.

Do **not** say “CQR mathematically guarantees 90%” unless they ask, and then add: weekly farm prices are not exchangeable, so we report empirical PICP plus width.

**CLICK** a crop–market pill (Carrot · Colombo).

> Amber is the real wholesale price. Lime is the median forecast. The band is the calibrated interval. Most realised prices sit inside it. PICP **89.2%**. Mean width **Rs. 203**. The correction q-hat is about **38** rupees added to both tails after calibration.

**CLICK** Novelty.

---

## 7. Novelty — 1.5 minutes

This is your individual mark. Slow down.

> Two experiments. Same LightGBM, same weeks, same seed. Only one thing changes at a time.

Point at A / B / C.

> **A** is lags and season — the historical baseline.  
> **B** adds origin weather.  
> **C** adds live cultivation intensity.  
> On this extract, B is slightly best on MAE — **48.57** versus C at **48.77**. I am not hiding that. A near-tie is still a result. C keeps PICP near 90%, and intensity ranks in the top features.

Point at week 2 / 5 / 8 / 12.

> This is the operational proof. We re-forecast the **same** harvest four times. Only the commitments known by that week are allowed in.  
> Week 2 MAE **49.64**. Week 12 MAE **49.10**. Error falls as plantings register — **before** harvest, not after the crash.  
> The commitment stream is labelled **simulated**. The function signature is the same as a future HARTI feed.

**CLICK** Live forecast.

---

## 8. Handoff contract — 1 minute

> Same packet as the home screen, in API shape. Left is what Member 3 sends. Middle is what I return. Right is their inbox — MSRS stays a placeholder until their engine is wired.

**CLICK** crop = carrot, market = colombo (or leave defaults).

**DRAG** the intensity slider slowly from low to high.

> When I raise cultivation intensity, I am telling the engine: more hectares are already committed in the supply districts. The interval updates. That is mid-season re-forecast. Member 3’s viability score needs this range. A single number would break them.

Point at `commitment_source: simulated`.

> The label travels with the result. We never call this live farmer data.

Stop. Ask if they want questions.

---

## If they attack — answer in one breath

**“A third of your weeks are missing.”**  
This extract is analysis-ready observed rows. Every metric is still scored on observed prices only. We never fill missing prices with zero.

**“You treated Dambulla as a growing district.”**  
I did not. Dambulla is a market and it is not in this file. Weather and commitments come from origin districts through the map on the Features page.

**“Your TAF named Big Onion.”**  
The TAF named it as a high-risk example, not a data contract. This workbook cannot support it. Building a headline on a 90% empty series would be worse.

**“Simulated commitments make the novelty fake.”**  
The stream is simulated and labelled. The mechanism is real and tested: as-of-week, A/B/C, and week 2/5/8/12. Swapping in HARTI registrations changes one table, not the pipeline.

**“Why not deep learning as the main model?”**  
This is a few thousand tabular weekly rows. Boosting is the right family. We still ran an LSTM-style comparison. It lost on calibration.

**“Random train/test?”**  
Never. One temporal split, frozen, train then calibration then test.

**“Is the interval guaranteed?”**  
We target 90% and measure 89.2% on later years. We do not claim a universal i.i.d. guarantee.

**“What does a farmer actually get?”**  
Not “carrot will be 240”. They get “90% likely between about 180 and 305”, and that band updates as neighbours register plantings.

---

## Close (15 seconds)

> So the prototype already shows the three things the plan said decide the grade: an interval not a number, a re-forecast when commitments change, and every claim backed by a table in this Studio — not a promise.

---

## If something breaks live

| Problem | Fix |
|---|---|
| Page will not open | `cd apps/forecasting-studio` then `npm run dev` |
| “Could not load demo artifacts” | Refresh. JSON lives in `public/data/` — do not delete it |
| Replay looks stuck | Wait 10 seconds; it is a recorded curve, then it jumps models |
| They ask for Azure / backend | “Training can run as the same Python job on Azure ML. API is frozen on this screen; backend is next after this checkpoint.” |
