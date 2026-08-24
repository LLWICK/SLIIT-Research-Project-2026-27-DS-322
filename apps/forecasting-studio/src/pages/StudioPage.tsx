import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useStudio } from "@/entities/results/load";
import { Panel } from "@/shared/ui/Panel";

const STEPS = [
  { id: "sarima", title: "Seasonal baseline", detail: "52-week seasonal naive / Holt-Winters family. Locked protocol, price only." },
  { id: "lightgbm", title: "LightGBM quantiles", detail: "objective=quantile at 0.05 / 0.50 / 0.95, then chronological CQR." },
  { id: "xgboost", title: "XGBoost quantiles", detail: "reg:quantileerror + tree_method=hist. Same folds, same seed." },
  { id: "lstm", title: "LSTM-style MLP", detail: "Optional stretch. Same features; no conformal wrap in this prototype." },
] as const;

export function StudioPage() {
  const { models, comparison, meta } = useStudio();
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(-1);
  const [cursor, setCursor] = useState(0);

  const active = step >= 0 ? STEPS[Math.min(step, STEPS.length - 1)] : null;
  const curve = useMemo(() => {
    if (!active) return [];
    const raw = models[active.id]?.train_curve ?? [];
    return raw
      .map((row, i) => ({
        iter: Number(row.iter ?? row.epoch ?? i + 1),
        mae: row.mae == null ? null : Number(row.mae),
      }))
      .filter((r) => r.mae != null);
  }, [active, models]);

  useEffect(() => {
    if (!running) return;
    setStep(0);
    setCursor(1);
    const timer = window.setInterval(() => {
      setCursor((c) => c + 1);
    }, 90);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (!running || !active) return;
    const limit = Math.max(curve.length, 8);
    if (cursor >= limit) {
      if (step < STEPS.length - 1) {
        setStep((s) => s + 1);
        setCursor(1);
      } else {
        setRunning(false);
      }
    }
  }, [cursor, curve.length, running, step, active]);

  const visibleCurve = curve.slice(0, Math.max(cursor, 1));
  const finished = step >= STEPS.length - 1 && !running && step >= 0;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.22em] text-harvest">Azure-style experiment console</p>
          <h2 className="font-display mt-2 text-4xl">Watch the registry fill</h2>
          <p className="mt-3 max-w-2xl text-mist">
            This is a replay of the real local / Azure-ready training run (seed {meta.seed}, {meta.elapsed_s}s wall
            time). Metrics are not animated fiction — they come from <code>results/demo</code>.
          </p>
        </div>
        <button
          onClick={() => {
            setRunning(true);
            setStep(-1);
            setCursor(0);
          }}
          className="rounded-full bg-harvest px-5 py-2 text-sm font-medium text-ink"
        >
          Replay training
        </button>
      </header>

      <div className="grid gap-3 md:grid-cols-4">
        {STEPS.map((item, i) => {
          const row = comparison.find((c) => c.id === item.id);
          const state = step > i ? "done" : step === i ? "live" : "idle";
          return (
            <div
              key={item.id}
              className={`rounded-2xl border p-4 ${
                state === "live" ? "border-harvest bg-harvest/10" : "border-line bg-panel"
              }`}
            >
              <p className="text-[11px] uppercase tracking-[0.16em] text-mist">{state}</p>
              <p className="font-display mt-1 text-lg">{item.title}</p>
              <p className="mt-2 text-xs text-mist">{item.detail}</p>
              {row && state !== "idle" && (
                <p className="mt-3 text-sm text-cream">
                  MAE {row.mae} · PICP {row.picp}%
                </p>
              )}
            </div>
          );
        })}
      </div>

      <Panel kicker="Live console" title={active ? active.title : "Waiting for a run"}>
        <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="h-64 rounded-2xl border border-line bg-ink/50 p-2">
            {visibleCurve.length > 1 ? (
              <ResponsiveContainer>
                <LineChart data={visibleCurve}>
                  <XAxis dataKey="iter" />
                  <YAxis />
                  <Tooltip contentStyle={{ background: "#10211b", border: "1px solid #1d3a30" }} />
                  <Line type="monotone" dataKey="mae" stroke="#c6f31a" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="grid h-full place-items-center text-sm text-mist">
                Press replay to stream the recorded training curve.
              </div>
            )}
          </div>
          <div className="space-y-2 font-mono text-xs">
            <AnimatePresence>
              {STEPS.slice(0, Math.max(step + 1, 0)).map((item) => {
                const m = models[item.id];
                return (
                  <motion.p
                    key={item.id}
                    initial={{ opacity: 0, x: 8 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="rounded-xl border border-line bg-ink-2 px-3 py-2 text-sky"
                  >
                    [{item.id}] backend={m?.backend} qhat={m?.qhat ?? "—"} MAE={m?.metrics.overall.mae} PICP=
                    {m?.metrics.overall.picp}
                  </motion.p>
                );
              })}
            </AnimatePresence>
            {finished && (
              <p className="rounded-xl border border-harvest/40 bg-harvest/10 px-3 py-2 text-harvest">
                registry flushed · artifacts/demo promoted for the Studio UI
              </p>
            )}
          </div>
        </div>
      </Panel>

      <Panel kicker="Why LSTM can look better on MAE" title="Do not ship an uncalibrated winner">
        <p className="text-sm leading-relaxed text-mist">
          The optional MLP prototype can post a lower point MAE, but its PICP is only{" "}
          {comparison.find((r) => r.id === "lstm")?.picp}% with a much tighter band. That is a failed uncertainty
          product. LightGBM + CQR is the primary model because coverage stays near 90% and the interval stays usable.
        </p>
      </Panel>
    </div>
  );
}
