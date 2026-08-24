import { useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useStudio } from "@/entities/results/load";
import { Panel } from "@/shared/ui/Panel";
import { lkr, titleCase } from "@/shared/lib/format";

export function ForecastPage() {
  const { live } = useStudio();
  const crops = [...new Set(live.map((r) => r.crop))];
  const markets = [...new Set(live.map((r) => r.market))];
  const [crop, setCrop] = useState(crops[0] ?? "carrot");
  const [market, setMarket] = useState(markets.includes("colombo") ? "colombo" : markets[0]);
  const [intensity, setIntensity] = useState(0.85);

  const match = useMemo(() => {
    const pool = live.filter((r) => r.crop === crop && r.market === market);
    if (!pool.length) return null;
    return pool.reduce((best, row) =>
      Math.abs(row.cultivation_intensity - intensity) < Math.abs(best.cultivation_intensity - intensity)
        ? row
        : best,
    );
  }, [live, crop, market, intensity]);

  const request = {
    crop: crop.toUpperCase(),
    market: market.toUpperCase(),
    forecast_week: match?.forecast_week ?? "2025-12-22",
    current_commitment_hectares: Math.round(intensity * 1000),
  };

  return (
    <div className="space-y-6">
      <header>
        <p className="text-[11px] uppercase tracking-[0.22em] text-harvest">Frozen contract · no backend this demo</p>
        <h2 className="font-display mt-2 text-4xl">POST /forecast → interval</h2>
        <p className="mt-3 max-w-3xl text-mist">
          Lecturer asked for a frontend-only prototype. The contract is still the FastAPI shape Members 3 and 4
          will call. Drag the commitment slider: the interval updates from the trained LightGBM + CQR grid.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel kicker="Request" title="What Member 3 sends">
          <div className="mb-4 grid gap-3 sm:grid-cols-2">
            <Field label="Crop">
              <select className="field" value={crop} onChange={(e) => setCrop(e.target.value)}>
                {crops.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </Field>
            <Field label="Market">
              <select className="field" value={market} onChange={(e) => setMarket(e.target.value)}>
                {markets.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            </Field>
          </div>
          <label className="block text-sm text-mist">
            Cultivation intensity {intensity.toFixed(2)}
            <input
              type="range"
              min={0.25}
              max={1.5}
              step={0.05}
              value={intensity}
              onChange={(e) => setIntensity(Number(e.target.value))}
              className="mt-2 w-full accent-harvest"
            />
          </label>
          <pre className="mt-4 overflow-x-auto rounded-2xl bg-ink p-4 text-xs text-sky">
            {JSON.stringify(request, null, 2)}
          </pre>
        </Panel>

        <Panel kicker="Response" title="What the engine returns">
          <AnimatePresence mode="wait">
            {match && (
              <motion.div
                key={`${match.crop}-${match.market}-${match.cultivation_intensity}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
              >
                <p className="text-sm text-mist">
                  {titleCase(match.crop)} at {titleCase(match.market)} · week {match.forecast_week}
                </p>
                <div className="mt-5 grid grid-cols-3 gap-3">
                  <Price label="Lower" value={match.lower_price} />
                  <Price label="Point" value={match.predicted_price} accent />
                  <Price label="Upper" value={match.upper_price} />
                </div>
                <p className="mt-4 text-sm text-mist">
                  Coverage {Math.round(match.coverage_level * 100)}% · source {match.commitment_source} ·{" "}
                  {match.model_version}
                </p>
                <pre className="mt-4 overflow-x-auto rounded-2xl bg-ink p-4 text-xs text-sky">
                  {JSON.stringify(
                    {
                      crop: request.crop,
                      market: request.market,
                      forecast_week: match.forecast_week,
                      predicted_price: match.predicted_price,
                      lower_price: match.lower_price,
                      upper_price: match.upper_price,
                      coverage_level: match.coverage_level,
                      cultivation_intensity: match.cultivation_intensity,
                      commitment_source: match.commitment_source,
                      model_version: match.model_version,
                    },
                    null,
                    2,
                  )}
                </pre>
              </motion.div>
            )}
          </AnimatePresence>
        </Panel>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-sm text-mist">
      {label}
      <div className="mt-1">{children}</div>
    </label>
  );
}

function Price({ label, value, accent }: { label: string; value: number; accent?: boolean }) {
  return (
    <div className={`rounded-2xl border px-3 py-4 ${accent ? "border-harvest bg-harvest/10" : "border-line bg-ink-2"}`}>
      <p className="text-[11px] uppercase tracking-[0.16em] text-mist">{label}</p>
      <p className={`font-display mt-1 text-2xl ${accent ? "text-harvest" : "text-cream"}`}>{lkr(value)}</p>
    </div>
  );
}
