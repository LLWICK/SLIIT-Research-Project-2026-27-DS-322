import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { useLiveMatch } from "@/features/handoff/useLiveMatch";
import { Fact } from "@/shared/ui/Fact";
import { Panel } from "@/shared/ui/Panel";
import { lkr, titleCase } from "@/shared/lib/format";

const STAGES = [
  { step: "01", title: "Data fusion", detail: "Weekly panel and reliability" },
  { step: "02", title: "Price forecast", detail: "Calibrated interval for the week", active: true },
  { step: "03", title: "Crop viability", detail: "Uses this forecast with weather" },
  { step: "04", title: "Season simulation", detail: "Tests advice at market scale" },
];

export function SystemPage() {
  const { crops, markets, crop, setCrop, market, setMarket, intensity, setIntensity, match } = useLiveMatch();

  return (
    <div className="space-y-8">
      <header>
        <p className="text-[11px] uppercase tracking-[0.22em] text-harvest">Cobweb decision support</p>
        <h2 className="font-display mt-2 max-w-4xl text-4xl leading-tight md:text-5xl">
          Weekly vegetable price forecast with a confidence range
        </h2>
        <p className="mt-4 max-w-3xl text-lg text-mist">
          Choose a crop and market, then set how much land is already committed this season. The engine returns a
          likely price and a 90% range that the next stage uses for planting advice.
        </p>
      </header>

      <div className="grid gap-3 md:grid-cols-4">
        {STAGES.map((stage, i) => (
          <motion.div
            key={stage.title}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06 }}
            className={`rounded-2xl border p-4 ${
              stage.active ? "border-harvest bg-harvest/10" : "border-line bg-panel"
            }`}
          >
            <p className="text-[11px] uppercase tracking-[0.16em] text-mist">{stage.step}</p>
            <p className="font-display mt-1 text-lg">{stage.title}</p>
            <p className="mt-2 text-sm text-mist">{stage.detail}</p>
          </motion.div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[0.85fr_1.2fr_0.95fr]">
        <Panel kicker="Inputs" title="Market and planting pressure">
          <label className="block text-sm text-mist">
            Crop
            <select className="field mt-1 capitalize" value={crop} onChange={(e) => setCrop(e.target.value)}>
              {crops.map((c) => (
                <option key={c} value={c}>
                  {titleCase(c)}
                </option>
              ))}
            </select>
          </label>
          <label className="mt-3 block text-sm text-mist">
            Market
            <select className="field mt-1 capitalize" value={market} onChange={(e) => setMarket(e.target.value)}>
              {markets.map((m) => (
                <option key={m} value={m}>
                  {titleCase(m)}
                </option>
              ))}
            </select>
          </label>
          <label className="mt-4 block text-sm text-mist">
            Land already committed this season
            <input
              type="range"
              min={0.25}
              max={1.5}
              step={0.05}
              value={intensity}
              onChange={(e) => setIntensity(Number(e.target.value))}
              className="mt-3 w-full accent-harvest"
            />
            <span className="mt-2 flex justify-between text-xs">
              <span>Low</span>
              <span className="text-cream">{Math.round(intensity * 100)}% of benchmark</span>
              <span>Crowded</span>
            </span>
          </label>
        </Panel>

        <Panel kicker="Forecast" title="Expected wholesale price">
          <AnimatePresence mode="wait">
            {match && (
              <motion.div
                key={`${match.crop}-${match.market}-${match.cultivation_intensity}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <p className="text-sm text-mist">
                  {titleCase(match.crop)} · {titleCase(match.market)} · week of {match.forecast_week}
                </p>
                <div className="mt-5 grid grid-cols-3 gap-2">
                  <Price label="Low" value={match.lower_price} />
                  <Price label="Expected" value={match.predicted_price} accent />
                  <Price label="High" value={match.upper_price} />
                </div>
                <RangeBar low={match.lower_price} mid={match.predicted_price} high={match.upper_price} />
                <div className="mt-4">
                  <Fact label="Confidence target" value="90%" />
                  <Fact label="Planting pressure used" value={`${Math.round(intensity * 100)}%`} />
                  <Fact label="Commitment source" value={titleCase(match.commitment_source)} />
                </div>
                <p className="mt-3 text-sm text-mist">
                  More land already committed usually eases the expected price — extra supply arrives later.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </Panel>

        <Panel kicker="Next stage" title="Crop viability">
          {match ? (
            <>
              <p className="text-sm text-mist">This forecast is ready for the viability component.</p>
              <div className="mt-4">
                <Fact label="Crop" value={titleCase(match.crop)} />
                <Fact label="Market" value={titleCase(match.market)} />
                <Fact label="Expected price" value={lkr(match.predicted_price)} />
                <Fact label="Range" value={`${lkr(match.lower_price)} – ${lkr(match.upper_price)}`} />
                <Fact label="Status" value="Ready" />
              </div>
              <p className="mt-4 text-sm text-mist">
                Viability scoring combines this range with weather. Season simulation can reuse the same forecast
                for many farmers.
              </p>
            </>
          ) : (
            <p className="text-sm text-mist">Select a crop and market to pass a forecast forward.</p>
          )}
        </Panel>
      </div>

      <Panel kicker="How this number was produced" title="Open the working notes">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["/evidence", "Results", "Accuracy and coverage"],
            ["/studio", "Training", "How the models were fit"],
            ["/compare", "Comparison", "Baseline against the engine"],
            ["/uncertainty", "Uncertainty", "How the 90% range is checked"],
          ].map(([to, title, hint]) => (
            <Link key={to} to={to} className="rounded-2xl border border-line bg-ink-2/50 p-4 transition hover:border-harvest/50">
              <p className="font-display text-lg">{title}</p>
              <p className="mt-1 text-sm text-mist">{hint}</p>
            </Link>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function Price({ label, value, accent }: { label: string; value: number; accent?: boolean }) {
  return (
    <div className={`rounded-2xl border px-2 py-3 ${accent ? "border-harvest bg-harvest/10" : "border-line bg-ink-2"}`}>
      <p className="text-[10px] uppercase tracking-[0.14em] text-mist">{label}</p>
      <p className={`font-display mt-1 text-xl ${accent ? "text-harvest" : "text-cream"}`}>{lkr(value)}</p>
    </div>
  );
}

function RangeBar({ low, mid, high }: { low: number; mid: number; high: number }) {
  const span = Math.max(high - low, 1);
  const pos = Math.min(100, Math.max(0, ((mid - low) / span) * 100));
  return (
    <div className="mt-5">
      <div className="relative h-2 rounded-full bg-line">
        <div className="absolute inset-y-0 left-0 rounded-full bg-harvest/40" style={{ width: "100%" }} />
        <div className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-harvest" style={{ left: `${pos}%` }} />
      </div>
      <div className="mt-2 flex justify-between text-xs text-mist">
        <span>90% likely range</span>
        <span>Expected sits inside the band</span>
      </div>
    </div>
  );
}
