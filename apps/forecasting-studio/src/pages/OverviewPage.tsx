import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useStudio } from "@/entities/results/load";
import { Metric, Panel } from "@/shared/ui/Panel";
import { pct } from "@/shared/lib/format";

export function OverviewPage() {
  const { meta, comparison, split } = useStudio();
  const lgb = comparison.find((r) => r.id === "lightgbm")!;
  const sarima = comparison.find((r) => r.id === "sarima")!;
  const beat = (((sarima.mae - lgb.mae) / sarima.mae) * 100).toFixed(0);

  return (
    <div className="space-y-8">
      <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
        <p className="text-[11px] uppercase tracking-[0.24em] text-harvest">Supervisor prototype · Member 2</p>
        <h2 className="font-display mt-3 max-w-4xl text-4xl leading-tight md:text-6xl">
          Weekly vegetable prices with a calibrated interval — not a single number.
        </h2>
        <p className="mt-5 max-w-3xl text-lg text-mist">
          Existing Sri Lankan forecasters, including Moratuwa 2026, emit a point estimate from static features.
          This engine outputs <span className="text-cream">(lower, point, upper)</span> and re-forecasts as
          within-season cultivation commitments accumulate in each market’s supply districts.
        </p>
      </motion.section>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="LightGBM MAE" value={`Rs. ${lgb.mae}`} hint={`SARIMA was Rs. ${sarima.mae} · ${beat}% lower`} />
        <Metric label="Empirical PICP" value={pct(lgb.picp)} hint="Target 90% on chronological holdout" />
        <Metric label="Interval width" value={`Rs. ${lgb.interval_width}`} hint="Always report with PICP" />
        <Metric label="Test weeks scored" value={String(meta.n_scored_test)} hint={`${split.test.start} → ${split.test.end}`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel kicker="Approved TAF" title="What this component owns">
          <p className="text-sm leading-relaxed text-mist">
            forecast(crop, market, week, commitment) → (point, lower, upper). Member 3’s DCVS needs the range.
            Member 4 needs the same contract in batch. Grain is crop × market × week.
          </p>
        </Panel>
        <Panel kicker="Honesty" title="What this file is">
          <p className="text-sm leading-relaxed text-mist">
            Good price dataset, incomplete novelty dataset. Commitments in this demo are a labelled historical
            simulation (<code>synthetic_proxy</code>), not live HARTI registrations. Dambulla is not in the extract;
            Colombo is the consumer-market series.
          </p>
        </Panel>
        <Panel kicker="Locked protocol" title="How numbers are earned">
          <ul className="space-y-2 text-sm text-mist">
            <li>Temporal split only — never random.</li>
            <li>Observed wholesale prices only — never zero-filled.</li>
            <li>2020–2021 kept as a disruption slice.</li>
            <li>CQR on a contiguous calibration window.</li>
          </ul>
        </Panel>
      </div>

      <Panel kicker="Walk the lecturer through" title="Suggested demo path">
        <div className="grid gap-3 md:grid-cols-4">
          {[
            ["01 Data", "/data", "Coverage, exclusions, cobweb plot"],
            ["02 ML Studio", "/studio", "Replay training + metrics"],
            ["03 Comparison", "/compare", "SARIMA vs LightGBM vs XGBoost"],
            ["04 Live forecast", "/forecast", "Move the commitment slider"],
          ].map(([label, to, hint]) => (
            <Link key={to} to={to} className="rounded-2xl border border-line bg-ink-2/60 p-4 transition hover:border-harvest/60">
              <p className="font-display text-lg">{label}</p>
              <p className="mt-1 text-sm text-mist">{hint}</p>
            </Link>
          ))}
        </div>
      </Panel>
    </div>
  );
}
