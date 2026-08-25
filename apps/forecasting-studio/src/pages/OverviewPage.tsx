import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useStudio } from "@/entities/results/load";
import { Panel } from "@/shared/ui/Panel";

const LABELS: Record<string, { name: string; role: string }> = {
  sarima: { name: "Seasonal baseline", role: "SARIMA-family reference" },
  lightgbm: { name: "LightGBM", role: "Primary engine" },
  xgboost: { name: "XGBoost", role: "Boosting comparison" },
  lstm: { name: "Sequential network", role: "Optional deep comparison" },
};

export function OverviewPage() {
  const { comparison, split } = useStudio();
  const lgb = comparison.find((r) => r.id === "lightgbm");

  return (
    <div className="space-y-8">
      <motion.section initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
        <p className="text-[11px] uppercase tracking-[0.24em] text-harvest">Model results</p>
        <h2 className="font-display mt-3 max-w-4xl text-4xl leading-tight md:text-5xl">
          Four models, same weeks, same rules
        </h2>
        <p className="mt-5 max-w-3xl text-lg text-mist">
          Seasonal baseline, LightGBM, XGBoost, and an optional sequential network were all scored on{" "}
          {split.test.start} to {split.test.end} — {split.test.n} observed test weeks. LightGBM is what the
          dashboard uses because the 90% range holds and stays usable.
        </p>
      </motion.section>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {comparison.map((row) => {
          const meta = LABELS[row.id] ?? { name: row.model, role: row.backend };
          const selected = row.id === "lightgbm";
          return (
            <article
              key={row.id}
              className={`rounded-2xl border p-4 ${
                selected ? "border-harvest bg-harvest/10" : "border-line bg-panel"
              }`}
            >
              <p className="text-[11px] uppercase tracking-[0.16em] text-mist">{meta.role}</p>
              <h3 className="font-display mt-1 text-xl">{meta.name}</h3>
              <dl className="mt-4 space-y-2 text-sm">
                <Row label="MAE" value={`Rs. ${row.mae}`} />
                <Row label="MAPE" value={`${row.mape}%`} />
                <Row label="Coverage" value={`${row.picp}%`} />
                <Row label="Range width" value={`Rs. ${row.interval_width}`} />
              </dl>
              {selected && <p className="mt-3 text-xs text-harvest">Used on the dashboard</p>}
            </article>
          );
        })}
      </div>

      <Panel kicker="Same test set" title="Side by side">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-mist">
              <tr>
                {["Model", "MAE", "RMSE", "MAPE", "Coverage", "Width"].map((h) => (
                  <th key={h} className="pb-2 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparison.map((row) => (
                <tr key={row.id} className={`border-t border-line ${row.id === "lightgbm" ? "text-cream" : ""}`}>
                  <td className="py-2.5 font-medium">{LABELS[row.id]?.name ?? row.model}</td>
                  <td>{row.mae}</td>
                  <td>{row.rmse}</td>
                  <td>{row.mape}%</td>
                  <td>{row.picp}%</td>
                  <td>{row.interval_width}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-sm text-mist">
          The sequential network can look better on MAE, but coverage drops to {comparison.find((r) => r.id === "lstm")?.picp}%
          with a tight band. LightGBM stays near the 90% target{lgb ? ` (${lgb.picp}%)` : ""} with a width of about
          Rs. {lgb?.interval_width}.
        </p>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel kicker="Data notes" title="What this extract covers">
          <p className="text-sm leading-relaxed text-mist">
            Commitments are a labelled historical simulation, not live HARTI registrations. Colombo is the
            consumer-market series; Badulla and Nuwara Eliya are origin-adjacent.
          </p>
        </Panel>
        <Panel kicker="Evaluation" title="How the numbers were scored">
          <ul className="space-y-2 text-sm text-mist">
            <li>Time order only — train, then calibration, then test.</li>
            <li>Only originally observed wholesale prices.</li>
            <li>2020–2021 kept as a disruption slice.</li>
            <li>Coverage and width reported together.</li>
          </ul>
        </Panel>
      </div>

      <p className="text-sm text-mist">
        Charts and per-crop slices are on <Link to="/compare" className="text-harvest underline-offset-2 hover:underline">Comparison</Link>
        . Training curves are on <Link to="/studio" className="text-harvest underline-offset-2 hover:underline">Training</Link>.
      </p>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-mist">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
