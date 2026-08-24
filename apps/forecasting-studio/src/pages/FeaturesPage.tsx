import { useStudio } from "@/entities/results/load";
import { Panel } from "@/shared/ui/Panel";
import { titleCase } from "@/shared/lib/format";

export function FeaturesPage() {
  const { originMap, dictionary, importance } = useStudio();

  return (
    <div className="space-y-6">
      <header>
        <p className="text-[11px] uppercase tracking-[0.22em] text-harvest">Leak-safe features</p>
        <h2 className="font-display mt-2 text-4xl">Markets observe prices. Districts grow crops.</h2>
        <p className="mt-3 max-w-3xl text-mist">
          Weather and commitments never join on the market name. They are aggregated from the origin map, and
          intensity is as-of-week: a registration after the forecast week cannot enter the feature.
        </p>
      </header>

      <Panel kicker="Origin map" title="Supply districts behind each modelled market">
        <div className="grid gap-3 md:grid-cols-3">
          {Object.entries(originMap).map(([crop, markets]) => (
            <div key={crop} className="rounded-2xl border border-line bg-ink-2/60 p-4">
              <p className="font-display text-xl">{titleCase(crop)}</p>
              <ul className="mt-3 space-y-2 text-sm text-mist">
                {Object.entries(markets).map(([market, districts]) => (
                  <li key={market}>
                    <span className="text-cream">{titleCase(market)}</span>
                    <span> ← {districts.map(titleCase).join(", ")}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel kicker="Novelty feature" title="Cultivation intensity">
          <pre className="overflow-x-auto rounded-2xl bg-ink p-4 text-xs leading-relaxed text-sky">
{`intensity(market, crop, week) =
  cumulative hectares in supply districts
  registered by this week
  ÷ demand benchmark for that season`}
          </pre>
          <p className="mt-3 text-sm text-mist">
            Source in this prototype: historically grounded simulation of seasonal extent
            (<code>commitment_source = simulated</code>). Same function signature as a future HARTI feed.
          </p>
        </Panel>
        <Panel kicker="Permutation importance" title="What Model C actually uses">
          <ul className="space-y-2">
            {importance.slice(0, 6).map((row) => (
              <li key={row.feature}>
                <div className="flex justify-between text-sm">
                  <span>{row.feature}</span>
                  <span className="text-harvest">{row.importance.toFixed(2)}</span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-line">
                  <div
                    className="h-full bg-harvest"
                    style={{ width: `${Math.min(100, (row.importance / importance[0].importance) * 100)}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-sm text-mist">
            Lag-1 dominates, as expected on weekly prices. Cultivation intensity ranks in the top features — the
            first supporting evidence for the novelty, not the whole proof.
          </p>
        </Panel>
      </div>

      <Panel kicker="Generated dictionary" title="Panel contract">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-mist">
              <tr>
                {["Column", "Type", "Role", "Notes"].map((h) => (
                  <th key={h} className="pb-2 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dictionary.map((row) => (
                <tr key={row.column} className="border-t border-line align-top">
                  <td className="py-2 font-medium">{row.column}</td>
                  <td className="text-mist">{row.dtype}</td>
                  <td>{row.role}</td>
                  <td className="text-mist">{row.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
