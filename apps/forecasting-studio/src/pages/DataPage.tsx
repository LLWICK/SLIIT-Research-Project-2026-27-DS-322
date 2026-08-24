import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useStudio } from "@/entities/results/load";
import { Panel } from "@/shared/ui/Panel";
import { titleCase } from "@/shared/lib/format";

export function DataPage() {
  const { coverage, series, cobweb, split, meta } = useStudio();
  const [key, setKey] = useState("carrot__colombo");
  const [crop, market] = key.split("__");
  const selected = series.find((s) => s.crop === crop && s.market === market);
  const chart = useMemo(
    () => (selected?.points ?? []).filter((p) => p.price != null),
    [selected],
  );
  const cob = cobweb.filter((r) => r.crop === crop && r.market === market);

  return (
    <div className="space-y-6">
      <header>
        <p className="text-[11px] uppercase tracking-[0.22em] text-harvest">Panel construction</p>
        <h2 className="font-display mt-2 text-4xl">The weekly panel, as the audit left it</h2>
        <p className="mt-3 max-w-3xl text-mist">
          {meta.n_panel.toLocaleString()} wholesale weekly rows after filters. Member 1’s analysis-ready extract
          already keeps observed prices; we still refuse to zero-fill, and we score only those observed rows.
        </p>
      </header>

      <div className="grid gap-3 md:grid-cols-3">
        {Object.entries(split).map(([name, block]) => (
          <div key={name} className="rounded-2xl border border-line bg-panel p-4">
            <p className="text-[11px] uppercase tracking-[0.18em] text-mist">{name}</p>
            <p className="font-display mt-1 text-2xl">{block.n} rows</p>
            <p className="text-sm text-mist">
              {block.start} → {block.end}
            </p>
          </div>
        ))}
      </div>

      <Panel kicker="Series explorer" title="Wholesale price · crop × market">
        <div className="mb-4 flex flex-wrap gap-2">
          {series.map((s) => {
            const id = `${s.crop}__${s.market}`;
            return (
              <button
                key={id}
                onClick={() => setKey(id)}
                className={`rounded-full px-3 py-1 text-sm ${
                  key === id ? "bg-harvest text-ink" : "border border-line text-mist"
                }`}
              >
                {titleCase(s.crop)} · {titleCase(s.market)}
              </button>
            );
          })}
        </div>
        <div className="h-72">
          <ResponsiveContainer>
            <LineChart data={chart}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="week" minTickGap={48} />
              <YAxis />
              <Tooltip contentStyle={{ background: "#10211b", border: "1px solid #1d3a30" }} />
              <Line type="monotone" dataKey="price" stroke="#c6f31a" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-xs text-mist">Amber-adjacent spikes in 2020–2021 are flagged as disruption, not interpolated.</p>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel kicker="Money plot" title="Season t intensity vs season t+1 price">
          <div className="h-64">
            <ResponsiveContainer>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="intensity_t" name="intensity" />
                <YAxis dataKey="price_t1" name="next price" />
                <Tooltip contentStyle={{ background: "#10211b", border: "1px solid #1d3a30" }} />
                <Scatter data={cob} fill="#f5b942" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-sm text-mist">
            This is the cobweb sketch in our own panel: planting pressure this season against the next season’s market price.
          </p>
        </Panel>
        <Panel kicker="Exclusions" title="What we refuse to pretend">
          <ul className="space-y-3">
            {coverage.exclusions.map((item) => (
              <li key={item.item} className="rounded-2xl border border-line bg-ink-2/50 p-3">
                <p className="text-sm font-medium">{item.item}</p>
                <p className="mt-1 text-sm text-mist">{item.reason}</p>
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <Panel kicker="Coverage" title="Wholesale series used in the model">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-mist">
              <tr>
                {["Crop", "Market", "Rows", "Coverage", "Start", "End"].map((h) => (
                  <th key={h} className="pb-2 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {coverage.by_series.map((row) => (
                <tr key={`${row.crop}-${row.market}`} className="border-t border-line">
                  <td className="py-2">{titleCase(String(row.crop))}</td>
                  <td>{titleCase(String(row.market))}</td>
                  <td>{row.rows}</td>
                  <td>{row.coverage_pct}%</td>
                  <td className="text-mist">{String(row.start).slice(0, 10)}</td>
                  <td className="text-mist">{String(row.end).slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
