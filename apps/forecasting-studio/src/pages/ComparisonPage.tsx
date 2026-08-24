import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useStudio } from "@/entities/results/load";
import { Panel } from "@/shared/ui/Panel";

export function ComparisonPage() {
  const { comparison, models } = useStudio();
  const lgb = models.lightgbm;

  return (
    <div className="space-y-6">
      <header>
        <p className="text-[11px] uppercase tracking-[0.22em] text-harvest">Same folds · same seed · observed only</p>
        <h2 className="font-display mt-2 text-4xl">Did we beat the baseline?</h2>
        <p className="mt-3 max-w-3xl text-mist">
          Yes. LightGBM quantile + CQR cuts MAE from Rs. {comparison[0].mae} to Rs.{" "}
          {comparison.find((r) => r.id === "lightgbm")?.mae} and lifts PICP from {comparison[0].picp}% to{" "}
          {comparison.find((r) => r.id === "lightgbm")?.picp}%.
        </p>
      </header>

      <Panel kicker="PP2 table" title="Headline comparison">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-mist">
              <tr>
                {["Model", "MAE", "RMSE", "MAPE", "Pinball", "PICP", "Width", "Backend"].map((h) => (
                  <th key={h} className="pb-2 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparison.map((row) => (
                <tr
                  key={row.id}
                  className={`border-t border-line ${row.id === "lightgbm" ? "bg-harvest/8 text-cream" : ""}`}
                >
                  <td className="py-2 font-medium">{row.model}</td>
                  <td>{row.mae}</td>
                  <td>{row.rmse}</td>
                  <td>{row.mape}%</td>
                  <td>{row.pinball}</td>
                  <td>{row.picp}%</td>
                  <td>{row.interval_width}</td>
                  <td className="text-mist">{row.backend}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel kicker="Point accuracy" title="MAE (LKR/kg)">
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={comparison}>
                <XAxis dataKey="id" />
                <YAxis />
                <Tooltip contentStyle={{ background: "#10211b", border: "1px solid #1d3a30" }} />
                <Bar dataKey="mae" fill="#c6f31a" radius={8} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel kicker="Uncertainty quality" title="PICP vs interval width">
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={comparison}>
                <XAxis dataKey="id" />
                <YAxis />
                <Tooltip contentStyle={{ background: "#10211b", border: "1px solid #1d3a30" }} />
                <Bar dataKey="picp" fill="#7dd3c0" radius={8} />
                <Bar dataKey="interval_width" fill="#f5b942" radius={8} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <Panel kicker="LightGBM slices" title="Per crop and market">
        <div className="grid gap-4 md:grid-cols-2">
          <SliceTable title="By crop" rows={lgb.metrics.by_crop} />
          <SliceTable title="By market" rows={lgb.metrics.by_market ?? {}} />
        </div>
      </Panel>
    </div>
  );
}

function SliceTable({ title, rows }: { title: string; rows: Record<string, { mae: number; picp: number; interval_width: number; mape: number }> }) {
  return (
    <div>
      <p className="mb-2 text-sm text-mist">{title}</p>
      <table className="w-full text-left text-sm">
        <thead className="text-mist">
          <tr>
            <th className="font-normal">Slice</th>
            <th className="font-normal">MAE</th>
            <th className="font-normal">MAPE</th>
            <th className="font-normal">PICP</th>
            <th className="font-normal">Width</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(rows).map(([k, v]) => (
            <tr key={k} className="border-t border-line">
              <td className="py-2">{k}</td>
              <td>{v.mae}</td>
              <td>{v.mape}%</td>
              <td>{v.picp}%</td>
              <td>{v.interval_width}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
