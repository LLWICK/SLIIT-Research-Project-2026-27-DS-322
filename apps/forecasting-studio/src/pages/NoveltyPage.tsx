import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useStudio } from "@/entities/results/load";
import { Panel } from "@/shared/ui/Panel";

export function NoveltyPage() {
  const { ablation, weeks, importance } = useStudio();
  const arms = ["A", "B", "C"] as const;
  const intensityRank = importance.findIndex((r) => r.feature === "cultivation_intensity") + 1;

  return (
    <div className="space-y-6">
      <header>
        <p className="text-[11px] uppercase tracking-[0.22em] text-harvest">The two experiments that carry the mark</p>
        <h2 className="font-display mt-2 text-4xl">A/B/C ablation and mid-season re-forecast</h2>
        <p className="mt-3 max-w-3xl text-mist">
          Same LightGBM family, same hyperparameters, same chronological folds, same seed. Only the column list
          changes. A rigorous near-tie is still a result — we do not hide it.
        </p>
      </header>

      <Panel kicker="Ablation" title="A historical · B multi-source · C + intensity">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-mist">
              <tr>
                {["Arm", "MAE", "RMSE", "MAPE", "Pinball", "PICP", "Width"].map((h) => (
                  <th key={h} className="pb-2 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {arms.map((arm) => {
                const m = ablation[arm].metrics;
                return (
                  <tr key={arm} className={`border-t border-line ${arm === "C" ? "bg-harvest/8" : ""}`}>
                    <td className="py-2 font-medium">{arm}</td>
                    <td>{m.mae}</td>
                    <td>{m.rmse}</td>
                    <td>{m.mape}%</td>
                    <td>{m.pinball}</td>
                    <td>{m.picp}%</td>
                    <td>{m.interval_width}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-sm text-mist">
          On this extract, B is slightly best on MAE. C stays within 0.2 LKR and keeps PICP near 90%. Intensity
          ranks #{intensityRank} in permutation importance. The operational proof is the week experiment, not a
          forced accuracy win.
        </p>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel kicker="Week 2 / 5 / 8 / 12" title="Error falls as commitments accumulate">
          <div className="h-64">
            <ResponsiveContainer>
              <LineChart data={weeks}>
                <XAxis dataKey="commitment_week" />
                <YAxis />
                <Tooltip contentStyle={{ background: "#10211b", border: "1px solid #1d3a30" }} />
                <Line dataKey="mae" stroke="#c6f31a" strokeWidth={2} />
                <Line dataKey="rmse" stroke="#f5b942" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-sm text-mist">
            MAE {weeks[0].mae} at week 2 → {weeks.at(-1)?.mae} at week 12. The forecast is updated from new supply
            information, not from later prices.
          </p>
        </Panel>
        <Panel kicker="Information states" title="What the farmer would have known">
          <ul className="space-y-3">
            {weeks.map((w) => (
              <li key={w.commitment_week} className="flex items-center justify-between rounded-2xl border border-line bg-ink-2/50 px-4 py-3">
                <div>
                  <p className="font-medium">Week {w.commitment_week}</p>
                  <p className="text-xs text-mist">mean intensity {w.mean_intensity}</p>
                </div>
                <div className="text-right text-sm">
                  <p>MAE {w.mae}</p>
                  <p className="text-mist">PICP {w.picp}%</p>
                </div>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
