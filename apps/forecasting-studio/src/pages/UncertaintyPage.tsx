import { useEffect, useMemo, useState } from "react";
import { Area, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useStudio } from "@/entities/results/load";
import { Panel } from "@/shared/ui/Panel";
import { titleCase } from "@/shared/lib/format";

type Pred = {
  crop: string;
  market: string;
  week_start: string;
  y_true: number;
  point: number;
  lower: number;
  upper: number;
};

export function UncertaintyPage() {
  const { models, comparison } = useStudio();
  const lgb = comparison.find((r) => r.id === "lightgbm")!;
  const [preds, setPreds] = useState<Pred[]>([]);
  const [key, setKey] = useState("carrot__colombo");

  useEffect(() => {
    fetch("/data/predictions.json")
      .then((r) => r.json())
      .then((json: { lightgbm: Pred[] }) => setPreds(json.lightgbm ?? []));
  }, []);

  const [crop, market] = key.split("__");
  const slice = useMemo(
    () => preds.filter((p) => p.crop === crop && p.market === market).slice(-80),
    [preds, crop, market],
  );
  const keys = useMemo(() => {
    const set = new Set(preds.map((p) => `${p.crop}__${p.market}`));
    return [...set];
  }, [preds]);

  return (
    <div className="space-y-6">
      <header>
        <p className="text-[11px] uppercase tracking-[0.22em] text-harvest">Chronological CQR</p>
        <h2 className="font-display mt-2 text-4xl">90% targeted coverage, empirically evaluated</h2>
        <p className="mt-3 max-w-3xl text-mist">
          We do not claim an i.i.d. mathematical guarantee on agricultural prices. We conformalize the 5th/95th
          LightGBM quantiles on a contiguous calibration block (q̂ = {models.lightgbm.qhat}) and then measure PICP
          on the later test weeks.
        </p>
      </header>

      <div className="grid gap-3 md:grid-cols-3">
        <Stat label="Empirical PICP" value={`${lgb.picp}%`} hint="Target 90%" />
        <Stat label="Mean width" value={`Rs. ${lgb.interval_width}`} hint="Sharpness half of the result" />
        <Stat label="CQR q-hat" value={`${models.lightgbm.qhat}`} hint="Added to both tails" />
      </div>

      <Panel kicker="Hold-out fan chart" title="Actual vs interval · last 80 test weeks">
        <div className="mb-3 flex flex-wrap gap-2">
          {keys.map((id) => (
            <button
              key={id}
              onClick={() => setKey(id)}
              className={`rounded-full px-3 py-1 text-sm ${
                key === id ? "bg-harvest text-ink" : "border border-line text-mist"
              }`}
            >
              {titleCase(id.replace("__", " · "))}
            </button>
          ))}
        </div>
        <div className="h-80">
          <ResponsiveContainer>
            <ComposedChart data={slice}>
              <XAxis dataKey="week_start" minTickGap={28} />
              <YAxis />
              <Tooltip contentStyle={{ background: "#10211b", border: "1px solid #1d3a30" }} />
              <Area dataKey="upper" stroke="none" fill="#c6f31a" fillOpacity={0.12} />
              <Area dataKey="lower" stroke="none" fill="#07110e" fillOpacity={1} />
              <Line dataKey="y_true" stroke="#f5b942" dot={false} strokeWidth={2} />
              <Line dataKey="point" stroke="#c6f31a" dot={false} strokeWidth={2} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-xs text-mist">Amber = realised wholesale price. Lime = median forecast. Band = calibrated 90% interval.</p>
      </Panel>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-2xl border border-line bg-panel p-4">
      <p className="text-[11px] uppercase tracking-[0.16em] text-mist">{label}</p>
      <p className="font-display mt-1 text-3xl text-harvest">{value}</p>
      <p className="mt-1 text-sm text-mist">{hint}</p>
    </div>
  );
}
