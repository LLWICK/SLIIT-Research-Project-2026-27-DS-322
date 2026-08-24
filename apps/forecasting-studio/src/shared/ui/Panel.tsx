import type { ReactNode } from "react";

export function Panel({
  title,
  kicker,
  children,
  className = "",
}: {
  title?: string;
  kicker?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-3xl border border-line/80 bg-panel/80 p-5 md:p-6 ${className}`}>
      {(kicker || title) && (
        <header className="mb-4">
          {kicker && <p className="text-[11px] uppercase tracking-[0.22em] text-mist">{kicker}</p>}
          {title && <h3 className="font-display mt-1 text-xl text-cream">{title}</h3>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-2xl border border-line bg-ink-2/70 px-4 py-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-mist">{label}</p>
      <p className="font-display mt-1 text-2xl text-harvest">{value}</p>
      {hint && <p className="mt-1 text-xs text-mist">{hint}</p>}
    </div>
  );
}
