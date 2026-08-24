export function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line/70 py-2 last:border-b-0">
      <span className="text-sm text-mist">{label}</span>
      <span className="text-right text-sm text-cream">{value}</span>
    </div>
  );
}
