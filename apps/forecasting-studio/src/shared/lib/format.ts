export const titleCase = (value: string) =>
  value.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());

export const lkr = (n: number, digits = 0) =>
  `Rs. ${n.toLocaleString("en-LK", { maximumFractionDigits: digits, minimumFractionDigits: digits })}`;

export const pct = (n: number, digits = 1) => `${n.toFixed(digits)}%`;
