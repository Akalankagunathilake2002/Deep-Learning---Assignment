/** "97.2%", but never claim 100% or 0% certainty. */
export function formatPercent(p) {
  if (p > 0.999) return ">99.9%";
  if (p < 0.001) return "<0.1%";
  return `${(p * 100).toFixed(1)}%`;
}

/** 0.8812 -> "88.1%" */
export const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;

export const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
