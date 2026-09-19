import { describe, expect, it } from "vitest";
import { formatPercent, pct, plural } from "./format.js";

describe("format helpers", () => {
  it("never claims 100% or 0% certainty", () => {
    expect(formatPercent(0.99999)).toBe(">99.9%");
    expect(formatPercent(0.00001)).toBe("<0.1%");
    expect(formatPercent(0.972)).toBe("97.2%");
    expect(formatPercent(0.5)).toBe("50.0%");
  });

  it("formats fractions as percentages", () => {
    expect(pct(0.88125)).toBe("88.1%");
    expect(pct(0.88125, 0)).toBe("88%");
  });

  it("pluralises", () => {
    expect(plural(1, "word")).toBe("1 word");
    expect(plural(0, "word")).toBe("0 words");
    expect(plural(12, "review")).toBe("12 reviews");
  });
});
