import { describe, expect, it } from "vitest";
import fixtures from "./__fixtures__/fixtures.json";
import { cleanText } from "./cleanText.js";

// fixtures.json is produced by scripts/export_model.py: for each text, Python's clean_text() output.
describe("cleanText gives exactly the same text as the Python clean_text", () => {
  it.each(fixtures.map((f, i) => [i, f]))("reference text %i", (_i, f) => {
    expect(cleanText(f.text)).toBe(f.clean);
  });

  it("keeps negation words", () => {
    expect(cleanText("The room wasn't clean and we can't recommend it!")).toBe("the room was not clean and we can not recommend it");
  });

  it("removes HTML, URLs, digits and punctuation", () => {
    expect(cleanText("<b>Great</b> stay!!! 5 stars, see www.example.com or http://x.org/a?b=1")).toBe("great stay stars see or");
  });

  it("returns an empty string when nothing is left", () => {
    expect(cleanText("???  12345")).toBe("");
  });
});
