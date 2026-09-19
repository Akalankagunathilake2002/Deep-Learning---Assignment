import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import fixtures from "./__fixtures__/fixtures.json";
import { analyze } from "./engine.js";
import { GruModel } from "./gru.js";
import { prepareInput } from "./pipeline.js";

// Load the exported model files straight from disk (the browser downloads the same files).
const dir = new URL("../../public/model/", import.meta.url);
const json = (path) => JSON.parse(readFileSync(new URL(path, dir), "utf8"));
const arrayBuffer = (path) => {
  const b = readFileSync(new URL(path, dir));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

const config = json("config.json");
const vocab = new Map(Object.entries(json("vocab.json")));
const models = config.models.map((m) => ({
  id: m.id,
  name: m.name,
  model: new GruModel(json(`${m.dir}/manifest.json`), arrayBuffer(`${m.dir}/weights.bin`)),
}));
const engine = { config, vocab, models };
const byId = Object.fromEntries(models.map((m) => [m.id, m.model]));

describe("word ids match Python", () => {
  it("has the training vocabulary", () => {
    expect(vocab.size).toBe(config.vocabSize);
    expect(vocab.get("<PAD>")).toBe(0);
    expect(vocab.get("<OOV>")).toBe(1);
  });

  it.each(fixtures.map((f, i) => [i, f]))("reference text %i", (_i, f) => {
    const prep = prepareInput(f.text, vocab, config);
    expect(prep.ids).toEqual(f.ids);
    expect(prep.input).toHaveLength(config.maxLen);
  });

  it("pads with zeros in front and cuts long reviews at the end", () => {
    const short = prepareInput("great hotel", vocab, config);
    expect(short.input.slice(0, config.maxLen - 2).every((v) => v === 0)).toBe(true);
    expect(short.paddingCount).toBe(config.maxLen - 2);
    const long = prepareInput("word ".repeat(1000), vocab, config);
    expect(long.truncated).toBe(true);
    expect(long.ids).toHaveLength(config.maxLen);
  });

  it("treats words like 'constructor' as ordinary words", () => {
    const prep = prepareInput("constructor toString __proto__ hasOwnProperty", vocab, config);
    expect(prep.tokens).toEqual(["constructor", "tostring", "proto", "hasownproperty"]);
    expect(prep.ids.every((id) => Number.isInteger(id))).toBe(true);
  });
});

describe("the JavaScript GRU gives the same probability as Keras", () => {
  it.each(["main", "short"])("%s model on all reference texts", (id) => {
    let worst = 0;
    for (const f of fixtures) {
      const p = byId[id].predict(prepareInput(f.text, vocab, config).input);
      worst = Math.max(worst, Math.abs(p - f[id]));
    }
    expect(worst).toBeLessThan(1e-4);
  });
});

describe("analyze()", () => {
  it("scores a clear review with both models", () => {
    const a = analyze("We had a wonderful stay. The staff were friendly, the room was spotless and the bed was very comfortable. We loved the location and the breakfast. Highly recommended and we will be back next year.", engine);
    expect(a.empty).toBe(false);
    expect(a.short).toBe(false);
    expect(a.results.map((r) => r.id)).toEqual(["main", "short"]);
    expect(a.results.every((r) => r.p >= 0 && r.p <= 1)).toBe(true);
  });

  it("flags reviews shorter than any training review", () => {
    expect(analyze("Great hotel, lovely staff.", engine).short).toBe(true);
  });

  it("returns no scores when nothing usable is left", () => {
    const a = analyze("???", engine);
    expect(a.empty).toBe(true);
    expect(a.results).toEqual([]);
  });

  it("uses the 0.5 threshold from the config", () => {
    const a = analyze("Terrible experience. The room was dirty, the staff were rude and the noise kept us awake all night. It was the worst hotel I have ever stayed in and a complete waste of money. Never again.", engine);
    for (const r of a.results) expect(r.label).toBe(r.p >= config.threshold ? "Positive" : "Negative");
  });
});
