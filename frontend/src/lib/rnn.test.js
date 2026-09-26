import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { prepareInput } from "./pipeline.js";
import { RnnModel } from "./rnn.js";

const dir = new URL("../../public/model/", import.meta.url);
const json = (path) => JSON.parse(readFileSync(new URL(path, dir), "utf8"));
const arrayBuffer = (path) => {
  const b = readFileSync(new URL(path, dir));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

const config = json("config.json");
const vocab = new Map(Object.entries(json("vocab.json")));
const manifest = json("rnn/manifest.json");
const weights = arrayBuffer("rnn/weights.bin");
const rnnModel = new RnnModel(manifest, weights);

describe("Simple RNN in-browser inference", () => {
  it("initializes with expected dimensions", () => {
    expect(rnnModel.units).toBe(64);
    expect(rnnModel.embedDim).toBe(64);
    expect(rnnModel.hidden).toBe(32);
    expect(rnnModel.w.embedding).toHaveLength(320000);
    expect(rnnModel.w.kernel).toHaveLength(4096);
    expect(rnnModel.w.recurrent).toHaveLength(4096);
    expect(rnnModel.w.bias).toHaveLength(64);
    expect(rnnModel.w.w1).toHaveLength(2048);
    expect(rnnModel.w.b1).toHaveLength(32);
    expect(rnnModel.w.w2).toHaveLength(32);
    expect(rnnModel.w.b2).toHaveLength(1);
  });

  it("predicts positive sentiment for a clearly positive review", () => {
    const text = "We had a wonderful stay. The room was clean, spacious, and the staff were exceptionally friendly. Highly recommend!";
    const prep = prepareInput(text, vocab, config);
    const p = rnnModel.predict(prep.input);
    expect(p).toBeGreaterThan(0.5);
    expect(p).toBeLessThanOrEqual(1.0);
  });

  it("predicts negative sentiment for a clearly negative review", () => {
    const text = "Worst hotel ever. The room was dirty, noisy, smelled terrible, and the staff was extremely rude and unhelpful.";
    const prep = prepareInput(text, vocab, config);
    const p = rnnModel.predict(prep.input);
    expect(p).toBeLessThan(0.5);
    expect(p).toBeGreaterThanOrEqual(0.0);
  });
});
