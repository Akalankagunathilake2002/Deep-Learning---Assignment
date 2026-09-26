import { Cnn1dModel } from "./cnn1d.js";
import { GruModel } from "./gru.js";
import { LstmModel } from "./lstm.js";
import { RnnModel } from "./rnn.js";
import { prepareInput } from "./pipeline.js";

const BASE = import.meta.env?.BASE_URL ?? "./";

async function get(base, path, kind) {
  const res = await fetch(`${base}${path}`);
  if (!res.ok) {
    throw new Error(`Could not load ${path} (HTTP ${res.status}). Open the app with "npm run dev" or "npm run preview", not by double-clicking index.html.`);
  }
  return kind === "json" ? res.json() : res.arrayBuffer();
}

/** Download the vocabulary and models (about 4.5 MB in total). Called once when the page opens. */
export async function loadEngine(base = BASE) {
  const config = await get(base, "model/config.json", "json");
  const vocab = new Map(Object.entries(await get(base, "model/vocab.json", "json"))); // a Map, so "constructor" is just a word
  const models = await Promise.all(
    config.models.map(async (m) => {
      const [manifest, weights] = await Promise.all([
        get(base, `model/${m.dir}/manifest.json`, "json"),
        get(base, `model/${m.dir}/weights.bin`, "arrayBuffer"),
      ]);
      return { id: m.id, name: m.name, model: new GruModel(manifest, weights) };
    }),
  );

  let cnnModel = null;
  try {
    const [cnnManifest, cnnWeights] = await Promise.all([
      get(base, "model/cnn1d/manifest.json", "json"),
      get(base, "model/cnn1d/weights.bin", "arrayBuffer"),
    ]);
    cnnModel = new Cnn1dModel(cnnManifest, cnnWeights);
  } catch (err) {
    console.warn("Could not load 1D CNN model files:", err);
  }

  return { config, vocab, models, cnnModel };
}

/** Load the trained LSTM model exported for the browser. */
export async function loadLstmModel(base = BASE) {
  const [manifest, weights] = await Promise.all([
    get(base, "model/lstm/manifest.json", "json"),
    get(base, "model/lstm/weights.bin", "arrayBuffer"),
  ]);
  return new LstmModel(manifest, weights);
}

/** Load the trained Simple RNN model exported for the browser. */
export async function loadRnnModel(base = BASE) {
  const [manifest, weights] = await Promise.all([
    get(base, "model/rnn/manifest.json", "json"),
    get(base, "model/rnn/weights.bin", "arrayBuffer"),
  ]);
  return new RnnModel(manifest, weights);
}

/** The saved test-set results shown on the Results tab. The page still works if this file is missing. */
export async function loadResults(base = BASE) {
  try {
    return await get(base, "results.json", "json");
  } catch {
    return null;
  }
}

/** Run one review through the same steps as training and ask every model for P(positive). */
export function analyze(text, engine) {
  const { config, vocab, models } = engine;
  const prep = prepareInput(text, vocab, config);
  const empty = prep.nWords === 0;
  const results = empty
    ? []
    : models.map(({ id, name, model }) => {
        const p = model.predict(prep.input);
        return { id, name, p, label: p >= config.threshold ? "Positive" : "Negative" };
      });
  return {
    ...prep,
    empty,
    short: !empty && prep.nWords < config.minTrainWords,
    disagree: results.length > 1 && results.some((r) => r.label !== results[0].label),
    results,
  };
}

/** Run inference on 1D CNN and compare with GRU reference */
export function analyzeCnn(text, engine, cnnModelInstance = null) {
  const { config, vocab, models } = engine;
  const prep = prepareInput(text, vocab, config);
  const empty = prep.nWords === 0;
  if (empty) {
    return { ...prep, empty: true, short: false, results: [] };
  }

  const modelToUse = cnnModelInstance || engine.cnnModel;
  const results = [];

  if (modelToUse) {
    const t0 = performance.now();
    const pCnn = modelToUse.predict(prep.input);
    const ms = Math.max(0.1, performance.now() - t0);
    results.push({
      id: "cnn1d",
      name: "1D CNN (Dilmith)",
      p: pCnn,
      label: pCnn >= config.threshold ? "Positive" : "Negative",
      badge: "Champion",
      latencyMs: ms,
    });
  }

  // Also query Main GRU for direct comparison
  const mainGru = models?.find((m) => m.id === "main");
  if (mainGru) {
    const t0 = performance.now();
    const pGru = mainGru.model.predict(prep.input);
    const ms = Math.max(0.1, performance.now() - t0);
    results.push({
      id: "main",
      name: "GRU Reference",
      p: pGru,
      label: pGru >= config.threshold ? "Positive" : "Negative",
      badge: "Reference",
      latencyMs: ms,
    });
  }

  return {
    ...prep,
    empty: false,
    short: !empty && prep.nWords < config.minTrainWords,
    disagree: results.length > 1 && results.some((r) => r.label !== results[0].label),
    results,
  };
}
