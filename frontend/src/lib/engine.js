import { GruModel } from "./gru.js";
import { prepareInput } from "./pipeline.js";

const BASE = import.meta.env?.BASE_URL ?? "./";

async function get(base, path, kind) {
  const res = await fetch(`${base}${path}`);
  if (!res.ok) {
    throw new Error(`Could not load ${path} (HTTP ${res.status}). Open the app with "npm run dev" or "npm run preview", not by double-clicking index.html.`);
  }
  return kind === "json" ? res.json() : res.arrayBuffer();
}

/** Download the vocabulary and both models (about 3 MB in total). Called once when the page opens. */
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
  return { config, vocab, models };
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
