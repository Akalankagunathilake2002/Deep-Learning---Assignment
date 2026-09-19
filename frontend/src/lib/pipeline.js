import { cleanText } from "./cleanText.js";

/**
 * Turn a review into what the model receives: clean -> split into words -> encode -> pad.
 * Uses the same vocabulary (built from the training set only) and the same padding as the notebooks.
 *
 * @param {string} text
 * @param {Map<string, number>} vocab word -> id (0 = <PAD>, 1 = <OOV>)
 * @param {{maxLen:number, oovToken:string, padding:string, truncating:string}} config
 */
export function prepareInput(text, vocab, config) {
  const clean = cleanText(text);
  const tokens = clean === "" ? [] : clean.split(" ");
  const oovId = vocab.get(config.oovToken);
  const allIds = tokens.map((t) => vocab.get(t) ?? oovId);

  // Reviews longer than maxLen are cut (truncating "post" keeps the first words).
  const ids = config.truncating === "post" ? allIds.slice(0, config.maxLen) : allIds.slice(-config.maxLen);
  const input = new Int32Array(config.maxLen); // zeros = <PAD>
  if (config.padding === "pre") input.set(ids, config.maxLen - ids.length);
  else input.set(ids, 0);

  return {
    clean,
    tokens,
    ids,
    input,
    nWords: tokens.length,
    truncated: tokens.length > config.maxLen,
    paddingCount: config.maxLen - ids.length,
    unknownWords: tokens.filter((t) => !vocab.has(t)),
  };
}
