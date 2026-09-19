// Port of clean_text() from the GRU notebooks.
// It must return exactly the same text as the Python version, because the model was trained on that text.
// cleanText.test.js checks this against Python on more than 80 reference texts.

// Python's \b and \w treat letters, digits and "_" as word characters. These lookarounds do the same.
const WORD = "\\p{L}\\p{N}_";
const NOT_AFTER_WORD = `(?<![${WORD}])`;
const NOT_BEFORE_WORD = `(?![${WORD}])`;
const rx = (source) => new RegExp(source, "gu");

const HTML = /<[^>]+>/gu;
const URL = /(https?:\/\/\S+|www\.\S+)/gu;

// Keep negation: "didn't" becomes "did not", not "didn t". Order matters, as in the Python list.
const CONTRACTIONS = [
  [rx(`${NOT_AFTER_WORD}won't${NOT_BEFORE_WORD}`), "will not"],
  [rx(`${NOT_AFTER_WORD}can't${NOT_BEFORE_WORD}`), "can not"],
  [rx(`${NOT_AFTER_WORD}cannot${NOT_BEFORE_WORD}`), "can not"],
  [rx(`n't${NOT_BEFORE_WORD}`), " not"],
  [rx(`'re${NOT_BEFORE_WORD}`), " are"],
  [rx(`'ve${NOT_BEFORE_WORD}`), " have"],
  [rx(`'ll${NOT_BEFORE_WORD}`), " will"],
  [rx(`'d${NOT_BEFORE_WORD}`), " would"],
  [rx(`'m${NOT_BEFORE_WORD}`), " am"],
  [rx(`'s${NOT_BEFORE_WORD}`), ""],
];

const NON_LETTER = /[^a-z\s]/gu; // digits, punctuation, symbols, accents
const SPACES = /\s+/gu;

/** lowercase -> strip HTML -> strip URLs -> expand contractions -> keep letters only. Stop-words are kept. */
export function cleanText(text) {
  let t = text.toLowerCase().replaceAll("’", "'");
  t = t.replace(HTML, " ").replace(URL, " ");
  for (const [pattern, replacement] of CONTRACTIONS) t = t.replace(pattern, replacement);
  return t.replace(NON_LETTER, " ").replace(SPACES, " ").trim();
}
