import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { RnnModel } from "../src/lib/rnn.js";
import { prepareInput } from "../src/lib/pipeline.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "../..");

const VOCAB_PATH = resolve(ROOT, "frontend/public/model/vocab.json");
const CONFIG_PATH = resolve(ROOT, "frontend/public/model/config.json");
const MANIFEST_PATH = resolve(ROOT, "frontend/public/model/rnn/manifest.json");
const WEIGHTS_PATH = resolve(ROOT, "frontend/public/model/rnn/weights.bin");
const DATASET_PATH = resolve(ROOT, "dataset/deceptive-opinion.csv");
const TEST_PREDS_PATH = resolve(ROOT, "simple_rnn/results/simple_rnn_test_predictions.csv");

const vocab = new Map(Object.entries(JSON.parse(readFileSync(VOCAB_PATH, "utf8"))));
const config = JSON.parse(readFileSync(CONFIG_PATH, "utf8"));
const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
const weightsBuf = readFileSync(WEIGHTS_PATH);
const weights = weightsBuf.buffer.slice(weightsBuf.byteOffset, weightsBuf.byteOffset + weightsBuf.byteLength);

const model = new RnnModel(manifest, weights);

// Parse CSV manually or simply
function parseCsv(content) {
  const lines = content.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;
    // Handle simple CSV splitting where text is not quoted or handle standard
    const cols = line.split(",");
    rows.push({
      row_id: parseInt(cols[0], 10),
      label: parseInt(cols[1], 10),
      hotel: cols[2],
      deceptive: cols[3],
      prob_positive: parseFloat(cols[4]),
      prediction: parseInt(cols[5], 10),
    });
  }
  return rows;
}

// Load dataset texts
const datasetContent = readFileSync(DATASET_PATH, "utf8");
// To reliably parse deceptive-opinion.csv text with commas and newlines
import { parse } from "node:path";

function parseFullDataset(csvText) {
  const records = new Map();
  // Simple regex or state machine for RFC 4180 CSV
  const lines = csvText.split("\n");
  const header = lines[0];
  let curRow = 0;
  let inQuotes = false;
  let curField = "";
  let curFields = [];

  for (let i = header.length + 1; i < csvText.length; i++) {
    const c = csvText[i];
    if (c === '"') {
      if (inQuotes && csvText[i + 1] === '"') {
        curField += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (c === "," && !inQuotes) {
      curFields.push(curField);
      curField = "";
    } else if ((c === "\r" || c === "\n") && !inQuotes) {
      if (c === "\r" && csvText[i + 1] === "\n") i++;
      curFields.push(curField);
      curField = "";
      if (curFields.length >= 5) {
        // [deceptive, hotel, polarity, source, text]
        records.set(curRow, curFields[4]);
        curRow++;
      }
      curFields = [];
    } else {
      curField += c;
    }
  }
  if (curFields.length >= 5) {
    records.set(curRow, curFields[4]);
  }
  return records;
}

const datasetTexts = parseFullDataset(datasetContent);
const testPreds = parseCsv(readFileSync(TEST_PREDS_PATH, "utf8"));

console.log(`Checking ${testPreds.length} test samples...`);

let maxDiff = 0;
let sumDiff = 0;
let agreements = 0;

for (const row of testPreds) {
  const text = datasetTexts.get(row.row_id);
  if (!text) {
    console.error(`Missing text for row_id ${row.row_id}`);
    continue;
  }

  const prep = prepareInput(text, vocab, config);
  const jsProb = model.predict(prep.input);
  const jsPred = jsProb >= 0.5 ? 1 : 0;

  const pyProb = row.prob_positive;
  const pyPred = row.prediction;

  const diff = Math.abs(jsProb - pyProb);
  if (diff > maxDiff) maxDiff = diff;
  sumDiff += diff;

  if (jsPred === pyPred) {
    agreements++;
  } else {
    console.error(`DISAGREEMENT on row ${row.row_id}: JS=${jsPred} (p=${jsProb}), PY=${pyPred} (p=${pyProb})`);
  }
}

const meanDiff = sumDiff / testPreds.length;
const agreementPct = (agreements / testPreds.length) * 100;

console.log("\n=== PYTHON VS JAVASCRIPT EQUIVALENCE RESULTS ===");
console.log(`Samples checked:           ${testPreds.length}`);
console.log(`Classification agreement:  ${agreements}/${testPreds.length} (${agreementPct.toFixed(2)}%)`);
console.log(`Max absolute probability diff:  ${maxDiff.toExponential(4)}`);
console.log(`Mean absolute probability diff: ${meanDiff.toExponential(4)}`);

if (agreements === testPreds.length && maxDiff < 1e-4) {
  console.log(">>> NUMERICAL EQUIVALENCE VERIFIED TO BE EXTREMELY CLOSE! <<<");
} else {
  process.exit(1);
}
