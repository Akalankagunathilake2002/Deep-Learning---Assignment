# Frontend: try the GRU in your browser

A React (Vite) app for the hotel-review sentiment model. You paste a review and two versions of the trained GRU say whether it sounds positive or negative. **The models run entirely in your browser**: there is no server and no Python at run time, so nothing you type leaves the page.

## Run it

You need Node.js 20 or newer.

```bash
cd frontend
npm install
npm run dev          # then open the address it prints (http://localhost:5173)
```

Other commands:

| Command | What it does |
|---|---|
| `npm test` | Runs the tests (see "How we know it is correct" below) |
| `npm run build` | Makes a static site in `dist/` that works on any web host or folder |
| `npm run preview` | Serves the built site locally |

Open the app through `npm run dev` or `npm run preview`. Double-clicking `index.html` does not work, because the browser must download the model files over HTTP.

## What you see

- **Try it**: type or paste a review, or pick an example, then press *Analyze review* (or Ctrl/⌘ + Enter). You get one answer per model, each with the probability that the review is positive and a meter with the 0.5 cut-off marked. A panel shows exactly what the models received: the cleaned text and each word, with unknown words marked.
- **Results & about**: how the model works, the saved test-set results (accuracy, precision, recall, F1, ROC-AUC), the training curves, the confusion matrix, the ROC curve, and the limitations.

The model is the **GRU** reported in the group comparison. The page warns when a review is shorter than any training review, because the model is unreliable on very short text.

## How it works

The trained Keras models are not used in the browser. Instead:

1. `scripts/export_model.py` saves each model's weights as a plain binary file (`public/model/*/weights.bin`, 1.4 MB each), plus the training vocabulary, the settings and the saved results.
2. `src/lib/cleanText.js` repeats the notebooks' text cleaning, and `src/lib/pipeline.js` turns the text into word ids and pads it to 300, exactly as in training.
3. `src/lib/gru.js` runs the model: Embedding → GRU → Dense (ReLU) → Dense (sigmoid), written out in plain JavaScript. A review takes about 6 ms per model.

## How we know it is correct

The JavaScript has to give the same answers as the Keras models. This was checked in three ways:

- On **all 1,596 dataset reviews**: identical cleaned text, identical word ids, no changed prediction, and the largest difference in probability from Keras was under 0.000001.
- `npm test` repeats this on 83 reference texts stored in `src/lib/__fixtures__/` (dataset reviews plus awkward inputs such as HTML, URLs, contractions, curly quotes, accents, emoji and words like "constructor"), and also tests the screens with the real models.
- The built app was driven in a real browser: examples, typing, both tabs, phone width, light and dark mode, with no console errors and no failed requests.

## If the models are retrained

Run this from the repository root with a Python that has TensorFlow (see `requirements.txt`), then run `npm test`:

```bash
python frontend/scripts/export_model.py
```

It rebuilds `public/model/`, `public/results.json`, the figures and the test fixtures, and it stops with an error if the exported model does not reproduce the reported test accuracy. It reads the shared data pipeline in `../shared/`.

## Files

| Path | What it is |
|---|---|
| `src/App.jsx`, `src/components/` | The screens |
| `src/lib/` | Text cleaning, word ids, the GRU and the model loader, with their tests |
| `src/examples.json` | The example reviews |
| `public/model/`, `public/results.json`, `public/figures/` | Exported weights, vocabulary, settings, saved results and the report figures |
| `scripts/export_model.py` | Creates everything in `public/` from the trained models |

## Limits to know about

The probabilities are over-confident, mixed reviews are hard, and one-line reviews are unreliable, especially for the main GRU. The *Results & about* tab lists these with the numbers.
