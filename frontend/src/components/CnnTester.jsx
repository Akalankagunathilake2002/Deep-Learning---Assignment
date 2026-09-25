import { useMemo, useRef, useState, useEffect } from "react";
import examples from "../examples.json";
import { formatPercent } from "../format.js";
import { analyzeCnn } from "../lib/engine.js";
import { Cnn1dModel } from "../lib/cnn1d.js";
import ModelCard from "./ModelCard.jsx";
import TokenView from "./TokenView.jsx";

const CNN_EXAMPLES = [
  {
    id: "pos",
    name: "Positive (Dataset)",
    text: "Triple AAA rate of 173 was a steal. 7th floor room complete with 44in plasma TV bose stereo, and gorgeous bathroom. Concierge was very helpful. You cannot beat this location. Food was very good so it was worth the wait. A gem in chicago.",
  },
  {
    id: "neg",
    name: "Negative (Dataset)",
    text: "Terrible experience. The room smelled of mildew, carpet was visibly stained, and the air conditioner made loud clanking noises all night. Staff at reception were indifferent and dismissive. Will never stay here again.",
  },
  {
    id: "negation",
    name: "Negation Challenge",
    text: "The room was not dirty and the front desk staff was certainly not unhelpful. Not what I feared at all, actually had a wonderful stay.",
  },
  {
    id: "short",
    name: "Short Phrase",
    text: "Excellent service and lovely clean rooms!",
  },
];

export default function CnnTester({ state, onSelectGru }) {
  const [text, setText] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [cnnModel, setCnnModel] = useState(state.engine?.cnnModel || null);
  const textarea = useRef(null);

  const ready = state.status === "ready";
  const words = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text]);

  // Load CNN model if not already present on state.engine
  useEffect(() => {
    if (state.engine?.cnnModel) {
      setCnnModel(state.engine.cnnModel);
      return;
    }
    let active = true;
    const base = import.meta.env?.BASE_URL ?? "./";
    Promise.all([
      fetch(`${base}model/cnn1d/manifest.json`).then((r) => r.json()),
      fetch(`${base}model/cnn1d/weights.bin`).then((r) => r.arrayBuffer()),
    ])
      .then(([manifest, buffer]) => {
        if (active) {
          const model = new Cnn1dModel(manifest, buffer);
          setCnnModel(model);
          if (state.engine) {
            state.engine.cnnModel = model;
          }
        }
      })
      .catch((err) => {
        // In Node / jsdom test environments without server origin, relative URLs in undici fetch throw ERR_INVALID_URL
        if (err?.code !== "ERR_INVALID_URL" && err?.cause?.code !== "ERR_INVALID_URL") {
          console.warn("Could not lazily fetch cnn1d browser weights:", err);
        }
      });
    return () => {
      active = false;
    };
  }, [state.engine]);

  const run = (value) => {
    if (!ready || !value.trim()) return;
    const res = analyzeCnn(value, state.engine, cnnModel);
    setAnalysis({ ...res, sourceText: value });
  };

  const pickExample = (ex) => {
    setText(ex.text);
    run(ex.text);
    textarea.current?.focus();
  };

  const clear = () => {
    setText("");
    setAnalysis(null);
    textarea.current?.focus();
  };

  return (
    <div className="try-grid">
      <section className="card input-card" aria-labelledby="cnn-input-title">
        <div className="cnn-header-row">
          <div>
            <h2 id="cnn-input-title">Test 1D CNN Live in Browser</h2>
            <p className="muted">
              Type or paste any hotel review to test the <strong>1D CNN Champion Model (Dilmith)</strong> running
              entirely in your browser with zero latency.
            </p>
          </div>
          <span className="status-badge active">CNN Inference Active</span>
        </div>

        <label htmlFor="cnn-review-input" className="sr-only">
          Hotel review
        </label>
        <textarea
          id="cnn-review-input"
          ref={textarea}
          className="textarea"
          rows={5}
          placeholder="e.g. We had a wonderful weekend stay. The room was spacious and the concierge gave great restaurant advice..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
              e.preventDefault();
              run(text);
            }
          }}
          disabled={!ready}
        />

        <div className="input-toolbar">
          <div className="words-counter" aria-live="polite">
            {words} {words === 1 ? "word" : "words"}
          </div>

          <div className="toolbar-buttons">
            {text && (
              <button type="button" className="button secondary" onClick={clear}>
                Clear
              </button>
            )}
            <button
              type="button"
              className="button primary"
              onClick={() => run(text)}
              disabled={!ready || !text.trim()}
            >
              Analyze with 1D CNN
            </button>
          </div>
        </div>

        <div className="sample-prompts-row">
          <span className="sample-lbl">Try sample review:</span>
          <div className="sample-chips">
            {CNN_EXAMPLES.map((ex) => (
              <button
                key={ex.id}
                type="button"
                className="chip-btn"
                onClick={() => pickExample(ex)}
                disabled={!ready}
              >
                {ex.name}
              </button>
            ))}
          </div>
        </div>
      </section>

      {analysis && !analysis.empty && (
        <>
          <section className="results-grid" aria-label="1D CNN Analysis Results">
            {analysis.results.map((res) => (
              <ModelCard key={res.id} {...res} />
            ))}
          </section>

          {analysis.disagree && (
            <div className="callout warning">
              <svg viewBox="0 0 20 20" width="18" height="18" fill="currentColor">
                <path
                  fillRule="evenodd"
                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
              <span>
                <strong>Models Disagree:</strong> 1D CNN and GRU produced contrasting predictions for this input.
                1D CNN focuses on local 5-gram keyphrase patterns while GRU unrolls recurrent memory across the whole
                review.
              </span>
            </div>
          )}

          <TokenView analysis={analysis} />
        </>
      )}

      <div className="cnn-quick-benchmark card">
        <div className="cnn-bench-header">
          <h3>1D CNN Benchmark Snapshot (Dilmith)</h3>
          <span className="status-badge ready">94.38% Test Accuracy</span>
        </div>
        <p className="muted small">
          Trained on the shared 80/10/10 split under SE4050 guidelines. 128 Conv1D filters (kernel 5) with Global Max
          Pooling.
        </p>
        <div className="cnn-bench-metrics">
          <div>
            <strong>Accuracy:</strong> 94.38% (+6.25% vs GRU)
          </div>
          <div>
            <strong>F1 Score:</strong> 0.9455
          </div>
          <div>
            <strong>Positive Recall:</strong> 97.50% (78/80)
          </div>
          <div>
            <strong>ROC-AUC:</strong> 0.9702
          </div>
          <div>
            <strong>Latency:</strong> ~4 ms in browser
          </div>
          <div>
            <strong>Speed:</strong> 2.2× faster training
          </div>
        </div>
      </div>
    </div>
  );
}
