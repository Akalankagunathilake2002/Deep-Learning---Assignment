import { useMemo, useRef, useState, useEffect } from "react";
import examples from "../examples.json";
import { formatPercent, plural } from "../format.js";
import { analyzeCnn } from "../lib/engine.js";
import { Cnn1dModel } from "../lib/cnn1d.js";
import ModelCard from "./ModelCard.jsx";
import TokenView from "./TokenView.jsx";

function Notice({ children }) {
  return (
    <p className="notice">
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false">
        <path d="M12 2a10 10 0 100 20 10 10 0 000-20zm1 15h-2v-6h2zm0-8h-2V7h2z" fill="currentColor" />
      </svg>
      <span>{children}</span>
    </p>
  );
}

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
      <section className="card input-card" aria-labelledby="input-title">
        <h2 id="input-title">Your review</h2>
        <p className="muted">Paste or type a hotel review. The models say whether it sounds positive or negative.</p>

        <label htmlFor="review" className="sr-only">
          Hotel review
        </label>
        <textarea
          id="review"
          ref={textarea}
          rows={9}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(text);
          }}
          placeholder="For example: We stayed three nights. The room was clean and quiet, the staff were friendly, and breakfast was great…"
          disabled={!ready}
        />
        <div className="input-meta">
          <span className="small muted">
            {plural(words, "word")} · a few sentences (40+ words) work best
          </span>
        </div>

        <div className="actions">
          <button type="button" className="button primary" onClick={() => run(text)} disabled={!ready || !text.trim()}>
            Analyze review
          </button>
          <button type="button" className="button" onClick={clear} disabled={!text && !analysis}>
            Clear
          </button>
          <span className="small muted hint-keys">Ctrl/⌘ + Enter</span>
        </div>

        <div className="examples">
          <h3>Or try an example</h3>
          <ul className="chips">
            {examples.map((ex) => (
              <li key={ex.id}>
                <button type="button" className="chip" title={ex.hint} onClick={() => pickExample(ex)} disabled={!ready}>
                  {ex.label}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="results-col" aria-labelledby="result-title">
        <h2 id="result-title" className="sr-only">
          Result
        </h2>

        {ready && !analysis && (
          <div className="card placeholder">
            <strong>Your result will appear here.</strong>
            <span>Type a review or pick an example, then press Analyze.</span>
            <span className="small muted">
              The <b>1D CNN model</b> applies parallel 1D convolutional filters across word vector embeddings to detect localized sentiment keyphrases with zero latency.
            </span>
          </div>
        )}

        {ready && analysis && (
          <>
            {analysis.sourceText !== text && (
              <Notice>You have edited the text since this result. Press Analyze to update it.</Notice>
            )}

            {analysis.empty && (
              <Notice>No usable words were found after cleaning. Try a sentence that contains some letters.</Notice>
            )}

            {!analysis.empty && (
              <>
                <section className="results-grid" aria-label="1D CNN Analysis Results">
                  {analysis.results.map((res) => (
                    <ModelCard key={res.id} {...res} />
                  ))}
                </section>

                <TokenView analysis={analysis} />
              </>
            )}
          </>
        )}

        <div className="cnn-quick-benchmark card">
          <div className="cnn-bench-header">
            <h3>1D CNN Benchmark Snapshot</h3>
            <span className="status-badge ready">94.38% Test Accuracy</span>
          </div>
          <p className="muted small">
            Trained on the common dataset (<code>dataset/deceptive-opinion.csv</code>) with the fixed 80/10/10 split under SE4050 guidelines. 128 Conv1D filters (kernel 5) with Global Max Pooling.
          </p>
          <div className="cnn-bench-metrics">
            <div>
              <strong>Accuracy:</strong> 94.38%
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
              <strong>Training Time:</strong> 5.92 s
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
