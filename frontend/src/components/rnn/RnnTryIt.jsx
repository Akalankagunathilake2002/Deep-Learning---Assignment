import { useEffect, useMemo, useRef, useState } from "react";
import examples from "../../examples.json";
import { formatPercent, plural } from "../../format.js";
import { loadRnnModel } from "../../lib/engine.js";
import { prepareInput } from "../../lib/pipeline.js";
import TokenView from "../TokenView.jsx";

function Arrow({ up }) {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">
      <path d={up ? "M12 4l8 9h-5v7H9v-7H4z" : "M12 20l-8-9h5V4h6v7h5z"} fill="currentColor" />
    </svg>
  );
}

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

let cachedRnnModel = null;
let rnnLoadPromise = null;

function getOrLoadRnn() {
  if (cachedRnnModel) return Promise.resolve(cachedRnnModel);
  if (!rnnLoadPromise) {
    rnnLoadPromise = loadRnnModel()
      .then((m) => {
        cachedRnnModel = m;
        return m;
      })
      .catch((err) => {
        rnnLoadPromise = null;
        throw err;
      });
  }
  return rnnLoadPromise;
}

export default function RnnTryIt({ state }) {
  const [text, setText] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [rnnModel, setRnnModel] = useState(() => cachedRnnModel);
  const [loadingModel, setLoadingModel] = useState(() => !cachedRnnModel);
  const [loadError, setLoadError] = useState(null);
  const textarea = useRef(null);

  const words = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text]);
  const config = state.status === "ready" ? state.engine?.config : null;
  const vocab = state.status === "ready" ? state.engine?.vocab : null;

  useEffect(() => {
    let active = true;
    if (!cachedRnnModel) {
      setLoadingModel(true);
      getOrLoadRnn()
        .then((m) => {
          if (active) {
            setRnnModel(m);
            setLoadingModel(false);
          }
        })
        .catch((err) => {
          if (active) {
            setLoadError(err.message || "Failed to load model");
            setLoadingModel(false);
          }
        });
    } else {
      setRnnModel(cachedRnnModel);
      setLoadingModel(false);
    }
    return () => {
      active = false;
    };
  }, []);

  const run = (value) => {
    if (!rnnModel || !vocab || !config || !value.trim()) return;
    const prep = prepareInput(value, vocab, config);
    const empty = prep.nWords === 0;
    let p = 0.5;
    let label = "Negative";

    if (!empty) {
      p = rnnModel.predict(prep.input);
      label = p >= config.threshold ? "Positive" : "Negative";
    }

    const confidence = p >= 0.5 ? p : 1 - p;

    setAnalysis({
      ...prep,
      empty,
      p,
      label,
      confidence,
      sourceText: value,
    });
  };

  const pickExample = (example) => {
    setText(example.text);
    run(example.text);
    textarea.current?.focus();
  };

  const clear = () => {
    setText("");
    setAnalysis(null);
    textarea.current?.focus();
  };

  const ready = !loadingModel && Boolean(rnnModel);

  return (
    <div className="try-grid">
      <section className="card input-card" aria-labelledby="input-title">
        <h2 id="input-title">Your review</h2>
        <p className="muted">
          Paste or type a hotel review. The Simple RNN model says whether it sounds positive or negative.
        </p>

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
        />
        <div className="input-meta">
          <span className="small muted">
            {plural(words, "word")} · a few sentences work best
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

        {loadingModel && (
          <div className="card placeholder" role="status">
            <span className="spinner" aria-hidden="true" />
            Loading the Simple RNN model (about 1.3 MB)…
          </div>
        )}

        {loadError && (
          <div className="card placeholder error" role="alert">
            <strong>The Simple RNN model could not be loaded.</strong>
            <span>{loadError}</span>
          </div>
        )}

        {ready && !analysis && (
          <div className="card placeholder">
            <strong>Your result will appear here.</strong>
            <span>Type a review or pick an example, then press Analyze.</span>
            <span className="small muted">
              The <b>Simple RNN model</b> uses an un-gated Elman recurrence with hidden state dimension 64 to classify review sentiment.
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

            {analysis.truncated && <Notice>Only the first {config.maxLen} words are used.</Notice>}

            {!analysis.empty && (
              <div className="model-grid">
                <article
                  className={`model-card ${analysis.label === "Positive" ? "is-positive" : "is-negative"}`}
                  aria-label={`Simple RNN: ${analysis.label}`}
                >
                  <header className="model-card-head">
                    <div>
                      <h3>Simple RNN Model</h3>
                      <p className="muted small">Trained on the original reviews</p>
                    </div>
                    <span className={`badge ${analysis.label === "Positive" ? "badge-positive" : "badge-negative"}`}>
                      <Arrow up={analysis.label === "Positive"} />
                      {analysis.label}
                    </span>
                  </header>

                  <p className="big-number">
                    {formatPercent(analysis.p)}
                    <span>chance this review is positive</span>
                  </p>

                  <div
                    className="meter"
                    role="meter"
                    aria-label="Probability that the review is positive"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(analysis.p * 100)}
                    aria-valuetext={`${(analysis.p * 100).toFixed(1)} percent`}
                  >
                    <div className="meter-fill" style={{ width: `${analysis.p * 100}%` }} />
                    <div className="meter-tick" />
                  </div>
                  <div className="meter-scale" aria-hidden="true">
                    <span>0%</span>
                    <span>50% cut-off</span>
                    <span>100%</span>
                  </div>

                  <div className="meta-row small muted" style={{ display: "flex", justifyContent: "space-between", marginTop: "8px" }}>
                    <span>P(positive) = {analysis.p.toFixed(3)}</span>
                    <span>Confidence: {(analysis.confidence * 100).toFixed(1)}%</span>
                  </div>
                </article>
              </div>
            )}

            {!analysis.empty && <TokenView analysis={analysis} maxLen={config.maxLen} />}

            <p className="footnote small muted">
              On 160 holdout test reviews the model had never seen during training, Simple RNN achieved {state.results?.rnn ? `${(state.results.rnn.accuracy * 100).toFixed(1)}%` : "95.0%"} test accuracy and {state.results?.rnn ? state.results.rnn.roc_auc.toFixed(4) : "0.9795"} ROC-AUC.
            </p>
          </>
        )}
      </section>
    </div>
  );
}
