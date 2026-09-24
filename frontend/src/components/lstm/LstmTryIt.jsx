import { useEffect, useMemo, useRef, useState } from "react";
import examples from "../../examples.json";
import { formatPercent, plural } from "../../format.js";
import { loadLstmModel } from "../../lib/engine.js";
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

let cachedLstmModel = null;
let lstmLoadPromise = null;

function getOrLoadLstm() {
  if (cachedLstmModel) return Promise.resolve(cachedLstmModel);
  if (!lstmLoadPromise) {
    lstmLoadPromise = loadLstmModel()
      .then((m) => {
        cachedLstmModel = m;
        return m;
      })
      .catch((err) => {
        lstmLoadPromise = null;
        throw err;
      });
  }
  return lstmLoadPromise;
}

export default function LstmTryIt({ state }) {
  const [text, setText] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [lstmModel, setLstmModel] = useState(() => cachedLstmModel);
  const [loadingModel, setLoadingModel] = useState(() => !cachedLstmModel);
  const [loadError, setLoadError] = useState(null);
  const textarea = useRef(null);

  const words = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text]);
  const config = state.status === "ready" ? state.engine?.config : null;
  const vocab = state.status === "ready" ? state.engine?.vocab : null;

  useEffect(() => {
    let active = true;
    if (!cachedLstmModel) {
      setLoadingModel(true);
      getOrLoadLstm()
        .then((m) => {
          if (active) {
            setLstmModel(m);
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
      setLstmModel(cachedLstmModel);
      setLoadingModel(false);
    }
    return () => {
      active = false;
    };
  }, []);

  const run = (value) => {
    if (!lstmModel || !vocab || !config || !value.trim()) return;
    const prep = prepareInput(value, vocab, config);
    const empty = prep.nWords === 0;
    let p = 0.5;
    let label = "Negative";

    if (!empty) {
      p = lstmModel.predict(prep.input);
      label = p >= config.threshold ? "Positive" : "Negative";
    }

    setAnalysis({
      ...prep,
      empty,
      p,
      label,
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

  const ready = !loadingModel && Boolean(lstmModel);

  return (
    <div className="try-grid">
      <section className="card input-card" aria-labelledby="input-title">
        <h2 id="input-title">Your review</h2>
        <p className="muted">Paste or type a hotel review. The LSTM model says whether it sounds positive or negative.</p>

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

        {loadingModel && (
          <div className="card placeholder" role="status">
            <span className="spinner" aria-hidden="true" />
            Loading the LSTM model (about 1.4 MB)…
          </div>
        )}

        {loadError && (
          <div className="card placeholder error" role="alert">
            <strong>The LSTM model could not be loaded.</strong>
            <span>{loadError}</span>
          </div>
        )}

        {ready && !analysis && (
          <div className="card placeholder">
            <strong>Your result will appear here.</strong>
            <span>Type a review or pick an example, then press Analyze.</span>
            <span className="small muted">
              The <b>LSTM model</b> uses a 4-gate recurrent architecture with an additive cell state to capture long-range sentiment context.
            </span>
          </div>
        )}

        {ready && analysis && (
          <>
            {analysis.sourceText !== text && <Notice>You have edited the text since this result. Press Analyze to update it.</Notice>}

            {analysis.empty && <Notice>No usable words were found after cleaning. Try a sentence that contains some letters.</Notice>}

            {analysis.truncated && <Notice>Only the first {config.maxLen} words are used.</Notice>}

            {!analysis.empty && (
              <div className="model-grid">
                <article
                  className={`model-card ${analysis.label === "Positive" ? "is-positive" : "is-negative"}`}
                  aria-label={`LSTM: ${analysis.label}`}
                >
                  <header className="model-card-head">
                    <div>
                      <h3>LSTM Model</h3>
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

                  <p className="raw small">P(positive) = {analysis.p.toFixed(3)}</p>
                </article>
              </div>
            )}

            {!analysis.empty && <TokenView analysis={analysis} maxLen={config.maxLen} />}

            <p className="footnote small muted">
              These probabilities are over-confident, so 99% here does not mean 99% sure. On 160 test reviews the model had never seen, the LSTM was right 93% of the time.
            </p>
          </>
        )}
      </section>
    </div>
  );
}
