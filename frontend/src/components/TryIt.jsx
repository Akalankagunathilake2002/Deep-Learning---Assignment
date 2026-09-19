import { useMemo, useRef, useState } from "react";
import examples from "../examples.json";
import { formatPercent, pct, plural } from "../format.js";
import { analyze } from "../lib/engine.js";
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

export default function TryIt({ state }) {
  const [text, setText] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const textarea = useRef(null);
  const ready = state.status === "ready";
  const words = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text]);

  const run = (value) => {
    if (!ready || !value.trim()) return;
    setAnalysis({ ...analyze(value, state.engine), sourceText: value });
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

  // A short spoken summary for screen readers; the visual result is not announced in full.
  const summary = !analysis
    ? ""
    : analysis.empty
      ? "No usable words found."
      : analysis.results.map((r) => `${r.name}: ${r.label}, ${formatPercent(r.p)} chance of positive.`).join(" ");

  const config = ready ? state.engine.config : null;
  const results = state.status === "ready" ? state.results : null;

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
        <p className="sr-only" role="status">
          {summary}
        </p>

        {state.status === "loading" && (
          <div className="card placeholder" role="status">
            <span className="spinner" aria-hidden="true" />
            Loading the models (about 3 MB)…
          </div>
        )}

        {state.status === "error" && (
          <div className="card placeholder error" role="alert">
            <strong>The models could not be loaded.</strong>
            <span>{state.message}</span>
          </div>
        )}

        {ready && !analysis && (
          <div className="card placeholder">
            <strong>Your result will appear here.</strong>
            <span>Type a review or pick an example, then press Analyze.</span>
            <span className="small muted">
              You get two answers. The <b>main GRU</b> was trained on the original reviews. <b>GRU + short examples</b> is an optional
              extension that also saw one- and two-sentence chunks.
            </span>
          </div>
        )}

        {ready && analysis && (
          <>
            {analysis.sourceText !== text && <Notice>You have edited the text since this result. Press Analyze to update it.</Notice>}

            {analysis.empty && <Notice>No usable words were found after cleaning. Try a sentence that contains some letters.</Notice>}

            {analysis.short && (
              <Notice>
                <b>Short review.</b> This has {plural(analysis.nWords, "word")}, and the shortest original training review has{" "}
                {config.minTrainWords}. The main GRU tends to call very short reviews positive, so lean on the second model here (it was
                also trained on short examples). Either can still be wrong.
              </Notice>
            )}
            {analysis.disagree && <Notice>The two models disagree. That usually happens with short or mixed reviews, so treat this result with care.</Notice>}
            {analysis.truncated && <Notice>Only the first {config.maxLen} words are used.</Notice>}

            {analysis.results.length > 0 && (
              <div className="model-grid">
                {analysis.results.map((r) => (
                  <ModelCard key={r.id} {...r} />
                ))}
              </div>
            )}

            {!analysis.empty && <TokenView analysis={analysis} maxLen={config.maxLen} />}

            {results && (
              <p className="footnote small muted">
                These probabilities are over-confident, so 99% here does not mean 99% sure. On {results.split.test} test reviews the model
                had never seen, the main GRU was right {pct(results.main.accuracy, 0)} of the time.
              </p>
            )}
          </>
        )}
      </section>
    </div>
  );
}
