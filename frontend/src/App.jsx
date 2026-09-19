import { useState } from "react";
import ResultsTab from "./components/ResultsTab.jsx";
import TryIt from "./components/TryIt.jsx";
import { useEngine } from "./useEngine.js";

const TABS = [
  { id: "try", label: "Try it" },
  { id: "results", label: "Results & about" },
];

export default function App({ load }) {
  const state = useEngine(load);
  const [tab, setTab] = useState("try");

  // Left/right arrow keys move between the tabs, as in a standard tab list.
  const onKeyDown = (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = TABS.findIndex((t) => t.id === tab);
    const next = TABS[(i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
    setTab(next.id);
    document.getElementById(`tab-${next.id}`)?.focus();
  };

  return (
    <div className="app">
      <header className="header">
        <span className="logo" aria-hidden="true">
          <svg viewBox="0 0 32 32" width="40" height="40">
            <rect width="32" height="32" rx="8" fill="currentColor" />
            <path d="M8 21l5-6 4 4 7-9" fill="none" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="24" cy="10" r="2" fill="#eb6834" />
          </svg>
        </span>
        <div>
          <h1>Hotel review sentiment</h1>
          <p className="muted">A GRU deep-learning model that reads a review and says positive or negative</p>
        </div>
      </header>

      <div role="tablist" aria-label="Sections" className="tabs" onKeyDown={onKeyDown}>
        {TABS.map((t) => (
          <button
            key={t.id}
            id={`tab-${t.id}`}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            tabIndex={tab === t.id ? 0 : -1}
            className="tab"
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <main>
        <div role="tabpanel" id="panel-try" aria-labelledby="tab-try" hidden={tab !== "try"}>
          <TryIt state={state} />
        </div>
        <div role="tabpanel" id="panel-results" aria-labelledby="tab-results" hidden={tab !== "results"}>
          <ResultsTab state={state} />
        </div>
      </main>

      <footer className="footer small muted">
        SE4050 Deep Learning 2026 · GRU individual contribution. The model runs entirely in your browser: nothing you type is sent anywhere.
      </footer>
    </div>
  );
}
