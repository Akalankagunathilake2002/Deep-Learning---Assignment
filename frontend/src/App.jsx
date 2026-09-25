import { useState } from "react";
import LstmResultsTab from "./components/lstm/LstmResultsTab.jsx";
import LstmTryIt from "./components/lstm/LstmTryIt.jsx";
import ModelProfile from "./components/ModelProfile.jsx";
import ResultsTab from "./components/ResultsTab.jsx";
import Sidebar from "./components/Sidebar.jsx";
import TryIt from "./components/TryIt.jsx";
import { useEngine } from "./useEngine.js";

const TABS = [
  { id: "try", label: "Try it" },
  { id: "results", label: "Results & about" },
];

export default function App({ load }) {
  const state = useEngine(load);
  const [activeModel, setActiveModel] = useState("gru");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [tab, setTab] = useState("try");

  // Left/right arrow keys move between the tabs
  const onKeyDown = (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = TABS.findIndex((t) => t.id === tab);
    const next = TABS[(i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
    setTab(next.id);
    document.getElementById(`tab-${next.id}`)?.focus();
  };

  return (
    <div className="layout-root">
      {/* Mobile Top Bar */}
      <div className="mobile-header">
        <button
          type="button"
          className="mobile-nav-toggle"
          onClick={() => setSidebarOpen((prev) => !prev)}
          aria-label={sidebarOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={sidebarOpen}
        >
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2.2">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        <span className="mobile-title">Hotel Review Sentiment</span>
        <span className="status-badge active">{activeModel.toUpperCase()}</span>
      </div>

      <Sidebar
        activeModel={activeModel}
        onSelectModel={(id) => setActiveModel(id)}
        isOpen={sidebarOpen}
        onToggleOpen={() => setSidebarOpen((prev) => !prev)}
      />

      <div className="main-content-wrapper">
        <div className="app">
          {activeModel === "gru" ? (
            <>
              <header className="header">
                <span className="logo" aria-hidden="true">
                  <svg viewBox="0 0 32 32" width="40" height="40">
                    <rect width="32" height="32" rx="8" fill="currentColor" />
                    <path
                      d="M8 21l5-6 4 4 7-9"
                      fill="none"
                      stroke="#fff"
                      strokeWidth="2.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <circle cx="24" cy="10" r="2" fill="#eb6834" />
                  </svg>
                </span>
                <div>
                  <div className="header-badge-row">
                    <h1>Hotel review sentiment</h1>
                    <span className="status-badge active">GRU Reference Model</span>
                  </div>
                  <p className="muted">
                    A GRU deep-learning model that reads a review and predicts positive or negative
                  </p>
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
            </>
          ) : activeModel === "lstm" ? (
            <>
              <header className="header">
                <span className="logo" aria-hidden="true">
                  <svg viewBox="0 0 32 32" width="40" height="40">
                    <rect width="32" height="32" rx="8" fill="currentColor" />
                    <path
                      d="M8 21l5-6 4 4 7-9"
                      fill="none"
                      stroke="#fff"
                      strokeWidth="2.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <circle cx="24" cy="10" r="2" fill="#eb6834" />
                  </svg>
                </span>
                <div>
                  <div className="header-badge-row">
                    <h1>Hotel review sentiment</h1>
                    <span className="status-badge active">LSTM Model</span>
                  </div>
                  <p className="muted">
                    An LSTM deep-learning model that reads a review and predicts positive or negative
                  </p>
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
                  <LstmTryIt state={state} />
                </div>
                <div role="tabpanel" id="panel-results" aria-labelledby="tab-results" hidden={tab !== "results"}>
                  <LstmResultsTab state={state} />
                </div>
              </main>

              <footer className="footer small muted">
                SE4050 Deep Learning 2026 · LSTM individual contribution. The model runs entirely in your browser: nothing you type is sent anywhere.
              </footer>
            </>
          ) : (
            <>
              <ModelProfile modelId={activeModel} state={state} onSelectGru={() => setActiveModel("gru")} />
              <footer className="footer small muted">
                {activeModel === "cnn1d"
                  ? "SE4050 Deep Learning 2026 · 1D CNN individual contribution. The model runs entirely in your browser: nothing you type is sent anywhere."
                  : activeModel === "rnn"
                  ? "SE4050 Deep Learning 2026 · Simple RNN individual contribution. The model runs entirely in your browser: nothing you type is sent anywhere."
                  : "SE4050 Deep Learning 2026 · Individual contribution. The model runs entirely in your browser: nothing you type is sent anywhere."}
              </footer>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
