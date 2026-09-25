import { useState } from "react";
import CnnTester from "./CnnTester.jsx";
import CnnResultsTab from "./CnnResultsTab.jsx";

const MODEL_INFO = {
  rnn: {
    name: "Simple RNN",
    arch: "SimpleRNN(64)",
    desc: "Vanilla recurrent neural network architecture for sequence classification.",
    status: "pending",
    badge: "Under Development",
  },
  lstm: {
    name: "LSTM",
    arch: "LSTM(64)",
    desc: "Long Short-Term Memory network with gating mechanisms and cell state.",
    status: "pending",
    badge: "Under Development",
  },
  cnn1d: {
    name: "1D CNN",
    arch: "Conv1D(64, kernel_size=3)",
    fullArch: "Input(300) → Embedding(5000, 64) → Conv1D(128, k=5, ReLU) → GlobalMaxPooling1D() → Dense(32, ReLU) → Dense(1, Sigmoid)",
    desc: "Temporal 1D convolutional feature extractor with global max-pooling for position-invariant sentiment detection.",
    status: "active",
    badge: "Benchmark Champion",
    author: "Dilmith",
    metrics: {
      accuracy: "94.38%",
      f1: "0.9455",
      recall: "97.50%",
      precision: "91.76%",
      rocAuc: "0.9702",
      trainingTime: "6.11 s",
      params: "365,249",
      epochs: "13 (best)",
      confusion: { tn: 73, fp: 7, fn: 2, tp: 78 },
    },
  },
};

export default function ModelProfile({ modelId, state, onSelectGru }) {
  const [activeTab, setActiveTab] = useState("try"); // "try" or "benchmark"

  const info = MODEL_INFO[modelId] || {
    name: "Model",
    arch: "Neural Network",
    desc: "Sequence classification model.",
    status: "pending",
    badge: "Under Development",
  };

  const isCnn = modelId === "cnn1d";

  // For models under development (RNN, LSTM), show the centered development card
  if (!isCnn) {
    return (
      <div className="dev-card card">
        <div className="dev-badge-wrap">
          <span className="status-badge pending">{info.badge}</span>
        </div>

        <div className="dev-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
          </svg>
        </div>

        <h1>{info.name}</h1>
        <p className="dev-arch-tag">
          <code>{info.arch}</code>
        </p>

        <p className="dev-message muted">
          This model is currently under development and training. Once ready, live browser inference and evaluation
          metrics will be available here.
        </p>

        <div className="dev-action">
          <button type="button" className="button secondary" onClick={onSelectGru}>
            Try Live GRU Model →
          </button>
        </div>
      </div>
    );
  }

  // 1D CNN View: Clean top-level header and tab navigation (matching GRU page layout)
  return (
    <div className="cnn-profile-wrapper">
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
            <span className="status-badge active">1D CNN Model</span>
          </div>
          <p className="muted">
            A 1D CNN deep-learning model that reads a review and predicts positive or negative
          </p>
        </div>
      </header>

      {/* Page Tabs */}
      <div role="tablist" aria-label="1D CNN Sections" className="tabs">
        <button
          type="button"
          role="tab"
          id="tab-cnn-try"
          aria-selected={activeTab === "try"}
          aria-controls="panel-cnn-try"
          tabIndex={activeTab === "try" ? 0 : -1}
          className="tab"
          onClick={() => setActiveTab("try")}
        >
          Try it
        </button>
        <button
          type="button"
          role="tab"
          id="tab-cnn-benchmark"
          aria-selected={activeTab === "benchmark"}
          aria-controls="panel-cnn-benchmark"
          tabIndex={activeTab === "benchmark" ? 0 : -1}
          className="tab"
          onClick={() => setActiveTab("benchmark")}
        >
          Results & about
        </button>
      </div>

      {/* Tab Panels */}
      <main>
        <div role="tabpanel" id="panel-cnn-try" aria-labelledby="tab-cnn-try" hidden={activeTab !== "try"}>
          <CnnTester state={state || { status: "ready" }} onSelectGru={onSelectGru} />
        </div>

        <div
          role="tabpanel"
          id="panel-cnn-benchmark"
          aria-labelledby="tab-cnn-benchmark"
          hidden={activeTab !== "benchmark"}
        >
          <CnnResultsTab state={state} />
        </div>
      </main>
    </div>
  );
}
