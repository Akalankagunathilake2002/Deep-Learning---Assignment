import { useState } from "react";
import CnnTester from "./CnnTester.jsx";

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
        <span className="logo cnn-logo" aria-hidden="true">
          <svg viewBox="0 0 32 32" width="40" height="40" fill="none">
            <rect width="32" height="32" rx="8" fill="rgba(42, 120, 214, 0.12)" stroke="var(--pos)" strokeWidth="1.6" />
            <path
              d="M7 16h3l2.5-6 4 12 3.5-8 2 4h3"
              stroke="var(--pos)"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        <div>
          <div className="header-badge-row">
            <h1>1D CNN</h1>
            <span className="status-badge active">Benchmark Champion</span>
            <span className="dev-arch-tag">
              <code>Conv1D(64, kernel_size=3)</code>
            </span>
          </div>
          <p className="muted">
            Temporal 1D convolutional feature extractor with global max-pooling · Implemented by Dilmith (SE4050)
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
          Try 1D CNN Live (In-Browser)
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
          Benchmark & Evaluation Report
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
          <div className="card cnn-benchmark-detail-card">
            <h2>1D CNN Empirical Benchmark (Dilmith)</h2>
            <p className="muted">
              Evaluated under the standardized SE4050 protocol: 1,596 reviews, fixed 80/10/10 split, seed 42.
            </p>

            <div className="cnn-metrics-grid">
              <div className="cnn-metric-box highlight">
                <span className="cnn-metric-val">{info.metrics.accuracy}</span>
                <span className="cnn-metric-lbl">Test Accuracy</span>
                <span className="cnn-metric-cmp">+6.25% vs GRU</span>
              </div>
              <div className="cnn-metric-box highlight">
                <span className="cnn-metric-val">{info.metrics.f1}</span>
                <span className="cnn-metric-lbl">Positive F1</span>
                <span className="cnn-metric-cmp">+0.0697 vs GRU</span>
              </div>
              <div className="cnn-metric-box">
                <span className="cnn-metric-val">{info.metrics.recall}</span>
                <span className="cnn-metric-lbl">Positive Recall</span>
                <span className="cnn-metric-sub">78 of 80 detected</span>
              </div>
              <div className="cnn-metric-box">
                <span className="cnn-metric-val">{info.metrics.rocAuc}</span>
                <span className="cnn-metric-lbl">ROC-AUC</span>
                <span className="cnn-metric-sub">0.940 - 0.994 (95% CI)</span>
              </div>
              <div className="cnn-metric-box">
                <span className="cnn-metric-val">{info.metrics.trainingTime}</span>
                <span className="cnn-metric-lbl">Training Speed</span>
                <span className="cnn-metric-cmp">2.2× faster than GRU</span>
              </div>
              <div className="cnn-metric-box">
                <span className="cnn-metric-val">{info.metrics.params}</span>
                <span className="cnn-metric-lbl">Parameters</span>
                <span className="cnn-metric-sub">100% Trainable</span>
              </div>
            </div>

            <div className="cnn-cm-summary">
              <strong>Holdout Test Confusion Matrix (N=160):</strong> TN={info.metrics.confusion.tn} | FP={info.metrics.confusion.fp} | FN={info.metrics.confusion.fn} | TP={info.metrics.confusion.tp}
            </div>

            <div className="cnn-architecture-banner">
              <span className="cnn-arch-title">Champion Architecture Pipeline:</span>
              <code>{info.fullArch}</code>
            </div>

            <div className="cnn-figures-row">
              <div className="figure-card">
                <h4>Training & Validation Accuracy</h4>
                <img src="./figures/cnn1d_accuracy_curve.png" alt="1D CNN Accuracy Curve" />
              </div>
              <div className="figure-card">
                <h4>Training & Validation Loss</h4>
                <img src="./figures/cnn1d_loss_curve.png" alt="1D CNN Loss Curve" />
              </div>
              <div className="figure-card">
                <h4>Confusion Matrix (Test Set)</h4>
                <img src="./figures/cnn1d_confusion_matrix.png" alt="1D CNN Confusion Matrix" />
              </div>
              <div className="figure-card">
                <h4>ROC Curve (AUC = 0.970)</h4>
                <img src="./figures/cnn1d_roc_curve.png" alt="1D CNN ROC Curve" />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
