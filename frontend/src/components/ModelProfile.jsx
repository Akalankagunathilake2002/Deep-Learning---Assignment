const MODEL_INFO = {
  rnn: {
    name: "Simple RNN",
    arch: "SimpleRNN(64)",
    desc: "Vanilla recurrent neural network architecture for sequence classification.",
  },
  lstm: {
    name: "LSTM",
    arch: "LSTM(64)",
    desc: "Long Short-Term Memory network with gating mechanisms and cell state.",
  },
  cnn1d: {
    name: "1D CNN",
    arch: "Conv1D(64, kernel_size=3)",
    desc: "Temporal 1D convolutional feature extractor with global pooling.",
  },
};

export default function ModelProfile({ modelId, onSelectGru }) {
  const info = MODEL_INFO[modelId] || {
    name: "Model",
    arch: "Neural Network",
    desc: "Sequence classification model.",
  };

  return (
    <div className="dev-card card">
      <div className="dev-badge-wrap">
        <span className="status-badge pending">Under Development</span>
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
        <button type="button" className="button primary" onClick={onSelectGru}>
          Try Live GRU Model →
        </button>
      </div>
    </div>
  );
}
