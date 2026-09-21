export const MODELS = [
  {
    id: "rnn",
    shortName: "RNN",
    fullName: "Simple RNN",
    category: "Recurrent",
    desc: "Vanilla recurrent network with un-gated hidden state",
    status: "pending",
    badge: "In Dev",
  },
  {
    id: "lstm",
    shortName: "LSTM",
    fullName: "LSTM",
    category: "Recurrent",
    desc: "Long Short-Term Memory with 3 gates and cell state",
    status: "pending",
    badge: "In Dev",
  },
  {
    id: "gru",
    shortName: "GRU",
    fullName: "GRU (Reference)",
    category: "Recurrent",
    desc: "Gated Recurrent Unit with reset and update gates",
    status: "active",
    badge: "Active",
  },
  {
    id: "cnn1d",
    shortName: "1D CNN",
    fullName: "1D CNN",
    category: "Convolutional",
    desc: "Temporal 1D convolutions with global max pooling",
    status: "pending",
    badge: "In Dev",
  },
];

export default function Sidebar({ activeModel, onSelectModel, isOpen, onToggleOpen }) {
  return (
    <>
      {/* Mobile backdrop overlay */}
      {isOpen && <div className="sidebar-backdrop" onClick={onToggleOpen} aria-hidden="true" />}

      <aside className={`sidebar ${isOpen ? "open" : ""}`} aria-label="Model Architectures">
        <div className="sidebar-brand">
          <div className="sidebar-logo" aria-hidden="true">
            <svg viewBox="0 0 32 32" width="28" height="28">
              <rect width="32" height="32" rx="8" fill="currentColor" />
              <path d="M7 16h6m6 0h6M13 16l3-6 3 6m-6 0l3 6 3-6" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <span className="sidebar-title">Deep Learning</span>
            <span className="sidebar-subtitle">Sentiment Comparison</span>
          </div>
        </div>

        <div className="sidebar-section-title">Models & Architectures</div>

        <nav className="sidebar-nav" aria-label="Model selector">
          {MODELS.map((m) => {
            const isSelected = activeModel === m.id;
            return (
              <button
                key={m.id}
                id={`model-btn-${m.id}`}
                type="button"
                className={`sidebar-model-btn ${isSelected ? "active" : ""} ${m.status === "active" ? "is-ready" : ""}`}
                onClick={() => {
                  onSelectModel(m.id);
                  if (isOpen) onToggleOpen();
                }}
                aria-current={isSelected ? "page" : undefined}
              >
                <div className="sidebar-model-icon" aria-hidden="true">
                  <span>{m.shortName.slice(0, 3)}</span>
                </div>
                <div className="sidebar-model-info">
                  <div className="sidebar-model-name-row">
                    <span className="sidebar-model-name">{m.shortName}</span>
                    <span className={`status-badge ${m.status}`}>{m.badge}</span>
                  </div>
                  <span className="sidebar-model-sub">{m.fullName}</span>
                </div>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer card">
          <div className="sidebar-footer-header">
            <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true">
              <path d="M8 0a8 8 0 100 16A8 8 0 008 0zm1 12H7V7h2v5zm0-6H7V4h2v2z" />
            </svg>
            <strong>Fair Benchmark</strong>
          </div>
          <ul className="sidebar-meta-list">
            <li><span>Dataset:</span> 1,596 reviews (balanced)</li>
            <li><span>Split:</span> 1,276 / 160 / 160 (fixed)</li>
            <li><span>Vocab:</span> 5,000 words</li>
            <li><span>Max Length:</span> 300 words (pre-pad)</li>
            <li><span>Embedding:</span> 64-dim (from scratch)</li>
          </ul>
        </div>
      </aside>
    </>
  );
}
