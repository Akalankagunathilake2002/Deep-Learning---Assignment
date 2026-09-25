import { formatPercent } from "../format.js";

const DESCRIPTION = {
  main: "Trained on the original reviews (Reference)",
  short: "Also trained on short examples",
  cnn1d: "Temporal convolutions with global pooling",
};

function Arrow({ up }) {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false">
      <path d={up ? "M12 4l8 9h-5v7H9v-7H4z" : "M12 20l-8-9h5V4h6v7h5z"} fill="currentColor" />
    </svg>
  );
}

/** One model's answer: verdict, probability, and a meter with the 50% cut-off marked. */
export default function ModelCard({ id, name, p, label }) {
  const positive = label === "Positive";
  return (
    <article className={`model-card ${positive ? "is-positive" : "is-negative"}`} aria-label={`${name}: ${label}`}>
      <header className="model-card-head">
        <div>
          <h3>{name}</h3>
          <p className="muted small">{DESCRIPTION[id]}</p>
        </div>
        <span className={`badge ${positive ? "badge-positive" : "badge-negative"}`}>
          <Arrow up={positive} />
          {label}
        </span>
      </header>

      <p className="big-number">
        {formatPercent(p)}
        <span>chance this review is positive</span>
      </p>

      <div
        className="meter"
        role="meter"
        aria-label="Probability that the review is positive"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(p * 100)}
        aria-valuetext={`${(p * 100).toFixed(1)} percent`}
      >
        <div className="meter-fill" style={{ width: `${p * 100}%` }} />
        <div className="meter-tick" />
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span>0%</span>
        <span>50% cut-off</span>
        <span>100%</span>
      </div>

      <p className="raw small">P(positive) = {p.toFixed(3)}</p>
    </article>
  );
}
