import { useMemo } from "react";

/** Shows what the models actually receive: the cleaned text and the words they look up in the vocabulary. */
export default function TokenView({ analysis, maxLen }) {
  const unknown = useMemo(() => new Set(analysis.unknownWords), [analysis]);
  const shown = analysis.tokens.slice(0, maxLen);

  return (
    <details className="details">
      <summary>What the models saw</summary>
      <div className="details-body">
        <p className="label">Cleaned text</p>
        <p className="mono">{analysis.clean || "(nothing left after cleaning)"}</p>

        <p className="label">
          Words the models receive ({shown.length}
          {analysis.unknownWords.length > 0 && `, ${analysis.unknownWords.length} not in the vocabulary`})
        </p>
        <ul className="tokens" aria-label="Words after cleaning">
          {shown.map((word, i) => (
            <li
              key={i}
              className={unknown.has(word) ? "token token-unknown" : "token"}
              title={unknown.has(word) ? "Not in the training vocabulary, so it becomes <OOV>" : undefined}
            >
              {word}
              {unknown.has(word) && <span aria-label="not in the vocabulary">?</span>}
            </li>
          ))}
        </ul>
        <p className="muted small">
          A dashed word with a "?" was never seen in training and is treated as unknown (&lt;OOV&gt;). The words are padded with{" "}
          {analysis.paddingCount} zeros in front so that every review is exactly {maxLen} long.
        </p>
      </div>
    </details>
  );
}
