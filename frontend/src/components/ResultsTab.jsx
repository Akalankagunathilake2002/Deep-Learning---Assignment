import { pct, plural } from "../format.js";

const asset = (path) => `${import.meta.env?.BASE_URL ?? "./"}${path}`;

function Stat({ label, value, note }) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd>{value}</dd>
      {note && <small>{note}</small>}
    </div>
  );
}

function Figure({ file, alt, children }) {
  return (
    <figure className="figure">
      <img src={asset(`figures/${file}`)} alt={alt} loading="lazy" />
      <figcaption>{children}</figcaption>
    </figure>
  );
}


export default function ResultsTab({ state }) {
  if (state.status === "loading") {
    return (
      <div className="card placeholder" role="status">
        <span className="spinner" aria-hidden="true" />
        Loading…
      </div>
    );
  }
  const r = state.status === "ready" ? state.results : null;
  if (!r) {
    return (
      <div className="card placeholder error" role="alert">
        <strong>The saved results could not be loaded.</strong>
        <span>{state.message ?? "results.json is missing. Run frontend/scripts/export_model.py to create it."}</span>
      </div>
    );
  }

  const { main, robustness, split } = r;
  const cm = main.confusion_matrix;
  const [accLow, accHigh] = main.bootstrap_95ci.accuracy;

  return (
    <div className="results-page">
      <section className="card" aria-labelledby="how-title">
        <h2 id="how-title">How the model works</h2>
        <ol className="steps">
          <li>
            <b>Clean</b> the text: lowercase, remove HTML, links, digits and punctuation, but keep negations such as “not”.
          </li>
          <li>
            <b>Look up</b> each word in a 5,000-word vocabulary built only from the training reviews. Unknown words become &lt;OOV&gt;.
          </li>
          <li>
            <b>Pad or cut</b> to 300 words so that every review has the same length.
          </li>
          <li>
            <b>Read with a GRU.</b> It reads the words in order and keeps a memory that two gates update, one deciding what to keep and
            one what to forget. It ends with the probability that the review is positive.
          </li>
        </ol>
        <p className="muted small">
          Trained on {split.train.toLocaleString()} hotel reviews (Deceptive Opinion Spam corpus, 20 Chicago hotels), with {split.validation}{" "}
          more for validation and {split.test} kept aside for the final test.
        </p>
      </section>

      <section className="card" aria-labelledby="score-title">
        <h2 id="score-title">How well it works</h2>
        <p className="muted">
          Final test on {split.test} reviews the model never saw during training or tuning (main GRU, cut-off 0.5, positive class).
        </p>
        <dl className="stats">
          <Stat label="Accuracy" value={pct(main.accuracy)} note={`95% range ${pct(accLow)} to ${pct(accHigh)}`} />
          <Stat label="Precision" value={pct(main.precision)} note="of reviews called positive, how many were" />
          <Stat label="Recall" value={pct(main.recall)} note="of positive reviews, how many it found" />
          <Stat label="F1 score" value={pct(main.f1)} />
          <Stat label="ROC-AUC" value={main.roc_auc.toFixed(3)} note="1.0 is perfect, 0.5 is guessing" />
        </dl>
        <p className="small muted">
          {main.trainable_params.toLocaleString()} parameters · trained in about {Math.round(main.training_time_sec)} s · stopped after{" "}
          {main.epochs_run} epochs (best epoch {main.best_epoch}) · across {robustness.runs} random seeds the test accuracy was{" "}
          {pct(robustness.test_acc_mean)} ± {pct(robustness.test_acc_std)}. With only {split.test} test reviews, every score is uncertain by
          a few points.
        </p>

        <div className="figures">
          <Figure
            file="gru_accuracy_curve.png"
            alt="Line chart of training and validation accuracy for each epoch. Training accuracy climbs to 100 percent while validation accuracy peaks at the best epoch."
          >
            <b>Accuracy per epoch.</b> Training reaches 100% but validation peaks at epoch {main.best_epoch}: the model overfits, so early
            stopping restores epoch {main.best_epoch}.
          </Figure>
          <Figure
            file="gru_loss_curve.png"
            alt="Line chart of training and validation loss for each epoch. Training loss keeps falling while validation loss is lowest at the best epoch and stays higher afterwards."
          >
            <b>Loss per epoch.</b> Validation loss is lowest at epoch {main.best_epoch} and stays above that minimum afterwards.
          </Figure>
          <Figure
            file="gru_confusion_matrix.png"
            alt={`Confusion matrix on the test set: ${cm.tn} true negatives, ${cm.fp} false positives, ${cm.fn} false negatives, ${cm.tp} true positives.`}
          >
            <b>Confusion matrix.</b> {cm.tn} negative and {cm.tp} positive reviews were right. {plural(cm.fn, "positive review")}{" "}
            {cm.fn === 1 ? "was" : "were"} missed (called negative) and {plural(cm.fp, "negative review")} {cm.fp === 1 ? "was" : "were"}{" "}
            wrongly called positive.
          </Figure>
          <Figure
            file="gru_roc_curve.png"
            alt={`ROC curve on the test set with an area under the curve of ${main.roc_auc.toFixed(3)}, well above the diagonal for guessing.`}
          >
            <b>ROC curve</b> (AUC {main.roc_auc.toFixed(3)}). Positive reviews are usually ranked above negative ones.
          </Figure>
        </div>
      </section>

      <section className="card" aria-labelledby="limits-title">
        <h2 id="limits-title">Limitations</h2>
        <ul className="limits">
          <li>
            <b>Very short reviews.</b> Every training review has at least 25 words and positive reviews are shorter on average, so the model
            learned that short means positive. One-line reviews are therefore unreliable, especially negative ones.
          </li>
          <li>
            <b>Over-confidence.</b> Wrong answers are often given with more than 90% confidence, so a high percentage is not a guarantee.
          </li>
          <li>
            <b>Mixed reviews.</b> Reviews that praise one thing and criticise another are the hardest for it.
          </li>
          <li>
            <b>Small test set.</b> Accuracy is {pct(main.accuracy)}, but the 95% range is {pct(accLow)} to {pct(accHigh)}.
          </li>
          <li>
            <b>One dataset.</b> The model saw only Chicago hotel reviews from one corpus, so it may not transfer to other sources.
          </li>
        </ul>
      </section>
    </div>
  );
}
