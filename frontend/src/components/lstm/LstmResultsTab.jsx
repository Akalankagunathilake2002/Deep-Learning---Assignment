import { pct, plural } from "../../format.js";

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

export default function LstmResultsTab({ state }) {
  if (state.status === "loading") {
    return (
      <div className="card placeholder" role="status">
        <span className="spinner" aria-hidden="true" />
        Loading…
      </div>
    );
  }

  const r = state.status === "ready" ? state.results : null;
  const lstm = r?.lstm;
  const split = r?.split ?? { train: 1276, validation: 160, test: 160 };

  if (!lstm) {
    return (
      <div className="card placeholder error" role="alert">
        <strong>The saved LSTM results could not be loaded.</strong>
        <span>Check that results.json contains the lstm section.</span>
      </div>
    );
  }

  const cm = lstm.confusion_matrix;
  const [accLow, accHigh] = lstm.bootstrap_95ci.accuracy;

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
            <b>Read with an LSTM.</b> It reads words in order and maintains an additive cell state (C<sub>t</sub>) regulated by 4 gates (forget, input, candidate, output) to eliminate vanishing gradients. It ends with the probability that the review is positive.
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
          Final test on {split.test} reviews the model never saw during training or tuning (unidirectional LSTM, cut-off 0.5, positive class).
        </p>
        <dl className="stats">
          <Stat label="Accuracy" value={pct(lstm.accuracy)} note={`95% range ${pct(accLow)} to ${pct(accHigh)}`} />
          <Stat label="Precision" value={pct(lstm.precision)} note="of reviews called positive, how many were" />
          <Stat label="Recall" value={pct(lstm.recall)} note="of positive reviews, how many it found" />
          <Stat label="F1 score" value={pct(lstm.f1)} />
          <Stat label="ROC-AUC" value={lstm.roc_auc.toFixed(3)} note="1.0 is perfect, 0.5 is guessing" />
        </dl>
        <p className="small muted">
          {lstm.trainable_params.toLocaleString()} parameters · trained in about {Math.round(lstm.training_time_sec)} s · stopped after{" "}
          {lstm.epochs_run} epochs (best epoch {lstm.best_epoch}) · across {lstm.robustness.runs} random seeds the test accuracy was{" "}
          {pct(lstm.robustness.test_acc_mean)} ± {pct(lstm.robustness.test_acc_std)}. With only {split.test} test reviews, every score is uncertain by
          a few points.
        </p>

        <div className="figures">
          <Figure
            file="lstm_accuracy_curve.png"
            alt="Line chart of training and validation accuracy for each epoch. Training accuracy climbs to 99 percent while validation accuracy peaks at epoch 7."
          >
            <b>Accuracy per epoch.</b> Training reaches 99% while validation peaks at 95.6% at epoch {lstm.best_epoch}. Early stopping restored epoch {lstm.best_epoch}.
          </Figure>
          <Figure
            file="lstm_loss_curve.png"
            alt="Line chart of training and validation loss for each epoch. Validation loss reaches its minimum at epoch 7."
          >
            <b>Loss per epoch.</b> Validation loss hits its minimum (0.125) at epoch {lstm.best_epoch} before slight overfitting occurs.
          </Figure>
          <Figure
            file="lstm_confusion_matrix.png"
            alt={`Confusion matrix on the test set: ${cm.tn} true negatives, ${cm.fp} false positives, ${cm.fn} false negatives, ${cm.tp} true positives.`}
          >
            <b>Confusion matrix.</b> {cm.tn} negative and {cm.tp} positive reviews were right. {plural(cm.fn, "positive review")}{" "}
            {cm.fn === 1 ? "was" : "were"} missed (called negative) and {plural(cm.fp, "negative review")} {cm.fp === 1 ? "was" : "were"}{" "}
            wrongly called positive.
          </Figure>
          <Figure
            file="lstm_roc_curve.png"
            alt={`ROC curve on the test set with an area under the curve of ${lstm.roc_auc.toFixed(3)}, well above the diagonal for guessing.`}
          >
            <b>ROC curve</b> (AUC {lstm.roc_auc.toFixed(3)}). Positive reviews are cleanly ranked above negative reviews.
          </Figure>
        </div>
      </section>

      <section className="card" aria-labelledby="compare-title">
        <h2 id="compare-title">Unidirectional and Bidirectional LSTM comparison</h2>
        <p className="muted">
          Following Lab Sheet 05 (Task 3), both Unidirectional and Bidirectional LSTM architectures were evaluated on the validation set.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Measure</th>
                <th scope="col">Unidirectional LSTM (Selected)</th>
                <th scope="col">Bidirectional LSTM</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Recurrent parameters</th>
                <td>33,024</td>
                <td>66,048</td>
              </tr>
              <tr>
                <th scope="row">Total trainable parameters</th>
                <td>355,137</td>
                <td>390,209</td>
              </tr>
              <tr>
                <th scope="row">Validation loss (best epoch)</th>
                <td>0.198</td>
                <td>0.224</td>
              </tr>
              <tr>
                <th scope="row">Validation accuracy</th>
                <td>93.1%</td>
                <td>92.5%</td>
              </tr>
              <tr>
                <th scope="row">Validation F1 score</th>
                <td>93.3%</td>
                <td>92.7%</td>
              </tr>
              <tr>
                <th scope="row">Verdict</th>
                <td><b>Selected winner</b></td>
                <td>Higher loss, extra 35k parameters</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="small muted">
          Unidirectional LSTM was selected because sentiment polarity cues in review text flow naturally forward (e.g. negations preceding adjectives). With only 1,276 training reviews, doubling the recurrent parameters increases overfitting risk without improving validation accuracy.
        </p>
      </section>

      <section className="card" aria-labelledby="limits-title">
        <h2 id="limits-title">Limitations</h2>
        <ul className="limits">
          <li>
            <b>Over-confidence.</b> Extreme probabilities (e.g. &gt;99% or &lt;1%) are common because sigmoid saturates; high confidence does not guarantee certainty.
          </li>
          <li>
            <b>Mixed reviews.</b> Reviews containing conflicting praise and complaints within different sentences remain the most challenging to classify.
          </li>
          <li>
            <b>Small test set.</b> Test accuracy is {pct(lstm.accuracy)}, with a 95% bootstrap confidence interval of {pct(accLow)} to {pct(accHigh)}.
          </li>
          <li>
            <b>Single domain.</b> Trained and tested exclusively on Chicago hotel reviews from the Deceptive Opinion corpus, so generalization to product reviews or other genres is unverified.
          </li>
        </ul>
      </section>
    </div>
  );
}
