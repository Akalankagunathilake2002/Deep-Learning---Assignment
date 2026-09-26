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

export default function RnnResultsTab({ state }) {
  if (state.status === "loading") {
    return (
      <div className="card placeholder" role="status">
        <span className="spinner" aria-hidden="true" />
        Loading…
      </div>
    );
  }

  const r = state.status === "ready" ? state.results : null;
  const rnn = r?.rnn;
  const split = r?.split ?? { train: 1276, validation: 160, test: 160 };

  if (!rnn) {
    return (
      <div className="card placeholder error" role="alert">
        <strong>The saved Simple RNN results could not be loaded.</strong>
        <span>Check that results.json contains the rnn section.</span>
      </div>
    );
  }

  const cm = rnn.confusion_matrix;
  const [accLow, accHigh] = rnn.bootstrap_95ci.accuracy;

  return (
    <div className="results-page">
      {/* Model overview */}
      <section className="card" aria-labelledby="model-title">
        <h2 id="model-title">Simple RNN Sentiment Model</h2>

        <p className="muted">
          Sequence-Averaged Simple RNN (Elman, 1990) with GlobalAveragePooling1D, trained to classify hotel reviews as positive or negative.
        </p>

        <p className="small muted">
          {split.train.toLocaleString()} training reviews · {split.validation} validation reviews · {split.test} test reviews
        </p>
      </section>

      {/* Performance */}
      <section className="card" aria-labelledby="score-title">
        <h2 id="score-title">Model Performance</h2>

        <p className="muted">
          Final evaluation on {split.test} test reviews that were not used during training or model selection (Seed 42).
        </p>

        <dl className="stats">
          <Stat
            label="Accuracy"
            value={pct(rnn.accuracy)}
            note={`95% range ${pct(accLow)} to ${pct(accHigh)}`}
          />

          <Stat
            label="Precision"
            value={pct(rnn.precision)}
            note="of reviews called positive, how many were correct"
          />

          <Stat
            label="Recall"
            value={pct(rnn.recall)}
            note="of positive reviews, how many it found"
          />

          <Stat
            label="F1 score"
            value={pct(rnn.f1)}
          />

          <Stat
            label="ROC-AUC"
            value={rnn.roc_auc.toFixed(3)}
            note="1.0 is perfect, 0.5 is guessing"
          />
        </dl>

        <p className="small muted">
          {rnn.trainable_params.toLocaleString()} parameters · trained in about {Math.round(rnn.training_time_sec)} s · {rnn.epochs_run} epochs · best epoch {rnn.best_epoch}
        </p>

        <div className="figures">
          <Figure
            file="simple_rnn_accuracy_curve.png"
            alt={`Line chart of Simple RNN training and validation accuracy. Validation accuracy peaks at epoch ${rnn.best_epoch}.`}
          >
            <b>Accuracy per epoch.</b> Validation accuracy reaches {pct(rnn.validation_accuracy)} at epoch {rnn.best_epoch}.
          </Figure>

          <Figure
            file="simple_rnn_loss_curve.png"
            alt={`Line chart of Simple RNN training and validation loss. Validation loss reaches its minimum at epoch ${rnn.best_epoch}.`}
          >
            <b>Loss per epoch.</b> Validation loss reaches its minimum of {rnn.validation_loss.toFixed(3)} at epoch {rnn.best_epoch}.
          </Figure>

          <Figure
            file="simple_rnn_confusion_matrix.png"
            alt={`Confusion matrix on the test set: ${cm.tn} true negatives, ${cm.fp} false positives, ${cm.fn} false negatives, and ${cm.tp} true positives.`}
          >
            <b>Confusion matrix.</b> {cm.tn} negative and {cm.tp} positive reviews were correctly classified. {plural(cm.fn, "positive review")} {cm.fn === 1 ? "was" : "were"} missed, while {plural(cm.fp, "negative review")} {cm.fp === 1 ? "was" : "were"} wrongly classified as positive.
          </Figure>

          <Figure
            file="simple_rnn_roc_curve.png"
            alt={`ROC curve on the test set with an area under the curve of ${rnn.roc_auc.toFixed(3)}.`}
          >
            <b>ROC curve.</b> AUC = {rnn.roc_auc.toFixed(3)}.
          </Figure>
        </div>
      </section>

      {/* Model complexity */}
      <section className="card" aria-labelledby="complexity-title">
        <h2 id="complexity-title">Model Complexity & Recurrence</h2>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Measure</th>
                <th scope="col">Value</th>
              </tr>
            </thead>

            <tbody>
              <tr>
                <th scope="row">Architecture</th>
                <td>Sequence-Averaged SimpleRNN(64) + GlobalAveragePooling1D</td>
              </tr>

              <tr>
                <th scope="row">Recurrent Parameters</th>
                <td>8,256 params (64×64 input + 64×64 recurrent + 64 bias)</td>
              </tr>

              <tr>
                <th scope="row">Total Trainable Parameters</th>
                <td>{rnn.trainable_params.toLocaleString()} (96.86% in Embedding)</td>
              </tr>

              <tr>
                <th scope="row">Training Time</th>
                <td>~{Math.round(rnn.training_time_sec)} seconds</td>
              </tr>

              <tr>
                <th scope="row">Best Epoch</th>
                <td>{rnn.best_epoch} (EarlyStopping restored)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Architectural Characteristics */}
      <section className="card" aria-labelledby="limits-title">
        <h2 id="limits-title">Architectural Characteristics & Limitations</h2>

        <ul className="limits">
          <li>
            <b>Vanishing Gradients.</b> Simple RNN lacks input, forget, and update gating mechanisms, meaning backpropagation through long token sequences suffers from exponential gradient decay. This makes long-range context harder to retain compared to GRU and LSTM.
          </li>

          <li>
            <b>Lightweight Sequential Baseline.</b> With only 8,256 recurrent parameters (vs. 24,960 in GRU and 33,024 in LSTM), Simple RNN serves as the foundational sequential benchmark of this comparative study.
          </li>

          <li>
            <b>Evaluation Protocol.</b> Evaluated on {split.test} test reviews with a 95% bootstrap confidence interval of {pct(accLow)} to {pct(accHigh)} for accuracy.
          </li>
        </ul>
      </section>
    </div>
  );
}
