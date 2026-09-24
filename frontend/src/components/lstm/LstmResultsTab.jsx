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

      {/* Model overview */}
      <section className="card" aria-labelledby="model-title">
        <h2 id="model-title">LSTM Sentiment Model</h2>

        <p className="muted">
          Unidirectional LSTM trained to classify hotel reviews as positive
          or negative.
        </p>

        <p className="small muted">
          {split.train.toLocaleString()} training reviews ·{" "}
          {split.validation} validation reviews ·{" "}
          {split.test} test reviews
        </p>
      </section>

      {/* Performance */}
      <section className="card" aria-labelledby="score-title">
        <h2 id="score-title">Model Performance</h2>

        <p className="muted">
          Final evaluation on {split.test} test reviews that were not used
          during training or model selection.
        </p>

        <dl className="stats">
          <Stat
            label="Accuracy"
            value={pct(lstm.accuracy)}
            note={`95% range ${pct(accLow)} to ${pct(accHigh)}`}
          />

          <Stat
            label="Precision"
            value={pct(lstm.precision)}
            note="of reviews called positive, how many were correct"
          />

          <Stat
            label="Recall"
            value={pct(lstm.recall)}
            note="of positive reviews, how many it found"
          />

          <Stat
            label="F1 score"
            value={pct(lstm.f1)}
          />

          <Stat
            label="ROC-AUC"
            value={lstm.roc_auc.toFixed(3)}
            note="1.0 is perfect, 0.5 is guessing"
          />
        </dl>

        <p className="small muted">
          {lstm.trainable_params.toLocaleString()} parameters ·{" "}
          trained in about {Math.round(lstm.training_time_sec)} s ·{" "}
          {lstm.epochs_run} epochs · best epoch {lstm.best_epoch}
        </p>

        <div className="figures">

          <Figure
            file="lstm_accuracy_curve.png"
            alt={`Line chart of training and validation accuracy. Validation accuracy peaks at epoch ${lstm.best_epoch}.`}
          >
            <b>Accuracy per epoch.</b>{" "}
            Validation accuracy peaks at {pct(0.9563)} at epoch{" "}
            {lstm.best_epoch}.
          </Figure>

          <Figure
            file="lstm_loss_curve.png"
            alt={`Line chart of training and validation loss. Validation loss reaches its minimum at epoch ${lstm.best_epoch}.`}
          >
            <b>Loss per epoch.</b>{" "}
            Validation loss reaches its minimum of 0.125 at epoch{" "}
            {lstm.best_epoch}.
          </Figure>

          <Figure
            file="lstm_confusion_matrix.png"
            alt={`Confusion matrix on the test set: ${cm.tn} true negatives, ${cm.fp} false positives, ${cm.fn} false negatives, and ${cm.tp} true positives.`}
          >
            <b>Confusion matrix.</b>{" "}
            {cm.tn} negative and {cm.tp} positive reviews were correctly
            classified. {plural(cm.fn, "positive review")}{" "}
            {cm.fn === 1 ? "was" : "were"} missed, while{" "}
            {plural(cm.fp, "negative review")}{" "}
            {cm.fp === 1 ? "was" : "were"} wrongly classified as positive.
          </Figure>

          <Figure
            file="lstm_roc_curve.png"
            alt={`ROC curve on the test set with an area under the curve of ${lstm.roc_auc.toFixed(3)}.`}
          >
            <b>ROC curve.</b>{" "}
            AUC = {lstm.roc_auc.toFixed(3)}.
          </Figure>

        </div>
      </section>

      {/* Model complexity */}
      <section className="card" aria-labelledby="complexity-title">
        <h2 id="complexity-title">Model Complexity</h2>

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
                <td>Unidirectional LSTM</td>
              </tr>

              <tr>
                <th scope="row">Trainable parameters</th>
                <td>{lstm.trainable_params.toLocaleString()}</td>
              </tr>

              <tr>
                <th scope="row">Training time</th>
                <td>~{Math.round(lstm.training_time_sec)} seconds</td>
              </tr>

              <tr>
                <th scope="row">Best epoch</th>
                <td>{lstm.best_epoch}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Limitations */}
      <section className="card" aria-labelledby="limits-title">
        <h2 id="limits-title">Limitations</h2>

        <ul className="limits">
          <li>
            <b>Mixed reviews.</b>{" "}
            Reviews containing conflicting praise and complaints can be
            difficult to classify.
          </li>

          <li>
            <b>Small test set.</b>{" "}
            Evaluation is based on {split.test} test reviews, with a 95%
            bootstrap confidence interval of {pct(accLow)} to{" "}
            {pct(accHigh)} for accuracy.
          </li>

          <li>
            <b>Single domain.</b>{" "}
            The model was trained and tested on Chicago hotel reviews, so
            performance on other review domains is unverified.
          </li>
        </ul>
      </section>

    </div>
  );
}