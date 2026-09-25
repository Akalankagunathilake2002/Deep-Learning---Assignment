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

export default function CnnResultsTab({ state }) {
  const cnn = {
    accuracy: 0.94375,
    precision: 0.9176,
    recall: 0.9750,
    f1: 0.9455,
    roc_auc: 0.9702,
    trainable_params: 365249,
    training_time_sec: 5.92,
    epochs_run: 18,
    best_epoch: 13,
    confusion_matrix: { tn: 73, fp: 7, fn: 2, tp: 78 },
    bootstrap_95ci: { accuracy: [0.90625, 0.9750] },
    robustness: { runs: 6, test_acc_mean: 0.9208, test_acc_std: 0.0145 },
  };

  const cm = cnn.confusion_matrix;
  const [accLow, accHigh] = cnn.bootstrap_95ci.accuracy;

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
            <b>Read with a 1D CNN.</b> It applies 128 sliding 5-gram convolutional filters across word embeddings to detect localized keyphrases, followed by global max-pooling for position invariance. It ends with the probability that the review is positive.
          </li>
        </ol>
        <p className="muted small">
          Trained on 1,276 hotel reviews (Deceptive Opinion Spam corpus, 20 Chicago hotels), with 160 more for validation and 160 kept aside for the final test.
        </p>
      </section>

      <section className="card" aria-labelledby="score-title">
        <h2 id="score-title">How well it works</h2>
        <p className="muted">
          Final test on 160 reviews the model never saw during training or tuning (1D CNN, cut-off 0.5, positive class).
        </p>
        <dl className="stats">
          <Stat label="Accuracy" value={pct(cnn.accuracy)} note={`95% range ${pct(accLow)} to ${pct(accHigh)}`} />
          <Stat label="Precision" value={pct(cnn.precision)} note="of reviews called positive, how many were" />
          <Stat label="Recall" value={pct(cnn.recall)} note="of positive reviews, how many it found" />
          <Stat label="F1 score" value={pct(cnn.f1)} />
          <Stat label="ROC-AUC" value={cnn.roc_auc.toFixed(3)} note="1.0 is perfect, 0.5 is guessing" />
        </dl>
        <p className="small muted">
          {cnn.trainable_params.toLocaleString()} parameters · trained in about {Math.round(cnn.training_time_sec)} s · stopped after{" "}
          {cnn.epochs_run} epochs (best epoch {cnn.best_epoch}) · across {cnn.robustness.runs} random seeds the test accuracy was{" "}
          {pct(cnn.robustness.test_acc_mean)} ± {pct(cnn.robustness.test_acc_std)}. With only 160 test reviews, every score is uncertain by
          a few points.
        </p>

        <div className="figures">
          <Figure
            file="cnn1d_accuracy_curve.png"
            alt="Line chart of training and validation accuracy for each epoch. Training accuracy climbs while validation accuracy peaks at epoch 13."
          >
            <b>Accuracy per epoch.</b> Training reaches 100% while validation peaks at epoch {cnn.best_epoch}: early stopping restores epoch {cnn.best_epoch}.
          </Figure>
          <Figure
            file="cnn1d_loss_curve.png"
            alt="Line chart of training and validation loss for each epoch. Validation loss is lowest at epoch 13 and stays above that minimum afterwards."
          >
            <b>Loss per epoch.</b> Validation loss is lowest at epoch {cnn.best_epoch} and stays above that minimum afterwards.
          </Figure>
          <Figure
            file="cnn1d_confusion_matrix.png"
            alt={`Confusion matrix on the test set: ${cm.tn} true negatives, ${cm.fp} false positives, ${cm.fn} false negatives, ${cm.tp} true positives.`}
          >
            <b>Confusion matrix.</b> {cm.tn} negative and {cm.tp} positive reviews were right. {plural(cm.fn, "positive review")}{" "}
            {cm.fn === 1 ? "was" : "were"} missed (called negative) and {plural(cm.fp, "negative review")} {cm.fp === 1 ? "was" : "were"}{" "}
            wrongly called positive.
          </Figure>
          <Figure
            file="cnn1d_roc_curve.png"
            alt={`ROC curve on the test set with an area under the curve of ${cnn.roc_auc.toFixed(3)}, well above the diagonal for guessing.`}
          >
            <b>ROC curve</b> (AUC {cnn.roc_auc.toFixed(3)}). Positive reviews are usually ranked above negative ones.
          </Figure>
        </div>
      </section>

      <section className="card" aria-labelledby="compare-title">
        <h2 id="compare-title">Controlled architecture and capacity variants</h2>
        <p className="muted">
          Evaluating the trade-off between n-gram filter width, feature map capacity, and generalization on the common dataset.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Configuration</th>
                <th scope="col">Compact (Trigram)</th>
                <th scope="col">Champion 1D CNN</th>
                <th scope="col">High Capacity</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Filter kernel size</th>
                <td>3-gram (k=3)</td>
                <td>5-gram (k=5)</td>
                <td>5-gram (k=5)</td>
              </tr>
              <tr>
                <th scope="row">Convolutional filters</th>
                <td>64</td>
                <td>128</td>
                <td>256</td>
              </tr>
              <tr>
                <th scope="row">Embedding dimensions</th>
                <td>64</td>
                <td>64</td>
                <td>128</td>
              </tr>
              <tr>
                <th scope="row">Validation accuracy</th>
                <td>88.1%</td>
                <td>88.1%</td>
                <td>86.3%</td>
              </tr>
              <tr>
                <th scope="row">Test accuracy</th>
                <td>91.9%</td>
                <td>94.4%</td>
                <td>90.6%</td>
              </tr>
              <tr>
                <th scope="row">Test F1 score</th>
                <td>0.920</td>
                <td>0.945</td>
                <td>0.908</td>
              </tr>
              <tr>
                <th scope="row">Trainable parameters</th>
                <td>656,577</td>
                <td>365,249</td>
                <td>1,477,121</td>
              </tr>
              <tr>
                <th scope="row">Training time / epochs</th>
                <td>1.8 s / 9</td>
                <td>5.9 s / 18</td>
                <td>5.1 s / 7</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="small muted">
          The 128-filter 5-gram architecture achieves the optimal balance of localized keyphrase extraction without the parameter bloat and early overfitting seen in the 256-filter model.
        </p>
      </section>

      <section className="card" aria-labelledby="limits-title">
        <h2 id="limits-title">Limitations</h2>
        <ul className="limits">
          <li>
            <b>Localized context window.</b> The 5-gram filters excel at localized keyphrase extraction, but cannot capture dependencies that span across distant sentences as recurrent networks or transformers do.
          </li>
          <li>
            <b>Over-confidence.</b> Wrong answers are often given with high confidence (&gt;90%) when strong sentiment words dominate a misleading sentence.
          </li>
          <li>
            <b>Distant negation.</b> When negation cues ("not", "never") are separated from the sentiment descriptor by more than 5 words, the convolutional filter window misses the polarity inversion.
          </li>
          <li>
            <b>Small test set.</b> Accuracy is {pct(cnn.accuracy)}, but the 95% range is {pct(accLow)} to {pct(accHigh)}.
          </li>
          <li>
            <b>One dataset.</b> The model saw only Chicago hotel reviews from one corpus, so it may not transfer to other sources.
          </li>
        </ul>
      </section>
    </div>
  );
}
