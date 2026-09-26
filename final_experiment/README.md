# Final controlled experiment

The single source of truth for the four-model comparison in the report.

Each model also has its own folder (`simple_rnn/`, `lstm/`, `gru/`, `cnn1d/`) containing
that member's development work, including their exploratory hyperparameter searches. Those
searches differed in size between members, so results taken from them are **not** directly
comparable. Nothing from them is active here.

This experiment trains all four architectures under one common configuration, in one
session, on one machine, and evaluates them on the same 160 test reviews.

## Run it

```bash
python final_experiment/run_final_experiment.py      # trains all four (~45 s on CPU)
python final_experiment/verify_fairness.py           # 31 checks, exits non-zero on failure
```

Optional subset: `--models gru cnn1d`.

Requires TensorFlow ≥ 2.16 (see `requirements.txt`). Run from the repository root.

## What is held constant

The data is prepared by **one** call to `shared.data_pipeline.prepare_data()` and reused by
every model, so the split, vocabulary, padding and class weights are not merely equivalent —
they are the same objects in memory.

| | |
|---|---|
| Loss | binary crossentropy |
| Optimiser | Adam, learning rate 0.001 |
| Batch size | 32 |
| Max epochs | 30 |
| Early stopping | monitor `val_loss`, patience 5, `restore_best_weights=True` |
| Threshold | 0.5 |
| Seed | 42, re-applied immediately before each model is built |
| Class weights | computed once from the training labels |
| Vocabulary | 5,000, built from training rows only |
| Sequence | `MAX_LEN` 300, pre-padding, post-truncation |
| Embedding | `Embedding(5000, 64)`, randomly initialised, learned |
| Head | `Dense(32, relu) → Dropout(0.3) → Dense(1, sigmoid)` |

## What differs

Only the layers between the embedding and the head.

| Model | Sequence layers |
|---|---|
| Simple RNN | `SimpleRNN(64, return_sequences=True)` → `GlobalAveragePooling1D` |
| LSTM | `LSTM(64)` |
| GRU | `GRU(64)` |
| 1D CNN | `Conv1D(128, kernel_size=5, relu)` → `GlobalMaxPooling1D` |

## Leakage

The test set is touched once per model, after training finishes. It is never passed to
`fit()`, never used for early stopping, and never used to select anything.

## Outputs

Written to `final_experiment/results/`:

| File | Contents |
|---|---|
| `final_results_table.csv` | The comparison table |
| `final_results.md` | Table, confusion matrices, architectures, significance, environment |
| `final_significance_mcnemar.csv` | Pairwise McNemar tests |
| `final_environment.json` | Versions and hardware, recorded automatically |
| `<model>_final_metrics.json` | Full metrics, bootstrap CIs, config actually used |
| `<model>_final_test_predictions.csv` | Per-review probability and prediction |
| `<model>_final_history.csv` | Per-epoch training history |
| `<model>_final_*.png` | Confusion matrix, ROC, loss and accuracy curves |
| `<model>_final.keras` | The trained model |

## Two notes on reproducibility

**Layer construction order matters.** Layers are built in forward order, because each draws
its initial weights from the seeded RNG as it is created. Building the classification head
before the embedding gives different initial weights for the same seed — this was a real bug
during development and it changed the LSTM's result substantially.

**Metrics reproduce exactly; times do not.** Running the script twice gives byte-identical
metrics. Wall-clock training time varies by a few tenths of a second between runs, which is
why it is reported as an observation and never used to rank the models.
