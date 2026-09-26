#!/usr/bin/env python3
"""FINAL CONTROLLED EXPERIMENT
============================

Trains and evaluates all four architectures under ONE common configuration, so
that the only difference between them is the architecture itself.

This script supersedes the per-model hyperparameter searches for the purpose of
the final comparison. Those searches remain in the individual model folders as a
record of the development process; none of their model-specific choices are
active here.

What is held constant
---------------------
    data            one call to shared.data_pipeline.prepare_data(), reused by all
                    four models, so the split, vocabulary, padding and class
                    weights are not merely equivalent but literally the same objects
    loss            binary crossentropy
    optimiser       Adam, learning rate 0.001
    batch size      32
    max epochs      30
    early stopping  monitor val_loss, patience 5, restore_best_weights=True
    threshold       0.5
    seed            42, re-applied immediately before each model is built
    class weights   computed once from the training labels

What differs
------------
    only the layers between the Embedding and the Dense(32) head.

Leakage
-------
The test set is used exactly once per model, after training is complete. It is
never passed to fit(), never used for early stopping and never used to select
anything.

Usage
-----
    python final_experiment/run_final_experiment.py
    python final_experiment/run_final_experiment.py --models gru cnn1d   # subset
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import shared.config as C                                    # noqa: E402
from shared.data_pipeline import prepare_data                # noqa: E402
from shared.evaluation import bootstrap_ci, compute_metrics  # noqa: E402
from shared.plots import plot_confusion_matrix, plot_curve, plot_roc  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# The one common training configuration. Nothing below is overridden per model.
# ---------------------------------------------------------------------------
COMMON = {
    "loss": "binary_crossentropy",
    "optimizer": "Adam",
    "learning_rate": C.LEARNING_RATE,   # 0.001
    "batch_size": C.BATCH_SIZE,         # 32
    "max_epochs": C.MAX_EPOCHS,         # 30
    "early_stopping_monitor": C.MONITOR,  # val_loss
    "early_stopping_patience": C.PATIENCE,  # 5
    "restore_best_weights": True,
    "threshold": C.THRESHOLD,           # 0.5
    "seed": C.SEED,                     # 42
    "vocab_size": C.VOCAB_SIZE,         # 5000
    "max_len": C.MAX_LEN,               # 300
    "embed_dim": C.EMBED_DIM,           # 64
    "padding": C.PADDING,               # pre
    "truncating": C.TRUNCATING,         # post
    "dense_head": "Dense(32, relu) -> Dropout(0.3) -> Dense(1, sigmoid)",
}

MODELS = ("simple_rnn", "lstm", "gru", "cnn1d")
DISPLAY = {"simple_rnn": "Simple RNN", "lstm": "LSTM", "gru": "GRU", "cnn1d": "1D CNN"}


# ---------------------------------------------------------------------------
# Architectures. Identical embedding, identical classification head; only the
# sequence-processing layers differ.
# ---------------------------------------------------------------------------
def build(name: str):
    import keras
    layers = keras.layers

    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}")

    # Layers are constructed in forward order. This matters: each layer draws its
    # initial weights from the seeded RNG as it is created, so building the head
    # before the stem would give different initial weights for the same seed.
    stack = [
        layers.Input(shape=(C.MAX_LEN,)),
        layers.Embedding(C.VOCAB_SIZE, C.EMBED_DIM),
    ]

    if name == "simple_rnn":
        stack += [layers.SimpleRNN(64, return_sequences=True), layers.GlobalAveragePooling1D()]
    elif name == "lstm":
        stack += [layers.LSTM(64)]
    elif name == "gru":
        stack += [layers.GRU(64)]
    elif name == "cnn1d":
        # activation="relu" is kept from the existing implementation; a linear
        # convolution feeding a max-pool would be an unusual and weaker design.
        stack += [layers.Conv1D(128, 5, activation="relu"), layers.GlobalMaxPooling1D()]

    stack += [
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ]
    return keras.Sequential(stack, name=f"{name}_final")


def architecture_string(model) -> str:
    parts = []
    for layer in model.layers:
        cls = layer.__class__.__name__
        cfg = layer.get_config()
        if cls == "Embedding":
            parts.append(f"Embedding({cfg['input_dim']}, {cfg['output_dim']})")
        elif cls in ("SimpleRNN", "LSTM", "GRU"):
            rs = ", return_sequences=True" if cfg.get("return_sequences") else ""
            parts.append(f"{cls}({cfg['units']}{rs})")
        elif cls == "Conv1D":
            parts.append(f"Conv1D({cfg['filters']}, kernel_size={cfg['kernel_size'][0]}, "
                         f"{cfg['activation']})")
        elif cls == "Dense":
            parts.append(f"Dense({cfg['units']}, {cfg['activation']})")
        elif cls == "Dropout":
            parts.append(f"Dropout({cfg['rate']})")
        else:
            parts.append(cls)
    return " -> ".join(parts)


# ---------------------------------------------------------------------------
def train_and_evaluate(name: str, data) -> dict:
    """Train one model under the common configuration and score it on the test set once."""
    import keras

    keras.utils.set_random_seed(COMMON["seed"])       # seeds Python, NumPy and TensorFlow
    model = build(name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=COMMON["learning_rate"]),
        loss=COMMON["loss"],
        metrics=["accuracy"],
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor=COMMON["early_stopping_monitor"],
        patience=COMMON["early_stopping_patience"],
        restore_best_weights=COMMON["restore_best_weights"],
    )

    print(f"\n{'=' * 78}\n{DISPLAY[name]}  |  {architecture_string(model)}\n{'=' * 78}")
    print(f"trainable parameters: {model.count_params():,}")

    start = time.perf_counter()
    history = model.fit(
        data.X_train, data.y_train,
        validation_data=(data.X_val, data.y_val),     # validation only; test is untouched
        epochs=COMMON["max_epochs"],
        batch_size=COMMON["batch_size"],
        class_weight=data.class_weight,
        callbacks=[early_stop],
        verbose=2,
    ).history
    train_time = time.perf_counter() - start

    epochs_run = len(history["loss"])
    best_idx = int(np.argmin(history["val_loss"]))
    best_epoch = best_idx + 1
    best_val_loss = float(history["val_loss"][best_idx])

    # ---- the single look at the test set -----------------------------------
    y_prob = model.predict(data.X_test, verbose=0).ravel()
    test = compute_metrics(data.y_test, y_prob, threshold=COMMON["threshold"])
    ci = bootstrap_ci(data.y_test, y_prob)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT / f"{name}_final.keras")

    pd.DataFrame(history).rename_axis("epoch").reset_index().assign(
        epoch=lambda d: d.epoch + 1).to_csv(OUT / f"{name}_final_history.csv", index=False)

    pd.DataFrame({
        "row_id": data.test_df["row_id"].values,
        "label": data.y_test,
        "prob_positive": y_prob,
        "prediction": (y_prob >= COMMON["threshold"]).astype(int),
        "hotel": data.test_df["hotel"].values,
        "deceptive": data.test_df["deceptive"].values,
    }).to_csv(OUT / f"{name}_final_test_predictions.csv", index=False)

    plot_confusion_matrix(np.array([[test["tn"], test["fp"]], [test["fn"], test["tp"]]]),
                          DISPLAY[name], path=OUT / f"{name}_final_confusion_matrix.png",
                          show=False)
    plot_roc(data.y_test, y_prob, test["roc_auc"], DISPLAY[name],
             threshold=COMMON["threshold"], path=OUT / f"{name}_final_roc_curve.png", show=False)
    plot_curve(history, "loss", DISPLAY[name], best_epoch,
               path=OUT / f"{name}_final_loss_curve.png", show=False)
    plot_curve(history, "accuracy", DISPLAY[name], best_epoch,
               path=OUT / f"{name}_final_accuracy_curve.png", show=False)

    record = {
        "model": DISPLAY[name],
        "key": name,
        "architecture": architecture_string(model),
        "trainable_params": int(model.count_params()),
        "total_params": int(model.count_params()),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_accuracy": round(float(history["val_accuracy"][best_idx]), 4),
        "training_time_sec": round(train_time, 2),
        "accuracy": test["accuracy"],
        "precision": test["precision"],
        "recall": test["recall"],
        "f1": test["f1"],
        "roc_auc": test["roc_auc"],
        "confusion_matrix": {k: test[k] for k in ("tn", "fp", "fn", "tp")},
        "bootstrap_95ci": ci,
        **{k: v for k, v in COMMON.items()},
        "class_weight": {str(k): v for k, v in data.class_weight.items()},
        "n_test": int(len(data.y_test)),
    }
    (OUT / f"{name}_final_metrics.json").write_text(json.dumps(record, indent=2))

    print(f"epochs {epochs_run} (best {best_epoch}, val_loss {best_val_loss:.4f})  "
          f"time {train_time:.2f}s")
    print(f"TEST  acc {test['accuracy']:.4f}  prec {test['precision']:.4f}  "
          f"rec {test['recall']:.4f}  f1 {test['f1']:.4f}  auc {test['roc_auc']:.4f}")

    keras.backend.clear_session()
    return record


def significance(records: list[dict]) -> pd.DataFrame:
    """McNemar's exact test on every pair of models.

    Valid here because all four models predicted on the identical test reviews, so
    the comparison is paired. With 160 reviews a gap of a few points is easily
    chance; this says which differences are larger than that.
    """
    import math
    from itertools import combinations

    preds, y = {}, None
    for r in records:
        p = pd.read_csv(OUT / f"{r['key']}_final_test_predictions.csv").sort_values("row_id")
        preds[r["model"]] = p["prediction"].tolist()
        y = p["label"].tolist()

    rows = []
    for a, b in combinations(preds, 2):
        pa, pb = preds[a], preds[b]
        bb = sum(1 for i in range(len(y)) if pa[i] == y[i] and pb[i] != y[i])
        cc = sum(1 for i in range(len(y)) if pa[i] != y[i] and pb[i] == y[i])
        n = bb + cc
        if n == 0:
            pval = 1.0
        else:
            k = min(bb, cc)
            pval = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        rows.append({"Model A": a, "Model B": b, "A right, B wrong": bb,
                     "A wrong, B right": cc, "p-value": round(pval, 4),
                     "Significant (0.05)": "yes" if pval < 0.05 else "no"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "final_significance_mcnemar.csv", index=False)
    return table


def environment_record() -> dict:
    import keras
    import sklearn
    import tensorflow as tf

    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    cpu = sysctl("machdep.cpu.brand_string")
    cores = sysctl("hw.physicalcpu")
    mem = sysctl("hw.memsize")
    return {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "physical_cores": int(cores) if cores else None,
        "ram_gb": round(int(mem) / 1024**3) if mem else None,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "note": ("All four models in this run were trained in one session on this single "
                 "machine, so their training times are directly comparable with each other. "
                 "They are still not comparable with times recorded in the earlier per-model "
                 "notebooks, which ran on different machines."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Final controlled experiment")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()

    print("=" * 78)
    print("FINAL CONTROLLED EXPERIMENT")
    print("=" * 78)
    print("One configuration, one data pipeline, one test set. Architecture is the")
    print("only variable. No model-specific hyperparameter tuning is applied.\n")
    for k, v in COMMON.items():
        print(f"  {k:26} {v}")

    # ---- the data is prepared ONCE and shared by every model ---------------
    data = prepare_data()
    print(f"\nData prepared once and reused for all models:")
    print(f"  train {data.X_train.shape}  val {data.X_val.shape}  test {data.X_test.shape}")
    print(f"  vocabulary {len(data.vocab):,} (built from training rows only)")
    print(f"  class weights {dict((k, round(v, 6)) for k, v in data.class_weight.items())}")

    OUT.mkdir(parents=True, exist_ok=True)
    records = [train_and_evaluate(name, data) for name in args.models]

    table = pd.DataFrame([{
        "Model": r["model"],
        "Best Epoch": r["best_epoch"],
        "Total Epochs": r["epochs_run"],
        "Best Val Loss": r["best_val_loss"],
        "Test Accuracy": round(r["accuracy"], 4),
        "Precision": round(r["precision"], 4),
        "Recall": round(r["recall"], 4),
        "F1": round(r["f1"], 4),
        "ROC-AUC": round(r["roc_auc"], 4),
        "Params": r["trainable_params"],
        "Training Time (s)": r["training_time_sec"],
    } for r in records])

    table.to_csv(OUT / "final_results_table.csv", index=False)
    env = environment_record()
    (OUT / "final_environment.json").write_text(json.dumps(env, indent=2))

    md = ["# Final controlled experiment — results", "",
          "All four models were trained with the identical configuration, on the identical",
          "data pipeline, and evaluated on the identical 160-review test set.", "",
          table.to_markdown(index=False), "",
          "## Confusion matrices", "",
          "| Model | TN | FP | FN | TP |", "|---|---|---|---|---|"]
    for r in records:
        c = r["confusion_matrix"]
        md.append(f"| {r['model']} | {c['tn']} | {c['fp']} | {c['fn']} | {c['tp']} |")
    md += ["", "## Architectures", "",
           "| Model | Layer stack |", "|---|---|"]
    for r in records:
        md.append(f"| {r['model']} | `{r['architecture']}` |")

    if len(records) > 1:
        sig = significance(records)
        md += ["", "## Are the differences significant?", "",
               "All four models predicted on the identical 160 test reviews, so the comparison",
               "is paired and McNemar's exact test applies.", "",
               sig.to_markdown(index=False), ""]
        n_sig = int((sig["Significant (0.05)"] == "yes").sum())
        md.append(f"{n_sig} of {len(sig)} pairs differ significantly at the 0.05 level."
                  + ("" if n_sig else " On this test set the four architectures are"
                                      " statistically indistinguishable."))
    md += ["", "## Environment", "",
           f"- {env['cpu']}, {env['physical_cores']} cores, {env['ram_gb']} GB RAM",
           f"- {env['platform']}",
           f"- Python {env['python']}, TensorFlow {env['tensorflow']}, Keras {env['keras']}",
           f"- Accelerator: {', '.join(env['gpus']) if env['gpus'] else 'none (CPU only)'}",
           "",
           "Training time is reported as an observation only. It is **not** used to rank the",
           "models: wall-clock time depends on hardware, and it also depends on how many",
           "epochs early stopping allowed, which differs legitimately between architectures.",
           ""]
    (OUT / "final_results.md").write_text("\n".join(md))

    print(f"\n{'=' * 78}\nFINAL RESULTS\n{'=' * 78}")
    print(table.to_string(index=False))
    print(f"\nWritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
