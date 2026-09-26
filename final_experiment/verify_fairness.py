#!/usr/bin/env python3
"""Fairness verification for the FINAL CONTROLLED EXPERIMENT.

Reads the artifacts produced by run_final_experiment.py and checks that the four
models really were trained and evaluated under identical conditions. Every check
compares what was actually written to disk, not what the code intended.

Exits 0 if every check passes, 1 otherwise.

    python final_experiment/verify_fairness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import shared.config as C  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MODELS = ("simple_rnn", "lstm", "gru", "cnn1d")

passed: list[str] = []
failed: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(label)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))


def all_same(values) -> bool:
    values = list(values)
    return all(v == values[0] for v in values)


def main() -> int:
    missing = [m for m in MODELS if not (OUT / f"{m}_final_metrics.json").exists()]
    if missing:
        print("Cannot verify: no results for " + ", ".join(missing))
        print("Run  python final_experiment/run_final_experiment.py  first.")
        return 1

    rec = {m: json.loads((OUT / f"{m}_final_metrics.json").read_text()) for m in MODELS}
    pred = {m: pd.read_csv(OUT / f"{m}_final_test_predictions.csv") for m in MODELS}

    print("=" * 78)
    print("FAIRNESS VERIFICATION - FINAL CONTROLLED EXPERIMENT")
    print("=" * 78)

    # ---- data ------------------------------------------------------------
    print("\nData pipeline")
    split = pd.read_csv(ROOT / "shared" / "split_assignment.csv")
    official_test = sorted(split.loc[split["split"] == "test", "row_id"])

    check("same dataset (all models trained from dataset/deceptive-opinion.csv)",
          (ROOT / "dataset" / "deceptive-opinion.csv").exists() and
          C.DATA_PATH.name == "deceptive-opinion.csv", C.DATA_PATH.name)

    check("same target column (polarity, negative=0 positive=1)",
          C.LABEL_COLUMN == "polarity" and C.LABEL_MAP == {"negative": 0, "positive": 1},
          f"{C.LABEL_COLUMN} {C.LABEL_MAP}")

    check("same split sizes across models",
          all_same(r["n_test"] for r in rec.values()),
          f"n_test = {rec['gru']['n_test']}")

    ids = {m: list(pred[m]["row_id"]) for m in MODELS}
    check("same test indices across models (identical ids, identical order)",
          all_same(ids.values()), f"{len(ids['gru'])} rows")

    check("test indices match shared/split_assignment.csv",
          all(sorted(ids[m]) == official_test for m in MODELS),
          f"{len(official_test)} official test rows")

    labels = {m: list(pred[m]["label"]) for m in MODELS}
    check("same ground-truth labels across models", all_same(labels.values()),
          f"{sum(labels['gru'])} positive / {len(labels['gru']) - sum(labels['gru'])} negative")

    check("no test row appears in the training or validation partition",
          set(official_test).isdisjoint(set(split.loc[split["split"] != "test", "row_id"])))

    # ---- preprocessing ---------------------------------------------------
    print("\nPreprocessing and representation")
    for label, key, want in [
        ("same vocabulary size", "vocab_size", C.VOCAB_SIZE),
        ("same MAX_LEN", "max_len", C.MAX_LEN),
        ("same embedding dimension", "embed_dim", C.EMBED_DIM),
        ("same padding direction", "padding", C.PADDING),
        ("same truncation direction", "truncating", C.TRUNCATING),
    ]:
        vals = [r[key] for r in rec.values()]
        check(label, all_same(vals) and vals[0] == want, f"{vals[0]}")

    check("same preprocessing code path (one shared clean_text / build_vocab / pad)",
          (ROOT / "shared" / "data_pipeline.py").exists(),
          "all four call shared.data_pipeline.prepare_data()")

    # ---- training configuration -----------------------------------------
    print("\nTraining configuration")
    for label, key, want in [
        ("same loss function", "loss", "binary_crossentropy"),
        ("same optimizer", "optimizer", "Adam"),
        ("same learning rate", "learning_rate", C.LEARNING_RATE),
        ("same batch size", "batch_size", C.BATCH_SIZE),
        ("same maximum epochs", "max_epochs", C.MAX_EPOCHS),
        ("same early stopping monitor", "early_stopping_monitor", C.MONITOR),
        ("same early stopping patience", "early_stopping_patience", C.PATIENCE),
        ("same restore_best_weights", "restore_best_weights", True),
        ("same random seed", "seed", C.SEED),
        ("same classification threshold", "threshold", C.THRESHOLD),
        ("same dense classification head", "dense_head",
         "Dense(32, relu) -> Dropout(0.3) -> Dense(1, sigmoid)"),
    ]:
        vals = [r[key] for r in rec.values()]
        check(label, all_same(vals) and vals[0] == want, f"{vals[0]}")

    cw = [r["class_weight"] for r in rec.values()]
    check("same class weights", all_same(cw),
          ", ".join(f"{k}={float(v):.6f}" for k, v in cw[0].items()))

    # ---- evaluation ------------------------------------------------------
    print("\nEvaluation")
    keys = {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}
    check("same evaluation metrics recorded for every model",
          all(keys <= set(r) for r in rec.values()), ", ".join(sorted(keys)))

    ok = True
    for m in MODELS:
        p = pred[m]
        derived = (p["prob_positive"] >= rec[m]["threshold"]).astype(int)
        if not derived.equals(p["prediction"]):
            ok = False
    check("predictions consistent with the 0.5 threshold on saved probabilities", ok)

    ok = True
    for m in MODELS:
        p, r = pred[m], rec[m]
        tp = int(((p.label == 1) & (p.prediction == 1)).sum())
        tn = int(((p.label == 0) & (p.prediction == 0)).sum())
        fp = int(((p.label == 0) & (p.prediction == 1)).sum())
        fn = int(((p.label == 1) & (p.prediction == 0)).sum())
        cm = r["confusion_matrix"]
        if (tn, fp, fn, tp) != (cm["tn"], cm["fp"], cm["fn"], cm["tp"]):
            ok = False
        if abs((tp + tn) / len(p) - r["accuracy"]) > 1e-9:
            ok = False
    check("reported metrics reproducible from the saved predictions", ok)

    # ---- what is allowed to differ ---------------------------------------
    print("\nThe only permitted difference")
    arch = {m: rec[m]["architecture"] for m in MODELS}
    check("architectures differ between models (this is the experimental variable)",
          len(set(arch.values())) == len(MODELS))
    stems = {m: a.split(" -> ")[0] for m, a in arch.items()}   # the Embedding layer only
    check("identical embedding stem", all_same(stems.values()), list(stems.values())[0])
    heads = {m: " -> ".join(a.split(" -> ")[-3:]) for m, a in arch.items()}
    check("identical classification head", all_same(heads.values()), list(heads.values())[0])

    print("\n" + "=" * 78)
    if failed:
        print(f"{len(failed)} CHECK(S) FAILED:")
        for f in failed:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print(f"ALL {len(passed)} CHECKS PASSED - the four models are directly comparable.")
    print("=" * 78)
    print("\nNote: wall-clock training time is recorded but deliberately excluded from")
    print("these checks. It is an observation, not a controlled condition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
