"""Shared evaluation: the same metrics, threshold and result schema for all four models.

Positive class (label 1 = positive review) is the class for precision / recall / F1,
because this is a balanced dataset and the group's table reports one number per metric.
"""
import json

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from shared import config as C


def compute_metrics(y_true, y_prob, threshold: float = C.THRESHOLD) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        # per-class view, needed for the class-imbalance / error analysis
        "negative_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "negative_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "negative_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, y_prob, n_boot: int = 2000, seed: int = C.SEED) -> dict:
    """95% bootstrap interval for accuracy / F1 / ROC-AUC on the test set.

    Only describes how uncertain the (small) test set makes the numbers.
    It is NOT used to tune or select anything.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).ravel()
    rng = np.random.default_rng(seed)
    out = {"accuracy": [], "f1": [], "roc_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_prob[idx])
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def save_result(path, model_name: str, test_metrics: dict, *, training_time_sec: float,
                trainable_params: int, total_params: int, epochs_run: int, best_epoch: int,
                extra: dict | None = None) -> dict:
    """Write one JSON file per model, with the same keys for every model,
    so the group can build the final comparison table from `results/*.json`."""
    row = {
        "model": model_name,
        "accuracy": test_metrics["accuracy"], "precision": test_metrics["precision"],
        "recall": test_metrics["recall"], "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "training_time_sec": training_time_sec,
        "trainable_params": trainable_params, "total_params": total_params,
        "epochs_run": epochs_run, "best_epoch": best_epoch,
        "seed": C.SEED, "max_len": C.MAX_LEN, "vocab_size": C.VOCAB_SIZE,
        "embed_dim": C.EMBED_DIM, "batch_size": C.BATCH_SIZE,
        "learning_rate": C.LEARNING_RATE, "threshold": C.THRESHOLD,
        "confusion_matrix": {k: test_metrics[k] for k in ("tn", "fp", "fn", "tp")},
    }
    if extra:
        row.update(extra)
    with open(path, "w") as f:
        json.dump(row, f, indent=2)
    return row
