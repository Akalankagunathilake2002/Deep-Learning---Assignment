"""
Model Evaluation, Learning Curves, ROC, Confusion Matrix, and Error Analysis Module
Author: Dilmith
Course: SE4050 - Deep Learning
Component: 1D CNN
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


def plot_training_curves(
    history,
    save_dir: str = "results/figures",
    prefix: str = "cnn"
) -> tuple[str, str]:
    """
    Plots and saves two separate learning curves:
    1. Training Accuracy vs Validation Accuracy
    2. Training Loss vs Validation Loss

    Parameters:
        history: Keras History object.
        save_dir (str): Directory to save plots.
        prefix (str): File prefix.

    Returns:
        tuple: (accuracy_plot_path, loss_plot_path)
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history.history["loss"]) + 1)

    # 1. Accuracy Curve
    acc_path = os.path.join(save_dir, f"{prefix}_accuracy_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["accuracy"], "o-", color="#1f77b4", linewidth=2, label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], "s--", color="#ff7f0e", linewidth=2, label="Validation Accuracy")
    plt.title("1D CNN Model Accuracy Across Epochs (SE4050)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, labelpad=8)
    plt.ylabel("Accuracy", fontsize=11, labelpad=8)
    plt.xticks(epochs)
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # 2. Loss Curve
    loss_path = os.path.join(save_dir, f"{prefix}_loss_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["loss"], "o-", color="#1f77b4", linewidth=2, label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], "s--", color="#d62728", linewidth=2, label="Validation Loss")
    plt.title("1D CNN Model Loss Across Epochs (Binary Crossentropy)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, labelpad=8)
    plt.ylabel("Loss (Binary Crossentropy)", fontsize=11, labelpad=8)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    return acc_path, loss_path


def compute_test_metrics(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Computes comprehensive binary classification metrics on unseen test data.
    Uses continuous probabilities for ROC-AUC and thresholded predictions for others.

    Parameters:
        model: Trained Keras model.
        X_test: Test input sequences.
        y_test: True test binary labels.
        threshold (float): Decision threshold (default: 0.5).

    Returns:
        dict: Test metrics dictionary containing y_prob and y_pred.
    """
    # Continuous probabilities
    y_prob = model.predict(X_test, verbose=0).ravel()

    # Binary thresholded predictions
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def plot_confusion_matrix_annotated(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "results/figures/cnn_confusion_matrix.png"
) -> np.ndarray:
    """
    Generates and saves a clearly annotated Confusion Matrix identifying
    TN (Genuine -> Genuine), FP (Genuine -> Fake),
    FN (Fake -> Genuine), and TP (Fake -> Fake).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    labels = np.array([
        [f"TN\n{tn}\n(Genuine as Genuine)", f"FP\n{fp}\n(Genuine as Fake)"],
        [f"FN\n{fn}\n(Fake as Genuine)", f"TP\n{tp}\n(Fake as Fake)"]
    ])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Genuine (0)", "Fake (1)"],
        yticklabels=["Genuine (0)", "Fake (1)"],
        annot_kws={"fontsize": 11, "fontweight": "medium"}
    )
    plt.title("1D CNN Confusion Matrix (SE4050 Evaluation)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return cm


def plot_roc_curve_annotated(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str = "results/figures/cnn_roc_curve.png"
) -> float:
    """
    Generates and saves the Receiver Operating Characteristic (ROC) curve
    with the computed AUC score and random guessing baseline.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2b5c8f", linewidth=2.5, label=f"1D CNN (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.5, label="Random Classifier (AUC = 0.5000)")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, labelpad=8)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return float(auc_score)


def save_classification_report_txt(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "results/metrics/cnn_classification_report.txt"
) -> str:
    """
    Generates sklearn classification report and saves to text file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Genuine (0)", "Fake (1)"],
        digits=4
    )
    with open(save_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("1D CNN CLASSIFICATION REPORT (SE4050)\n")
        f.write("Student: Dilmith\n")
        f.write("Target Classes: 0 = Genuine, 1 = Fake\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n")
    return report


def perform_detailed_error_analysis(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_samples: int = 5
) -> dict:
    """
    Extracts representative False Positives and False Negatives for academic inspection.

    False Positives (FP): True Genuine (0) misclassified as Fake (1).
    False Negatives (FN): True Fake (1) misclassified as Genuine (0).

    Returns:
        dict: Dictionary containing FP and FN DataFrames with review text,
              true label, predicted label, and probability.
    """
    df_eval = pd.DataFrame({
        "review_text": list(texts),
        "true_label": list(y_true),
        "pred_label": list(y_pred),
        "pred_probability": [round(float(p), 4) for p in y_prob]
    })

    # False Positives: True=0, Pred=1
    fp_df = df_eval[(df_eval["true_label"] == 0) & (df_eval["pred_label"] == 1)].copy()
    fp_df = fp_df.sort_values(by="pred_probability", ascending=False).head(n_samples)

    # False Negatives: True=1, Pred=0
    fn_df = df_eval[(df_eval["true_label"] == 1) & (df_eval["pred_label"] == 0)].copy()
    fn_df = fn_df.sort_values(by="pred_probability", ascending=True).head(n_samples)

    return {
        "false_positives": fp_df,
        "false_negatives": fn_df,
        "total_false_positives": int(((df_eval["true_label"] == 0) & (df_eval["pred_label"] == 1)).sum()),
        "total_false_negatives": int(((df_eval["true_label"] == 1) & (df_eval["pred_label"] == 0)).sum()),
    }


def save_final_metrics_csv(
    metrics_dict: dict,
    model_name: str = "1D CNN",
    student_name: str = "Dilmith",
    save_path: str = "results/metrics/cnn_final_metrics.csv"
) -> pd.DataFrame:
    """
    Saves final standardized test evaluation metrics into CSV.
    Formatted specifically to enable seamless combining with Simple RNN, LSTM, and GRU metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    record = {
        "Model": model_name,
        "Student": student_name,
        "Accuracy": round(float(metrics_dict["accuracy"]), 4),
        "Precision": round(float(metrics_dict["precision"]), 4),
        "Recall": round(float(metrics_dict["recall"]), 4),
        "F1": round(float(metrics_dict["f1"]), 4),
        "ROC-AUC": round(float(metrics_dict["roc_auc"]), 4),
        "Training Time (s)": round(float(metrics_dict.get("training_time_sec", 0.0)), 2),
        "Parameter Count": int(metrics_dict.get("param_count", 0)),
        "Best Epoch": int(metrics_dict.get("best_epoch", 0)),
        "Embedding Dimension": int(metrics_dict.get("embedding_dim", 128)),
        "Filters": int(metrics_dict.get("filters", 128)),
        "Kernel Size": int(metrics_dict.get("kernel_size", 5)),
        "Dense Units": int(metrics_dict.get("dense_units", 64)),
        "Dropout": float(metrics_dict.get("dropout", 0.5)),
        "Batch Size": int(metrics_dict.get("batch_size", 32)),
    }

    df = pd.DataFrame([record])
    df.to_csv(save_path, index=False)
    return df
