# LSTM: Backend Model & Notebooks

Hotel review sentiment classification (SE4050 Deep Learning 2026). This directory contains the **Long Short-Term Memory (LSTM)** model implementation, including dataset audit, EDA, architecture design, training, evaluation, error analysis, and seed robustness checks.

---

## Folder Structure

```
lstm/
├── LSTM_model.ipynb    # Master self-contained notebook (Data pipeline → Training → Evaluation)
├── README.md           # Quick summary and usage guide
└── results/            # Trained model weights, evaluation metrics, CSVs, and plots
```

---

## Key Results (Seed 42)

Evaluated on 160 unseen test reviews (80 positive, 80 negative):

- **Accuracy**: 93.13% (95% CI: 88.75% – 96.88%)
- **Precision**: 93.67%
- **Recall**: 92.50%
- **F1-Score**: 93.08%
- **ROC-AUC**: 0.953
- **Trainable Parameters**: 355,137 (`LSTM(64)` + Embedding & Dense Head)
- **Epochs Run**: 12 (Best epoch 7 restored via EarlyStopping)

---

## How to Run

1. Open `lstm/LSTM_model.ipynb` in VS Code or Jupyter.
2. Select your Python kernel with TensorFlow installed (`.venv`).
3. Run all cells to execute data preparation, model training, and evaluation.

---

## Generated Deliverables (`results/`)

- `lstm_model.keras`: Serialized trained Keras model.
- `lstm_metrics.json`: Standardized JSON metric dictionary.
- `lstm_history.csv`: Per-epoch training/validation loss and accuracy.
- `lstm_test_predictions.csv`: Test set row predictions (`label`, `prob_positive`, `pred`).
- `lstm_accuracy_curve.png`, `lstm_loss_curve.png`, `lstm_confusion_matrix.png`, `lstm_roc_curve.png`, `lstm_probability_hist.png`: Evaluation figures.
- `lstm_eda_*.png`, `lstm_audit_*.png`: Data audit and EDA plots.
