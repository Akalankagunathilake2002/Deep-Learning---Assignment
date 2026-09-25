# LSTM: Backend Model & Notebooks

Hotel review sentiment classification (SE4050 Deep Learning 2026). This directory contains the **Long Short-Term Memory (LSTM)** model implementation, including dataset audit, EDA, architecture design, training, evaluation, error analysis, and seed robustness checks.

---

## Folder Structure

```
lstm/
├── LSTM_model.ipynb             # Master all-in-one notebook (Data pipeline → Training → Evaluation)
├── README.md                    # Summary, structure, and usage guide
├── results/                     # Saved model weights, metrics JSON, predictions CSV, and plots
├── 00_Setup/                    # Step 0: Environment setup, imports, and shared helper functions
├── 01_Data_Audit/               # Step 1: Missing values, duplicates, encoding checks, and outlier audit
├── 02_Data_Labels_and_Split/    # Step 2: Label encoding, deduplication, and fixed 80/10/10 split
├── 03_EDA/                      # Step 3: Class balance, token distribution, negation analysis, distinctive words
├── 04_Feature_Engineering/      # Step 4: Tokenization, integer encoding, padding/truncating (MAX_LEN=300)
├── 05_Class_Weights/            # Step 5: Class weight calculation for training
├── 06_LSTM_Model/               # Step 6: Architecture definition & comparison (Unidirectional vs. Bidirectional)
├── 07_Training/                 # Step 7: Model training, early stopping, and validation selection
├── 08_Learning_Curves/          # Step 8: Training vs. validation loss and accuracy curves
├── 09_Final_Evaluation/         # Step 9: Test set evaluation, metrics, ROC curve, and confusion matrix
└── 10_Error_Analysis/           # Step 10: False positive / false negative inspection and error rates
```

---

## Master Notebook vs. Modular Step Notebooks

- **Master Notebook (`LSTM_model.ipynb`)**:
  - Contains the entire workflow sequentially in a single self-contained notebook.
  - Recommended for complete end-to-end training runs, reproducing final results, or executing the full assignment pipeline in one pass.

- **Modular Step Notebooks (`00_Setup` through `10_Error_Analysis`)**:
  - Each step is isolated in its own dedicated directory and can run independently.
  - **Granular Inspection & Debugging**: Allows targeted review of specific phases (e.g., auditing raw data, examining token distributions, or analyzing specific misclassifications) without re-running preceding compute-heavy training steps.
  - **Focused Experimentation**: Facilitates modular parameter tuning or individual component inspection with clean isolation of outputs.

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

### Option 1: Run the Full Pipeline (All-in-One)
1. Open [`lstm/LSTM_model.ipynb`](file:///c:/USH_/The%20White%20Hat/DL%20Assignment/Deep-Learning---Assignment/lstm/LSTM_model.ipynb).
2. Select your Python kernel with dependencies installed (`.venv`).
3. Run all cells to execute the entire end-to-end pipeline.

### Option 2: Run Individual Step Notebooks
1. Open any specific step notebook (e.g., [`lstm/01_Data_Audit/01_Data_Audit.ipynb`](file:///c:/USH_/The%20White%20Hat/DL%20Assignment/Deep-Learning---Assignment/lstm/01_Data_Audit/01_Data_Audit.ipynb) or [`lstm/09_Final_Evaluation/09_Final_Evaluation.ipynb`](file:///c:/USH_/The%20White%20Hat/DL%20Assignment/Deep-Learning---Assignment/lstm/09_Final_Evaluation/09_Final_Evaluation.ipynb)).
2. Select the `.venv` kernel.
3. Run all cells within that step notebook independently.

---

## Generated Deliverables (`results/`)

- `lstm_model.keras`: Serialized trained Keras model.
- `lstm_metrics.json`: Standardized JSON metric dictionary.
- `lstm_history.csv`: Per-epoch training/validation loss and accuracy.
- `lstm_test_predictions.csv`: Test set row predictions (`label`, `prob_positive`, `pred`).
- `lstm_accuracy_curve.png`, `lstm_loss_curve.png`, `lstm_confusion_matrix.png`, `lstm_roc_curve.png`, `lstm_probability_hist.png`: Evaluation figures.
- `lstm_eda_*.png`, `lstm_audit_*.png`: Data audit and EDA plots.
