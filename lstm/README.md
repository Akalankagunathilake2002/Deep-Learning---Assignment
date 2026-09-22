# LSTM: Individual Contribution

Hotel review sentiment classification (SE4050 Deep Learning 2026). This folder contains the **Long Short-Term Memory (LSTM)** model implementation, including an architectural comparison between **Unidirectional LSTM** and **Bidirectional LSTM** in accordance with Lab Sheet 05 (Task 3 / Q3: `IT23190870Q3.ipynb`).

---

## Folder Structure

```
lstm/
├── 01_Setup/                  LSTM_Setup.ipynb
├── 02_Data_Audit/             LSTM_Data_Audit.ipynb
├── 03_Dataset_and_Split/      LSTM_Dataset_and_Split.ipynb
├── 04_EDA/                    LSTM_EDA.ipynb
├── 05_Feature_Engineering/    LSTM_Feature_Engineering.ipynb
├── 06_Class_Weights/          LSTM_Class_Weights.ipynb
├── 07_LSTM_Model/             LSTM_Model.ipynb               (Architecture & Uni vs. Bi formulation)
├── 08_Training/               LSTM_Training.ipynb            (Uni vs. Bi experiment & winning model training)
├── 09_Learning_Curves/        LSTM_Learning_Curves.ipynb     (Loss & accuracy curves)
├── 10_Final_Evaluation/       LSTM_Final_Evaluation.ipynb    (Test evaluation, confusion matrix, ROC curve)
├── 11_Error_Analysis/         LSTM_Error_Analysis.ipynb      (Error breakdown by class, length, confidence)
├── 12_Robustness_Check/       LSTM_Robustness_Check.ipynb    (5-seed robustness evaluation)
├── 13_Handover/               LSTM_Handover.ipynb            (Table row for group report & checklist)
├── LSTM_model.ipynb           Master ALL-IN-ONE notebook     (Steps 0 to 13 in a single runnable notebook)
├── results/                   Generated figures, metrics JSON, test predictions, and trained model
└── docs/                      Report section and viva preparation notes
```

Every notebook is **self-contained**: it defines the group settings, fixed split row IDs, and data pipeline helpers locally without external dependencies. The only required file is `dataset/deceptive-opinion.csv`.

---

## How to Run the Notebooks

### Option A: Master All-in-One Notebook (`LSTM_model.ipynb`) — Recommended
This is the simplest way to run the entire pipeline from end to end:
1. **Locally in VS Code / Jupyter:**
   - Open `lstm/LSTM_model.ipynb`.
   - Select your Python environment (`.venv`).
   - Click **Run All**.
2. **In Google Colab:**
   - Upload `lstm/LSTM_model.ipynb` to Colab.
   - Click **Runtime → Run all**.
   - When prompted, upload `deceptive-opinion.csv` from your computer (only required once).

### Option B: Single-Task Step Notebooks (`01_Setup` to `13_Handover`)
Run each numbered notebook sequentially from `01_Setup/LSTM_Setup.ipynb` through `13_Handover/LSTM_Handover.ipynb`:
* Any notebook can be run on its own: Notebooks 09 through 13 automatically load the trained model from `results/lstm_model.keras` if Notebook 08 has already run, or train it automatically on demand.

---

## Fine-Tuning & Hyperparameter Exploration Guide

Throughout the notebooks, specific parameters are clearly marked with `# <-- Modify ...` comments to allow straightforward experimentation:

| Parameter | Location in Code | Default | How to Tune |
|---|---|---|---|
| **Embedding Dimension** | `layers.Embedding(VOCAB_SIZE, 64)` | 64 | Change to 32 (lighter, less overfitting) or 128 (richer semantic capture). |
| **LSTM Hidden Units** | `layers.LSTM(64)` | 64 | Change to 32 or 128 to test model capacity. |
| **Bidirectional Wrapping** | `layers.Bidirectional(layers.LSTM(...))` | Comparison in Step 7 | Toggle between Unidirectional and Bidirectional sequence reading. |
| **Dropout Regularization** | `layers.Dropout(0.5)` | 0.5, 0.3 | Increase (e.g. 0.5/0.4) if validation loss starts rising early; decrease (e.g. 0.3/0.2) if underfitting. |
| **Stacked LSTM Layers** | `layers.LSTM(64, return_sequences=True)` | Single layer | Add a second LSTM layer before the dense classification head. |
| **Dense Head Units** | `layers.Dense(32, activation="relu")` | 32 | Modify intermediate non-linear projection units (e.g., 16, 64). |
| **Training Epochs** | `MAX_EPOCHS = 30` | 30 | Increase to 40 or 50 if the model has not converged when early stopping occurs. |
| **Batch Size** | `BATCH_SIZE = 32` | 32 | Try 16 (more frequent updates) or 64 (smoother gradients). |
| **Learning Rate** | `LEARNING_RATE = 1e-3` | 1e-3 | Adjust Adam step size (e.g., 5e-4, 1e-4). |
| **EarlyStopping Patience** | `PATIENCE = 5` | 5 | Change to 3 or 7 to alter sensitivity to validation loss fluctuations. |

---

## Testing Custom Phrases

In `LSTM_model.ipynb` (Step 13), you can test review sentences outside the dataset:
* **Pre-Loaded Battery**: Evaluates 8 diverse hardcoded sentences covering strong positive, strong negative, tricky negation (*"not clean at all"*), subtle disappointment, and short one-liners.
* **Interactive Input**: Run the last cell and type any review into the prompt to see real-time classification, probability $P(positive)$, and confidence score.

---

## Deliverables Generated in `results/`

When you run Notebook 08 and Notebook 10, the following group-standard files are created in `lstm/results/`:
* `lstm_model.keras`: Serialized trained Keras model.
* `lstm_history.csv`: Per-epoch training/validation loss and accuracy.
* `lstm_training_info.json`: Training time, epochs run, and best epoch index.
* `lstm_metrics.json`: Standardized JSON metric dictionary matching `TEAM_GUIDE.md`.
* `lstm_test_predictions.csv`: Row-by-row test predictions (`row_id, label, prob_positive, pred`).
* `lstm_accuracy_curve.png` & `lstm_loss_curve.png`: Learning curves.
* `lstm_confusion_matrix.png`, `lstm_roc_curve.png`, `lstm_probability_hist.png`: Test evaluation figures.
* `lstm_seed_robustness.csv`: 5-seed stability validation results.
