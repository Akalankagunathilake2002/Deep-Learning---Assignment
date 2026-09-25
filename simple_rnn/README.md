# Simple RNN: Individual Model Contribution

**SE4050 – Deep Learning 2026**
**Project:** Hotel Review Sentiment Classification Using Deep Learning
**Individual Contribution:** Simple RNN (Vanilla Elman Recurrent Neural Network)

---

## 1. Overview & Architectural Role

In this group research study, four distinct deep learning architectures are evaluated on an identical hotel review benchmark dataset under strictly controlled conditions:
1. **Simple RNN** (*This contribution*)
2. **LSTM** (Long Short-Term Memory)
3. **GRU** (Gated Recurrent Unit)
4. **1D CNN** (One-Dimensional Convolutional Neural Network)

The **Simple RNN** serves as the foundational sequential baseline. Based on the classic Elman architecture (Elman, 1990), it processes text token by token, maintaining a recurrence relation:
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Unlike LSTM and GRU, Simple RNN lacks internal gating mechanisms (input, forget, or update gates) and has no additive cell state. This makes it computationally lightweight (only 8,256 recurrent parameters for 64 units), but inherently susceptible to the **vanishing gradient problem** across long token spans.

---

## 2. Directory Structure

```
simple_rnn/
├── 01_Setup/                  RNN_Setup.ipynb
├── 02_Data_Audit/             RNN_Data_Audit.ipynb
├── 03_Dataset_and_Split/      RNN_Dataset_and_Split.ipynb
├── 04_EDA/                    RNN_EDA.ipynb
├── 05_Feature_Engineering/    RNN_Feature_Engineering.ipynb
├── 06_Class_Weights/          RNN_Class_Weights.ipynb
├── 07_SimpleRNN_Model/        RNN_Model.ipynb
├── 08_Training/               RNN_Training.ipynb
├── 09_Learning_Curves/        RNN_Learning_Curves.ipynb
├── 10_Final_Evaluation/       RNN_Final_Evaluation.ipynb
├── 11_Error_Analysis/         RNN_Error_Analysis.ipynb
├── 12_Robustness_Check/       RNN_Robustness_Check.ipynb
├── 13_Handover/               RNN_Handover.ipynb
├── SimpleRNN_model.ipynb      Master ALL-IN-ONE notebook (recommended for Colab)
├── results/                   Output models, figures, metrics, and CSV records
└── docs/
    ├── SimpleRNN_report_section.md   Formal report chapter (with metric placeholders)
    └── SimpleRNN_theory_viva.md      Comprehensive theory & Viva Voce Q&A guide
```

Every step notebook is **100% self-contained**: it embeds all necessary group settings, data preparation helpers, fixed row split IDs, and plotting styles. Step notebooks 09 through 13 automatically load the trained model from `results/simple_rnn_model.keras` or train it automatically on demand.

---

## 3. How to Run

### Option A: Master All-In-One Notebook (`SimpleRNN_model.ipynb`) — Recommended
This is the fastest, cleanest way to execute the complete pipeline in Google Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `simple_rnn/SimpleRNN_model.ipynb`.
3. Choose **Runtime → Run all**.
4. If `deceptive-opinion.csv` is not detected in `/content` or `dataset/`, Colab will present an interactive file picker: upload `deceptive-opinion.csv`.
5. The notebook will run all 16 sections end-to-end (takes ~30-45 seconds total on CPU/T4).
6. Download the generated `results/` folder from the Colab file browser.

### Option B: Step Notebooks (`01_Setup` to `13_Handover`)
To run or present each step individually:
1. Activate your local environment (`.venv`).
2. Run notebooks sequentially from `01_Setup/RNN_Setup.ipynb` through `13_Handover/RNN_Handover.ipynb`.
3. The dataset is automatically located at `../dataset/deceptive-opinion.csv` or `../../dataset/deceptive-opinion.csv`.

---

## 4. Hyperparameter & Architecture Specification

| Component / Hyperparameter | Value | Rationale & Locking Rule |
|---|---|---|
| **Input Shape** | `(300,)` | Max sequence length of 300 integer token IDs |
| **Embedding Layer** | `Embedding(5000, 64)` | 320,000 parameters. Trained from scratch (no pretrained embeddings) |
| **Recurrent Layer** | `SimpleRNN(64)` | 64 hidden units, `tanh` activation. 8,256 parameters |
| **Recurrent Dropout** | `Dropout(0.5)` | Applied after Simple RNN to prevent memorization on small corpus |
| **Dense Hidden Layer** | `Dense(32, activation="relu")` | 2,080 parameters. Non-linear feature combination |
| **Dense Dropout** | `Dropout(0.3)` | Regularization before decision boundary |
| **Output Layer** | `Dense(1, activation="sigmoid")` | 33 parameters. Outputs probability $P(\text{positive}) \in [0, 1]$ |
| **Total Parameters** | **330,369** | 100% trainable. Embedding accounts for 96.86% of total capacity |
| **Optimizer / LR** | Adam, $\eta = 10^{-3}$ | Standard adaptive learning rate |
| **Loss Function** | Binary Cross-Entropy | Canonical loss for binary probabilistic output |
| **Batch Size** | 32 | ~40 batches per epoch on 1,276 training samples |
| **Max Epochs** | 30 | Upper bound limit |
| **EarlyStopping** | Monitor `val_loss`, patience 5 | Prevents overfitting and restores weights from best epoch |
| **Split Strategy** | 80/10/10 Stratified | 1,276 train / 160 validation / 160 test |

---

## 5. Experimentation & Fine-Tuning Guide

To test alternative configurations, look for the `# <-- Modify` tags in Notebook 07/08 or Section 9/10 of `SimpleRNN_model.ipynb`:
- **Embedding Dimension (`EMBED_DIM`)**: Default 64. Try 32 (lighter, faster) or 128 (richer semantic representation).
- **Recurrent Units**: Default 64. Adjust to 32 or 128 to analyze model capacity vs. overfitting.
- **Patience**: Default 5. Changing to 3 stops earlier; changing to 7 gives more opportunity to escape local plateaus.
- **Learning Rate**: Default 1e-3. Try 5e-4 for smoother gradient convergence.

---

## 6. Generated Result Deliverables (`simple_rnn/results/`)

Running the training and evaluation notebooks populates `simple_rnn/results/` with:
- `simple_rnn_model.keras`: Serialized trained Keras model.
- `simple_rnn_history.csv`: Per-epoch training/validation loss and accuracy.
- `simple_rnn_training_info.json`: Training duration, epochs run, and best epoch index.
- `simple_rnn_metrics.json`: Standardized JSON metrics matching `TEAM_GUIDE.md`.
- `simple_rnn_test_predictions.csv`: Row-by-row test set predictions (`row_id`, `label`, `prob_positive`, `prediction`).
- `simple_rnn_accuracy_curve.png` & `simple_rnn_loss_curve.png`: Learning curves with best epoch marker.
- `simple_rnn_confusion_matrix.png`: Test set confusion matrix heatmap.
- `simple_rnn_roc_curve.png`: ROC curve with AUC score and operating threshold marker.
- `simple_rnn_probability_hist.png`: Confidence distribution histogram by true class.
- `simple_rnn_seed_robustness.csv`: Stability metrics over 6 random seeds.
- `simple_rnn_audit_length_outliers.png`, `simple_rnn_eda_class_balance.png`, `simple_rnn_eda_distinctive_words.png`, `simple_rnn_eda_review_length.png`: Audit and EDA figures.
