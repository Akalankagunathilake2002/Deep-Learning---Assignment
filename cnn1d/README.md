# 1D CNN: Individual Contribution
**Student:** Dilmith  
**Course:** SE4050 – Deep Learning (2026)  
**Component:** 1D Convolutional Neural Network (1D CNN) for Sentiment Classification / Review Authenticity

---

## 1. Structure

```
cnn1d/
├── 01_Setup/                  CNN1D_Setup.ipynb
├── 02_Data_Audit/             CNN1D_Data_Audit.ipynb
├── 03_Dataset_and_Split/      CNN1D_Dataset_and_Split.ipynb
├── 04_EDA/                    CNN1D_EDA.ipynb
├── 05_Feature_Engineering/    CNN1D_Feature_Engineering.ipynb
├── 06_Class_Weights/          CNN1D_Class_Weights.ipynb
├── 07_CNN1D_Model/            CNN1D_Model.ipynb
├── 08_Training/               CNN1D_Training.ipynb
├── 09_Learning_Curves/        CNN1D_Learning_Curves.ipynb
├── 10_Final_Evaluation/       CNN1D_Final_Evaluation.ipynb
├── 11_Error_Analysis/         CNN1D_Error_Analysis.ipynb
├── 12_Robustness_Check/       CNN1D_Robustness_Check.ipynb
├── 13_Handover/               CNN1D_Handover.ipynb
├── CNN1D_model.ipynb          Standardized group 1D CNN master notebook
├── Dilmith_1D_CNN.ipynb       Dilmith's comprehensive 24-step research & hyperparameter experiment notebook
├── src/                       Reusable modular Python engineering components
│   ├── __init__.py
│   ├── cnn_model.py           Model building, training, and controlled hyperparameter experimentation
│   ├── preprocessing.py       Text cleaning, tokenization, padding, and EDA visualizers
│   ├── evaluation.py          Detailed test evaluation, confusion matrices, ROC curves, error analysis
│   └── utils.py               Seed reproducibility and directory management
├── results/                   Figures, metrics, predictions, and trained models
│   ├── cnn1d_model.keras      Trained champion Keras model (365,249 parameters)
│   ├── cnn1d_metrics.json     Standardized benchmark metrics (Accuracy: 94.38%, F1: 0.9455, ROC-AUC: 0.9702)
│   ├── cnn1d_history.csv      Epoch-by-epoch training/validation loss and accuracy
│   ├── cnn1d_training_info.json Training runtime and best epoch restoration
│   ├── cnn1d_test_predictions.csv Record-by-record test predictions and error flags
│   ├── cnn1d_seed_robustness.csv 6-seed stability verification (mean: 92.08% ± 1.45%)
│   ├── cnn_classification_report.txt Precision, recall, and F1 by class
│   ├── cnn_final_metrics.csv  Group comparison row matching SE4050 assignment schema
│   ├── experiments/           Controlled hyperparameter trial records (cnn_experiments.csv)
│   └── figures/               All publication-ready charts (loss, accuracy, confusion matrix, ROC, EDA)
└── docs/
    ├── CNN1D_report_section.md Full written academic report section for Dilmith's contribution
    └── CNN1D_theory_viva.md    Theory and oral examination (viva) defense preparation guide
```

---

## 2. Model Architecture & Rationale

Unlike recurrent neural networks (Simple RNN, LSTM, GRU) that maintain sequential internal states and process text token-by-token, a **1D Convolutional Neural Network (1D CNN)** applies one-dimensional sliding filters across contiguous word vector embeddings.

```
Input (300 token ids)
  → Embedding(5000, 64)        Dense continuous representation (320,000 params)
  → Conv1D(128, kernel_size=5) Sliding 5-gram filters with ReLU activation (41,088 params)
  → GlobalMaxPooling1D()       Extracts peak activation per filter (Position Invariance)
  → Dropout(0.5)               Mitigates co-adaptation of features
  → Dense(32, ReLU)            Non-linear feature combination (4,128 params)
  → Dropout(0.3)               Lighter regularization on dense head
  → Dense(1, Sigmoid)          Posterior probability P(Positive | Review) (33 params)
Total Parameters: 365,249 (100% trainable)
```

### Architectural Decisions & Theoretical Justifications
1. **1D Convolution (`Conv1D`):** Filters of size $k=5$ slide over the sequence to detect localized keyphrases, emotional hyperbole, and promotional idioms characteristic of reviews.
2. **ReLU Activation:** Applies $f(x) = \max(0, x)$ to introduce non-linearity without suffering from gradient vanishing issues common to sigmoid or hyperbolic tangent activations in deep networks.
3. **Global Max-Pooling (`GlobalMaxPooling1D`):** Reduces the temporal dimension by extracting the maximum activation per feature map. This guarantees **position invariance**—detecting a strong sentiment cue regardless of whether it appears in the opening sentence, body, or conclusion of the review.
4. **Computational Efficiency:** Convolutions across time steps can be computed completely in parallel without sequential backpropagation through time (BPTT), making training up to 2× faster than recurrent architectures.

---

## 3. Results Summary (Fair Comparison Benchmark)

Evaluated under the **exact identical experimental protocol** defined in `TEAM_GUIDE.md`:
- Fixed Stratified Split: 1,276 Train, 160 Validation, 160 Test
- Vocabulary: 5,000 tokens (Training split only)
- Maximum Sequence Length: 300 (pre-padded)
- Random Seed: 42

| Metric | 1D CNN (Dilmith) | GRU Reference (Akalanka) | Difference |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **94.38%** | 88.13% | **+6.25%** |
| **Precision** | **91.76%** | 91.78% | -0.02% |
| **Recall** | **97.50%** | 83.75% | **+13.75%** |
| **F1 Score** | **0.9455** | 0.8758 | **+0.0697** |
| **ROC-AUC** | **0.9702** | 0.9367 | **+0.0335** |
| **Training Time** | **6.11 s** | 13.68 s | **2.2× faster** |
| **Trainable Parameters** | 365,249 | 347,073 | ~5% diff |
| **Best Epoch** | Epoch 13 | Epoch 5 | — |

---

## 4. How to Run

### Option A: Standardized Step-by-Step Notebooks
Follow notebooks in order:
```bash
# Activate your environment
source .venv/bin/activate

# Open step notebooks
jupyter notebook cnn1d/01_Setup/CNN1D_Setup.ipynb
```
Run `01_Setup` through `13_Handover`. Each notebook is self-contained and automatically references the shared dataset in `dataset/deceptive-opinion.csv`.

### Option B: Consolidated All-In-One Notebook
Open `cnn1d/CNN1D_model.ipynb` for a single end-to-end execution.

### Option C: Dilmith's Master Research & Experimentation Notebook
Open `cnn1d/Dilmith_1D_CNN.ipynb` to inspect all 24 research sections, including EDA, baseline construction, controlled hyperparameter experiments (Compact vs Baseline vs High Capacity), and error analysis.

---

## 5. Teammate Handover

All output files required for the group report and consolidated table are located in:
- `cnn1d/results/cnn1d_metrics.json`
- `cnn1d/results/cnn_final_metrics.csv`
- `cnn1d/results/cnn1d_confusion_matrix.png`
- `cnn1d/results/cnn1d_roc_curve.png`
- `cnn1d/results/cnn1d_accuracy_curve.png`
- `cnn1d/results/cnn1d_loss_curve.png`
- `cnn1d/results/cnn1d_seed_robustness.csv`
