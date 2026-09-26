# Simple RNN: Individual Model Contribution

**SE4050 – Deep Learning 2026**  
**Project:** Hotel Review Sentiment Classification Using Deep Learning  
**Individual Contribution:** Simple RNN (Sequence-Averaged Elman Recurrent Neural Network)

---

## 1. Overview & Architectural Role

In this group research study, four distinct deep learning architectures are evaluated on an identical hotel review benchmark dataset under strictly controlled conditions:
1. **Simple RNN** (*This contribution*)
2. **LSTM** (Long Short-Term Memory)
3. **GRU** (Gated Recurrent Unit)
4. **1D CNN** (One-Dimensional Convolutional Neural Network)

The **Simple RNN** serves as the foundational sequential baseline. Based on the classic Elman recurrence relation (Elman, 1990):
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Unlike LSTM and GRU, Simple RNN lacks internal gating mechanisms (input, forget, or update gates) and has no additive cell state. This makes it computationally lightweight (only 8,256 recurrent parameters for 64 units). In the final official architecture (**R3-3**), the model emits hidden states for all sequence timesteps and applies temporal global average pooling (`GlobalAveragePooling1D()`), overcoming the single-state recency bottleneck of the traditional Elman architecture while preserving authentic un-gated recurrence.

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
├── 07_SimpleRNN_Model/        RNN_Model.ipynb & RNN_Hyperparameter_Experiments.ipynb
├── 08_Training/               RNN_Training.ipynb
├── 09_Learning_Curves/        RNN_Learning_Curves.ipynb
├── 10_Final_Evaluation/       RNN_Final_Evaluation.ipynb
├── 11_Error_Analysis/         RNN_Error_Analysis.ipynb
├── 12_Robustness_Check/       RNN_Robustness_Check.ipynb
├── 13_Handover/               RNN_Handover.ipynb
├── SimpleRNN_model.ipynb      Master ALL-IN-ONE notebook (recommended for Colab)
├── results/                   Official model, metrics, figures, and CSV records
│   ├── simple_rnn_model.keras             Official promoted model (R3-3)
│   ├── simple_rnn_metrics.json            Official test-set metrics
│   ├── simple_rnn_test_predictions.csv    Official row-by-row predictions
│   ├── simple_rnn_history.csv             Training & validation curves (18 epochs)
│   ├── baseline_archive/                  Preserved original baseline artifacts
│   ├── simple_rnn_tuning_round2.csv       Round 2 experiment metrics
│   ├── simple_rnn_tuning_round3.csv       Round 3 experiment metrics
│   └── r3_3_final_test_metrics.json       R3-3 final test evaluation record
```

---

## 3. Controlled Experimentation & Model Selection

### A. The Original Baseline Model
* **Architecture:** `Embedding(5000, 64) -> SimpleRNN(64, final step) -> Dense(32) -> Dense(1)`
* **Parameters:** 330,369
* **Original Test Results:**
  * Test Accuracy: **85.00%** ($136/160$)
  * Test Precision: **90.00%** | Test Recall: **78.75%** | Test F1: **0.8400** | ROC-AUC: **0.8961**
  * Confusion Matrix: $\text{TN}=73, \text{FP}=7, \text{FN}=17, \text{TP}=63$ (24 errors total)
* **Diagnosis:** The baseline suffered from an asymmetric error profile with high false negatives ($17$ FNs vs $7$ FPs), indicating that the model frequently doubted positive reviews.

### B. Round 2 Hyperparameter Experiments (T1–T5)
A controlled grid varying learning rate ($\eta \in \{0.001, 0.0005\}$), recurrent dropout ($0.4 \to 0.5$), hidden units ($64 \to 96$), and embedding dimensions ($64 \to 96$) was evaluated strictly on the validation set ($N=160$).
* **Finding:** No configuration demonstrated a meaningful improvement over the official baseline:
  * Widening recurrent units to 96 (T3, T5) worsened validation loss ($0.4185 - 0.4917$) due to state-transition overfitting.
  * Widening embeddings to 96 (T4) yielded identical validation accuracy ($92.50\%$) and negligible loss difference ($-0.0028$) at the cost of a **+49% parameter expansion** (492,417 weights).
* **Decision:** The baseline was retained.

### C. Systematic Error Analysis
An audit of training and validation errors revealed two fundamental structural limitations of the single-state Elman RNN:
1. **Severe Recency Bias:** Because gradients attenuate over sequences of up to 300 steps ($\approx W_{hh}^{300}$), the final hidden state $h_T$ was heavily dominated by the final 15–20 tokens. Reviews with divergent concluding remarks (e.g., negative reviews ending with *"tech support was helpful"* or positive reviews ending with *"my only regret is not staying longer"*) were misclassified.
2. **Information Bottleneck:** Compressing up to 300 words into a single 64-dimensional vector $h_T$ caused earlier clauses to be washed out.

### D. Round 3 Controlled Architectural Experiments (R3-1 to R3-5)
Five targeted candidate models within the genuine Simple RNN family were formulated:
* **R3-1:** Baseline + $L_2$ Regularization ($10^{-4}$) $\to$ Underfit ($60.62\%$ val accuracy).
* **R3-2:** Bidirectional SimpleRNN(32) $\to$ High precision ($97.18\%$), but low recall ($86.25\%$).
* **R3-3 (Sequence-Averaged):** `SimpleRNN(64, return_sequences=True) + GlobalAveragePooling1D()` $\to$ **Selected Winner**:
  * Validation Accuracy: **94.38%** ($151/160$, $+3$ reviews correct)
  * Validation Loss: **0.137641** (**$-41.9\%$ loss reduction**)
  * Validation F1: **0.943396**
  * Validation Confusion Matrix: $\text{TN}=76, \text{FP}=4, \text{FN}=5, \text{TP}=75$
  * Resolved 10 of the 12 baseline error cases while maintaining the exact 330,369 parameter count.
* **R3-4:** Baseline with `MAX_LEN=200` $\to$ Truncation hurt long reviews ($88.75\%$ val accuracy).
* **R3-5:** Bidirectional SimpleRNN(32) + $L_2$ Reg $\to$ $93.75\%$ val accuracy, $0.1813$ loss.

---

## 4. Final Official Model Specification (Promoted R3-3)

```
Layer (type)                     Output Shape          Param #    Activation / Config
=====================================================================================
embedding (Embedding)            (None, 300, 64)       320,000    Vocab: 5000, Dim: 64
simple_rnn (SimpleRNN)           (None, 300, 64)         8,256    tanh, return_sequences=True, drop=0.5
global_average_pooling1d         (None, 64)                  0    Arithmetic mean over time
dense (Dense)                    (None, 32)              2,080    ReLU
dropout (Dropout)                (None, 32)                  0    Dropout rate: 0.3
dense_1 (Dense)                  (None, 1)                  33    Sigmoid
=====================================================================================
Total Trainable Parameters: 330,369 (100% trainable, 1.26 MB)
```

### Final Test Set Evaluation Results (Single Evaluation on $N=160$ Test Set)

| Metric | Baseline Test Result | Final Promoted Model (R3-3) | Delta ($\Delta$) |
|---|:---:|:---:|:---:|
| **Test Accuracy** | $85.00\%$ ($136/160$) | **95.00%** ($152/160$) | **$+10.00\%$** |
| **Test Precision** | $90.00\%$ | **95.00%** | $+5.00\%$ |
| **Test Recall** | $78.75\%$ | **95.00%** | **$+16.25\%$** |
| **Test F1 Score** | $0.8400$ | **0.9500** | **$+0.1100$** |
| **Test ROC-AUC** | $0.8961$ | **0.9795** | **$+0.0834$** |
| **Test Binary Loss** | $0.3705$ | **0.184034** | **$-50.3\%$** |
| **Confusion Matrix** | $\text{TN}=73, \text{FP}=7$<br>$\text{FN}=17, \text{TP}=63$ | **$\text{TN}=76, \text{FP}=4$<br>$\text{FN}=4, \text{TP}=76$** | **Symmetric class performance** |
| **Total Test Errors** | 24 errors | **8 errors** | **$66.7\%$ error reduction** |

*95% Bootstrap Confidence Intervals (1,000 iterations):*
* **Accuracy:** $[0.9125, 0.9813]$
* **F1 Score:** $[0.9139, 0.9818]$
* **ROC-AUC:** $[0.9572, 0.9955]$

> **Generalization Note:** While the model achieves high empirical accuracy ($95.00\%$), it does not claim 100% classification certainty. The model can still make errors on unseen reviews featuring highly dense negations, extreme sarcasm, or domain-atypical phrase structures.

---

## 5. How to Run

### Option A: Master All-In-One Notebook (`SimpleRNN_model.ipynb`) — Recommended
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `simple_rnn/SimpleRNN_model.ipynb`.
3. Choose **Runtime → Run all**.
4. The notebook will execute the complete pipeline end-to-end.

### Option B: Step Notebooks (`01_Setup` to `13_Handover`)
1. Activate your local environment (`.venv`).
2. Run notebooks sequentially from `01_Setup/RNN_Setup.ipynb` through `13_Handover/RNN_Handover.ipynb`.
3. Step notebooks 09 through 13 automatically load the promoted model from `results/simple_rnn_model.keras`.

---

## 6. Generated Deliverables (`simple_rnn/results/`)

* `simple_rnn_model.keras`: Serialized official R3-3 trained Keras model (330,369 weights).
* `simple_rnn_metrics.json`: Standardized JSON test metrics matching the group schema.
* `simple_rnn_test_predictions.csv`: Per-review test predictions (`row_id, label, hotel, deceptive, prob_positive, prediction`).
* `simple_rnn_history.csv`: Per-epoch training & validation metrics for the 18 epochs run.
* `simple_rnn_training_info.json`: Training duration, epochs run, and best epoch index.
* `simple_rnn_confusion_matrix.png`: Heatmap showing $\text{TN}=76, \text{FP}=4, \text{FN}=4, \text{TP}=76$.
* `simple_rnn_roc_curve.png`: Test ROC curve ($\text{AUC} = 0.980$).
* `simple_rnn_probability_hist.png`: Confidence distribution histogram by true class.
* `simple_rnn_accuracy_curve.png` & `simple_rnn_loss_curve.png`: Learning curves.
* `baseline_archive/`: Secure backup of the original baseline model, metrics, and figures.
