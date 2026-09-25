# 1D CNN Model: Hotel Review Sentiment & Authenticity Classification

**Student:** Dilmith  
**Course:** SE4050 – Deep Learning  
**Component:** 1D Convolutional Neural Network (1D CNN)  

*Figure files are in `cnn1d/results/`. All metrics and parameters are computed on the shared, stratified 80/10/10 split (seed 42) in full compliance with the group protocol in `TEAM_GUIDE.md`.*

---

## 1. Role of the 1D CNN in the Group Study

In our comparative evaluation of neural architectures for hotel review sentiment classification (Simple RNN, LSTM, GRU, 1D CNN), the **1D Convolutional Neural Network (1D CNN)** serves as the non-recurrent, spatial/temporal feature extractor. While recurrent architectures (RNN, LSTM, GRU) process sequence elements sequentially and maintain internal hidden memories, a 1D CNN slides one-dimensional discrete filters across contiguous word vector embeddings.

This architectural paradigm offers distinct theoretical and practical advantages:
1. **Parallelized Temporal Convolutions:** Word embeddings across all time steps are processed concurrently through matrix multiplications on tensor hardware, avoiding the sequential bottleneck of backpropagation through time (BPTT).
2. **N-gram Keyphrase Detection:** Convolutional kernels of width $k=5$ act as soft n-gram detectors that extract localized semantic cues (e.g., *"not clean at all"*, *"best hotel in chicago"*), regardless of where they appear in the sequence.
3. **Position Invariance via Global Max-Pooling:** The pooling mechanism collapses the sequence length by selecting the strongest activation per filter, ensuring sentiment signals are captured irrespective of their temporal index.

---

## 2. Experimental Data and Fair Comparison Protocol

To ensure an academically rigorous and strictly controlled comparison, all four models were trained, validated, and evaluated under identical conditions:

| Parameter | Value | Academic Rationale |
| :--- | :--- | :--- |
| **Dataset** | Deceptive Opinion Spam Corpus | 1,600 hotel reviews of 20 Chicago hotels (1,596 after deduplication) |
| **Labeling** | `polarity` column | $0 = \text{Negative}$, $1 = \text{Positive}$ (balanced: 640 train pos / 636 train neg) |
| **Deduplication** | 4 identical review pairs removed | Prevents identical reviews leaking across training, validation, and test sets |
| **Data Partitioning** | Stratified 80 / 10 / 10 | **1,276 Train, 160 Validation, 160 Test** (fixed by row IDs, seed 42) |
| **Vocabulary** | Top 5,000 token IDs | Built strictly from the **training set only** (`<PAD>=0`, `<OOV>=1`) |
| **Sequence Length** | `MAX_LEN = 300` | Pre-padded with zeros; post-truncated (preserves 94% of reviews in full) |
| **Embedding** | `Embedding(5000, 64)` | 64-dimensional dense vectors learned from scratch; no pretrained vectors |
| **Loss & Optimizer** | Binary Cross-Entropy / Adam | Learning rate $\eta = 10^{-3}$, batch size $B=32$ |
| **Early Stopping** | Patience = 5 on $\text{val\_loss}$ | Restores best weights to prevent overfitting on the small training set |

---

## 3. Architecture Specification & Parameter Accounting

```
Input Sequence: (300 token IDs)
  │
  ├── 1. Embedding Layer: Embedding(5000, 64)
  │      Output Shape: (None, 300, 64)
  │      Parameters: 5,000 × 64 = 320,000
  │
  ├── 2. 1D Convolution: Conv1D(128 filters, kernel_size=5, activation='relu')
  │      Output Shape: (None, 296, 128)
  │      Parameters: (5 × 64 × 128) + 128 = 41,088
  │
  ├── 3. Temporal Pooling: GlobalMaxPooling1D()
  │      Output Shape: (None, 128)
  │      Parameters: 0
  │
  ├── 4. Dropout Regularization: Dropout(rate=0.5)
  │      Output Shape: (None, 128)
  │      Parameters: 0
  │
  ├── 5. Dense Hidden Layer: Dense(32 units, activation='relu')
  │      Output Shape: (None, 32)
  │      Parameters: (128 × 32) + 32 = 4,128
  │
  ├── 6. Dropout Regularization: Dropout(rate=0.3)
  │      Output Shape: (None, 32)
  │      Parameters: 0
  │
  └── 7. Output Classification: Dense(1 unit, activation='sigmoid')
         Output Shape: (None, 1)
         Parameters: (32 × 1) + 1 = 33

Total Parameters: 365,249 (100% Trainable)
```

The embedding represents 87.6% of the total network capacity (320,000 parameters), while the convolutional feature extractor and classification head comprise 45,249 parameters. This matches the capacity of the reference GRU (347,073 total parameters, ~5% delta), isolating architecture as the sole independent variable.

---

## 4. Empirical Test Results & Evaluation

The final champion 1D CNN model was evaluated **exactly once** on the unseen 160-review test set at the default decision threshold $\tau = 0.5$:

| Metric | 1D CNN (Dilmith) | GRU Reference | Comparison / Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **94.38%** | 88.13% | **+6.25%** |
| **Precision (Positive)** | **91.76%** | 91.78% | -0.02% |
| **Recall (Positive)** | **97.50%** | 83.75% | **+13.75%** |
| **F1-Score (Positive)** | **0.9455** | 0.8758 | **+0.0697** |
| **ROC-AUC** | **0.9702** | 0.9367 | **+0.0335** |
| **Negative Precision** | **97.33%** | 85.06% | **+12.27%** |
| **Negative Recall** | **91.25%** | 92.50% | -1.25% |
| **Negative F1** | **0.9419** | 0.8862 | **+0.0557** |
| **Macro Average F1** | **0.9437** | 0.8810 | **+0.0627** |
| **Training Duration** | **6.11 s** | 13.68 s | **2.2× faster convergence** |
| **Epochs to Best Val Loss** | 13 (stopped at 18) | 5 (stopped at 10) | Stable feature learning |

### Confusion Matrix Breakdown (Test Set, N=160)
- **True Negatives (TN):** 73 (correctly classified negative hotel reviews)
- **False Positives (FP):** 7 (negative reviews predicted as positive)
- **False Negatives (FN):** 2 (positive reviews predicted as negative)
- **True Positives (TP):** 78 (correctly classified positive hotel reviews)

The 1D CNN demonstrated exceptional positive recall (97.5%), missing only 2 positive reviews out of 80 in the holdout test set.

### 95% Bootstrap Confidence Intervals (2,000 resamples)
- **Accuracy:** $[90.63\%, 97.50\%]$
- **F1-Score:** $[0.9054, 0.9773]$
- **ROC-AUC:** $[0.9402, 0.9937]$

---

## 5. Multi-Seed Robustness Analysis

To verify that the model's high performance was not an artifact of random weight initialization, the complete training and evaluation cycle was repeated across **6 distinct random seeds** (Seed 42, 1, 2, 3, 4, 5):

| Seed | Stopped Epoch | Best Val Acc | Test Accuracy | Test F1 | Test ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42 (Champion)** | 18 | 94.38% | 94.38% | 0.9455 | 0.9702 |
| **1** | 12 | 93.13% | 90.00% | 0.9012 | 0.9570 |
| **2** | 17 | 93.13% | 92.50% | 0.9268 | 0.9631 |
| **3** | 20 | 95.00% | 91.25% | 0.9146 | 0.9577 |
| **4** | 15 | 93.13% | 91.88% | 0.9222 | 0.9564 |
| **5** | 19 | 94.38% | 92.50% | 0.9268 | 0.9642 |
| **Mean ± Std** | **16.8 ± 3.0** | **93.86% ± 0.81%** | **92.08% ± 1.45%** | **0.9228 ± 0.0148** | **0.9614 ± 0.0053** |

The narrow spread ($\pm 1.45\%$ in test accuracy, $\pm 0.0053$ in ROC-AUC) confirms that the 1D CNN architecture exhibits high initialization stability.

---

## 6. Discussion and Architectural Takeaways

1. **Why 1D CNN Outperformed GRU:** Hotel reviews predominantly express sentiment through localized phrases (e.g., *"unfriendly front desk"*, *"view of the river was magnificent"*) rather than complex, long-range sentence dependencies spanning hundreds of tokens. The 1D CNN with kernel size 5 directly targets these 5-gram windows.
2. **Speed & Efficiency:** By eliminating recurrent step-by-step dependency loops, the 1D CNN trained in 6.11 seconds (2.2× faster than the GRU), proving ideal for deployment in latency-critical environments.
3. **Position Invariance Advantage:** Global max-pooling successfully aggregated features regardless of whether the reviewer summarized their sentiment in the opening lines or the concluding paragraph.
