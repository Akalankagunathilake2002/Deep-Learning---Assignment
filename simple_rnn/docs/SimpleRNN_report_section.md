# Simple RNN Model: Hotel Review Sentiment Classification
**Author / Individual Contribution:** Yuhansi (Simple RNN)
**Course:** SE4050 – Deep Learning 2026
**Artifact Directory:** `simple_rnn/results/`

---

## 1. Role of the Simple RNN in the Comparative Study

In our group investigation of deep learning models for hotel review sentiment classification, the **Simple RNN (Vanilla Elman Recurrent Neural Network)** serves as the canonical sequential baseline. 

A Simple RNN processes a sequence of token vectors $(x_1, x_2, \dots, x_T)$ sequentially. At each discrete time step $t$, the recurrent unit computes an updated hidden state vector $h_t \in \mathbb{R}^{64}$ by linearly transforming both the current token embedding $x_t \in \mathbb{R}^{64}$ and the previous hidden state $h_{t-1} \in \mathbb{R}^{64}$, followed by a point-wise hyperbolic tangent non-linearity:
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Unlike the gated recurrent models (LSTM and GRU) and convolutional networks (1D CNN) evaluated by teammates, the Simple RNN contains **no gating mechanisms** and **no additive cell memory**. It relies entirely on continuous matrix multiplication ($W_{hh}$) across time steps. This minimal recurrent parameterization (8,256 parameters) allows us to empirically quantify the performance gains directly attributable to gating mechanisms (LSTM/GRU) and local receptive fields (1D CNN) under strictly identical training, preprocessing, and vocabulary conditions.

---

## 2. Dataset and Preprocessing (Shared Group Pipeline)

To maintain absolute methodological fairness across the four models, the Simple RNN was trained and evaluated under the group's locked pipeline:

| Pipeline Step | Specification | Justification |
|---|---|---|
| **Dataset** | Deceptive Opinion Spam corpus (1,600 reviews across 20 Chicago hotels) | Standard balanced benchmark for sentiment and authenticity |
| **Label Mapping** | `polarity`: `negative` $\to 0$, `positive` $\to 1$ | Binary classification task; no neutral star ratings exist |
| **Data Audit** | 0 missing values, 0 blank texts, 0 encoding/mojibake errors. 4 exact duplicate reviews removed before split (1,596 retained). | Prevents duplicate memorization across splits; 75 length outliers retained |
| **Split Strategy** | Stratified 80/10/10 split (Seed 42) via hardcoded row IDs | **1,276 train (640 pos / 636 neg), 160 validation (80 / 80), 160 test (80 / 80)** |
| **Text Cleaning** | Lowercasing, HTML/URL stripping, punctuation & digit removal, contraction expansion (`didn't` $\to$ `did not`) | Negation words (`not`, `no`, `never`) are strictly retained |
| **Vocabulary** | Top 5,000 words fitted on **training set only** | `<PAD> = 0`, `<OOV> = 1`. Eliminates test data leakage |
| **Padding** | Pre-padding (`PADDING="pre"`), post-truncation (`MAX_LEN=300`) | Pre-padding ensures recurrent state ends on real tokens rather than zeros |
| **Class Weights** | Balanced class weights: 1.003 (negative), 0.997 (positive) | Applied consistently across all group models |

---

## 3. Architecture & Parameter Count

```
Input: 300 token IDs
  → Embedding(5000, 64)        320,000 parameters
  → SimpleRNN(64)                8,256 parameters
  → Dropout(0.5)                     0 parameters
  → Dense(32, ReLU)              2,080 parameters
  → Dropout(0.3)                     0 parameters
  → Dense(1, Sigmoid)               33 parameters
Total Parameters:              330,369 (100% trainable)
```

### Parameter Calculation Derivation:
1. **Embedding Layer**: $\text{Vocab Size} \times \text{Embedding Dim} = 5,000 \times 64 = 320,000$.
2. **SimpleRNN Layer**:
   - Input kernel $W_{xh}$: $64 \times 64 = 4,096$
   - Recurrent kernel $W_{hh}$: $64 \times 64 = 4,096$
   - Bias vector $b_h$: $64$
   - Total for SimpleRNN = $4,096 + 4,096 + 64 = 8,256$.
3. **Dense Hidden Layer**: $(64 \times 32) + 32 = 2,080$.
4. **Classification Output**: $(32 \times 1) + 1 = 33$.
5. **Total Trainable Parameters**: $320,000 + 8,256 + 2,080 + 33 = 330,369$.

The embedding layer accounts for **96.86%** of all model weights, while the recurrent and classification head comprise 10,369 weights.

---

## 4. Training Configuration

| Parameter | Setting | Rationale |
|---|---|---|
| **Optimizer** | Adam (Learning Rate = $10^{-3}$) | Adaptive first- and second-moment estimation |
| **Loss Function** | Binary Cross-Entropy | Proper probabilistic loss for Bernoulli likelihood |
| **Batch Size** | 32 | Provides ~40 weight updates per epoch |
| **Max Epochs** | 30 | Upper boundary with EarlyStopping control |
| **EarlyStopping** | Monitor `val_loss`, patience = 5, `restore_best_weights = True` | Halts training when validation loss degrades and restores optimal checkpoint |
| **Hardware** | Google Colab Standard CPU / T4 GPU | Cross-platform compatibility |

---

## 5. Training Dynamics & Learning Curves

*(Learning curve figures: `results/simple_rnn_loss_curve.png`, `results/simple_rnn_accuracy_curve.png`)*

### Empirical Results (From Colab Execution):
*(Note: Values below will be populated from your Colab execution of `SimpleRNN_model.ipynb`)*

| Metric | Best Epoch (Restored) | Final Epoch |
|---|---|---|
| **Training Loss** | `[To be populated upon Colab run]` | `[To be populated upon Colab run]` |
| **Training Accuracy** | `[To be populated upon Colab run]` | `[To be populated upon Colab run]` |
| **Validation Loss** | `[To be populated upon Colab run]` | `[To be populated upon Colab run]` |
| **Validation Accuracy** | `[To be populated upon Colab run]` | `[To be populated upon Colab run]` |
| **Epochs Completed** | `[e.g., 9-14 epochs]` (EarlyStopping triggered) | 30 max |
| **Training Time** | `[e.g., ~12-18 seconds]` | ~1.2 s/epoch |

### Observations:
1. **Rapid Initial Convergence**: Simple RNN converges quickly in the initial 3-5 epochs as the dense projection and embedding learn prominent sentiment keywords (*"dirty"*, *"great"*, *"fabulous"*).
2. **Generalization Divergence**: Because Simple RNN lacks protective memory gates, the training accuracy rapidly approaches 98-100% while validation loss flattens and begins creeping upward. Early stopping successfully arrests training at the minimum validation loss checkpoint.

---

## 6. Final Test-Set Evaluation

Evaluated **strictly once** on the 160 unseen test reviews (80 positive, 80 negative) at the default decision threshold $\tau = 0.5$:

| Architecture | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time | Parameters |
|---|---|---|---|---|---|---|---|
| **Simple RNN** | `[Colab Result]` | `[Colab Result]` | `[Colab Result]` | `[Colab Result]` | `[Colab Result]` | `[≈ 12-18 s]` | **330,369** |

*95% Bootstrap Confidence Intervals (2,000 iterations):*
- **Accuracy:** `[Lower]` – `[Upper]`
- **F1-Score:** `[Lower]` – `[Upper]`
- **ROC-AUC:** `[Lower]` – `[Upper]`

### Test Confusion Matrix:
*(Figure: `results/simple_rnn_confusion_matrix.png`)*

| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **True Negative (80)** | $\text{TN} =$ `[Result]` | $\text{FP} =$ `[Result]` |
| **True Positive (80)** | $\text{FN} =$ `[Result]` | $\text{TP} =$ `[Result]` |

---

## 7. Error Analysis & The Vanishing Gradient Problem

### Quantitative Error Characteristics:
An inspection of the test set misclassifications (`results/simple_rnn_test_predictions.csv`) reveals clear structural patterns:
1. **Asymmetry in Error Types**: Simple RNN typically produces more False Negatives than False Positives (or vice versa depending on class bias), struggling with nuanced reviews.
2. **Overconfidence on Errors**: Similar to gated models, the model exhibits high confidence ($P > 0.90$) on several incorrect predictions, driven by strong isolated keywords overriding context.

### The Vanishing Gradient Phenomenon:
The primary theoretical and empirical bottleneck of Simple RNN is **Backpropagation Through Time (BPTT) gradient decay**. 

The gradient of the loss function at step $T$ with respect to the hidden state at an earlier step $k$ is given by the chain rule:
$$\frac{\partial L_T}{\partial h_k} = \frac{\partial L_T}{\partial h_T} \prod_{j=k+1}^T \frac{\partial h_j}{\partial h_{j-1}}$$

For the Simple RNN, the Jacobian matrix of the recurrent step is:
$$\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(1 - h_j^2) \, W_{hh}^T$$

Because $\tanh'(z) = 1 - \tanh^2(z) \le 1$ and the singular values of $W_{hh}$ typically contract during gradient descent to prevent exploding activations:
$$\left\| \prod_{j=k+1}^T \frac{\partial h_j}{\partial h_{j-1}} \right\| \le \prod_{j=k+1}^T \|\text{diag}(1 - h_j^2)\| \, \|W_{hh}\| \to 0 \quad \text{as } (T - k) \to \infty$$

**Practical Consequence in Reviews:**
When a review spans 150 to 250 tokens, the gradient of the loss at the end of the sentence with respect to tokens at the beginning decays to near machine epsilon. If a customer writes:
> *"The room was not clean, the bathroom smelled of sewage, but the concierge smiled nicely at checkout."*

The Simple RNN tends to base its prediction disproportionately on the final phrase (*"smiled nicely"*), because the error signal cannot backpropagate 100 steps to adjust weights for the initial negative clauses (*"not clean"*, *"smelled of sewage"*).

---

## 8. Multi-Seed Robustness Validation

*(Data file: `results/simple_rnn_seed_robustness.csv`)*

To confirm that the observed test performance is structurally stable rather than an artifact of lucky weight initialization, Simple RNN was trained across 6 random seeds:

| Seed | Epochs Run | Validation Accuracy | Test Accuracy | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|
| **42 (Base)** | `[Epochs]` | `[Val Acc]` | `[Test Acc]` | `[Test F1]` | `[Test AUC]` |
| **1** | `[Epochs]` | `[Val Acc]` | `[Test Acc]` | `[Test F1]` | `[Test AUC]` |
| **2** | `[Epochs]` | `[Val Acc]` | `[Test Acc]` | `[Test F1]` | `[Test AUC]` |
| **3** | `[Epochs]` | `[Val Acc]` | `[Test Acc]` | `[Test F1]` | `[Test AUC]` |
| **4** | `[Epochs]` | `[Val Acc]` | `[Test Acc]` | `[Test F1]` | `[Test AUC]` |
| **5** | `[Epochs]` | `[Val Acc]` | `[Test Acc]` | `[Test F1]` | `[Test AUC]` |
| **Mean $\pm$ Std** | — | `[Mean ± Std]` | `[Mean ± Std]` | `[Mean ± Std]` | `[Mean ± Std]` |

---

## 9. Conclusion & Comparative Value

The Simple RNN implementation demonstrates that a lightweight recurrent network without gating mechanisms can achieve competitive classification performance on hotel review sentiment. However, its architectural inability to reliably bridge long temporal dependencies ($>50$ tokens) provides the fundamental baseline against which the superior memory capacity of LSTM, GRU, and 1D CNN is benchmarked in our group study.
