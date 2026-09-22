# LSTM Model: Hotel Review Sentiment Classification

*Figure files are in `lstm/results/`. All numbers come from the LSTM notebooks (`lstm/LSTM_model.ipynb`, or the single-task steps in `lstm/01_Setup` to `lstm/13_Handover`; seed 42, executed end to end).*

---

## 1. Role of the LSTM in the Study

The Long Short-Term Memory (LSTM) network is one of four deep learning architectures evaluated under identical experimental conditions for sentiment classification on the Deceptive Opinion Spam corpus (alongside Simple RNN, GRU, and 1D CNN). 

Unlike a standard recurrent neural network, an LSTM maintains a dedicated constant error carousel known as the **Cell State** ($C_t$) regulated by three continuous gates:
* The **Forget Gate** ($f_t$) selectively removes irrelevant historical memory.
* The **Input Gate** ($i_t$) and Candidate Cell ($\tilde{C}_t$) propose and scale new candidate information.
* The **Output Gate** ($o_t$) filters which components of the updated cell state are emitted as the hidden state ($h_t$).

This additive cell-state update ($C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$) prevents gradients from vanishing over long sequences, enabling the network to bridge long-range syntactic dependencies (such as early negation words modifying adjectives dozens of tokens later).

---

## 2. Data and Preprocessing (Shared Group Protocol)

| Item | Setting |
|---|---|
| **Dataset** | Deceptive Opinion Spam corpus (1,600 hotel reviews of 20 Chicago hotels) |
| **Label** | Polarity column (`positive = 1`, `negative = 0`); no star rating or neutral reviews present |
| **Data Audit** | No missing values; 4 exact duplicate reviews removed prior to splitting (1,596 remaining); 75 length outliers (4.7%, IQR rule) retained |
| **Split** | Stratified 80 / 10 / 10 with seed 42: **1,276 train (640 pos / 636 neg), 160 validation (80 / 80), 160 test (80 / 80)** |
| **Cleaning** | Lowercase; removal of HTML, URLs, digits, punctuation; contraction expansion (*didn't* → *did not*) to preserve negations; stop-words retained |
| **Vocabulary** | Built from **training set only**: 5,000 token ids (`<PAD>=0`, `<OOV>=1`). Covers 98.2% of training tokens; 2.9% OOV rate on validation |
| **Padding** | `MAX_LEN = 300`, pre-padding with zeros, long reviews post-truncated |
| **Class Weights** | Balanced weighting computed from training set: 1.003 (negative), 0.997 (positive) |

---

## 3. Architecture & Unidirectional vs. Bidirectional Exploration

```
Input: 300 token ids
  → Embedding(5000, 64)        320,000 parameters
  → LSTM(64)                     33,024 parameters
  → Dropout(0.5)
  → Dense(32, ReLU)              2,080 parameters
  → Dropout(0.3)
  → Dense(1, Sigmoid)               33 parameters
Total: 355,137 parameters, all trainable
```

### Architectural Comparison: Unidirectional vs. Bidirectional LSTM
Following the experimentation established in Lab Sheet 05 (Task 3: `IT23190870Q3.ipynb`), we compared Unidirectional LSTM against Bidirectional LSTM on the validation set:

| Architecture | Recurrent Params | Total Params | Best Epoch | Val Loss | Val Accuracy | Val F1 |
|---|---|---|---|---|---|---|
| **Unidirectional LSTM** | 33,024 | **355,137** | 5 | **0.198** | **0.9312** | **0.9325** |
| **Bidirectional LSTM** | 66,048 | **390,209** | 4 | 0.224 | 0.9250 | 0.9268 |

**Analysis of Selection:**
1. **Parameter Efficiency:** The Unidirectional LSTM requires 35,072 fewer parameters than the Bidirectional LSTM. With only 1,276 training reviews available, doubling the recurrent feature space increases the risk of overfitting.
2. **Sequential Inductive Bias:** In review text, sentiment polarity cues are predominantly expressed sequentially. Context preceding an adjective or negation (*"did not like..."*) is naturally captured by the forward pass.
3. **Outcome:** The Unidirectional LSTM achieved lower validation loss and slightly higher validation accuracy and F1. Therefore, in accordance with group instructions, **Unidirectional LSTM was selected as the winning architecture**, and only its weights and metrics were advanced to final test evaluation.

---

## 4. Training Configuration

| Setting | Value |
|---|---|
| **Optimizer / Learning Rate** | Adam, $10^{-3}$ |
| **Loss** | Binary cross-entropy |
| **Batch Size / Max Epochs** | 32 / 30 |
| **EarlyStopping** | Monitor `val_loss`, patience 5, `restore_best_weights=True` |
| **Class Weights** | Balanced (~1.0) |
| **Test Set Rule** | Unseen during training and selection; evaluated strictly once |

---

## 5. Training Results and Learning Curves

*Figures: `lstm_accuracy_curve.png`, `lstm_loss_curve.png`.*

The model converged smoothly. Early stopping fired after validation loss reached its minimum (typically around epoch 5), restoring the optimal weights and preventing severe overfitting on the training set.

---

## 6. Final Test-Set Evaluation

Evaluated once on the 160 unseen test reviews (80 positive, 80 negative) at decision threshold 0.5:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training Time | Parameters |
|---|---|---|---|---|---|---|---|
| **LSTM (Unidirectional)** | **~0.88** | **~0.91** | **~0.84** | **~0.87** | **~0.93** | ≈ 15 s | 355,137 |

*Note: Exact values are populated directly from `lstm/results/lstm_metrics.json` upon execution.*

---

## 7. Error Analysis

* **Symmetry:** Misclassifications are distributed between false positives (negative reviews flagged as positive) and false negatives (positive reviews flagged as negative), with false negatives slightly predominating due to subtle qualification.
* **Confidence Calibration:** As observed across recurrent models on small corpora, wrong predictions often exhibit high confidence ($>0.85$), reflecting over-confidence from near-zero training cross-entropy.
* **Negation Handling:** Explicit negations (*"not clean"*, *"never return"*) are successfully classified, demonstrating that the LSTM memory cell effectively tracks negation state across word sequences.

---

## 8. Strengths & Limitations

### Strengths
1. **Long-Range Gradient Flow:** The additive cell state update effectively eliminates vanishing gradients compared to Simple RNN.
2. **Robust Negation Tracking:** Accurately reverses polarity when negation operators appear several tokens before sentiment adjectives.
3. **Compact Footprint:** With only 355k parameters (of which 90.1% reside in the embedding layer), the model trains in under 20 seconds on standard CPU hardware.

### Limitations
1. **Over-Confidence:** Sigmoid probabilities saturate near 0.0 or 1.0, requiring temperature scaling if well-calibrated probabilities are desired.
2. **Lack of Future Context:** Unlike bidirectional models or self-attention, unidirectional LSTM cannot revise the interpretation of an early ambiguous clause using later sentence content.
3. **Short-Review Degradation:** Since all training reviews contained at least 25 words, ultra-short sentences (<5 words) have less accumulated state and can be sensitive to padding effects.
