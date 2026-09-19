# GRU model: hotel review sentiment classification

*Figure files are in `gru/results/`. All numbers come from the GRU notebooks (`gru/GRU_model.ipynb`, or the same steps split into `gru/01_Setup` to `gru/14_Optional_Extension`; seed 42, executed end to end; re-runs and the split notebooks reproduced identical results).*

## 1. Role of the GRU in the study

The Gated Recurrent Unit (GRU) is one of four architectures compared on the same task, the others being Simple RNN, LSTM and 1D CNN. A GRU reads a review word by word and keeps a memory vector that two learned gates update: the *update gate* decides how much of the old memory to keep, and the *reset gate* decides how much of it to use when proposing new content. This lets it carry cues such as negation across a review while having fewer parameters than an LSTM (24,960 vs 33,024 recurrent parameters for 64 units and 64-dimensional input).

## 2. Data and preprocessing (shared by all four models)

| Item | Setting |
|---|---|
| Dataset | Deceptive Opinion Spam corpus: 1,600 hotel reviews of 20 Chicago hotels |
| Label | The dataset's `polarity` column: positive = 1, negative = 0. The file has no star rating, so the rating rule (≥4 / <3 / 3 removed) could not be applied and no neutral reviews exist |
| Data audit | No missing values or blank texts; no encoding problems (all text is ASCII); 4 exact duplicate reviews (same label each) removed before splitting, leaving 1,596; 75 length outliers (4.7%, IQR rule) kept |
| Split | Stratified 80 / 10 / 10 with seed 42, fixed by explicit validation and test row-id lists in the notebook's Settings cell: **1,276 train (640 pos / 636 neg), 160 validation (80 / 80), 160 test (80 / 80)** |
| Cleaning | Lowercase; remove HTML, URLs, digits and punctuation; contractions expanded so negation survives (*didn't* → *did not*); stop-words kept |
| Vocabulary | Built from the **training set only**: 5,000 ids (`<PAD>` = 0, `<OOV>` = 1). It covers 98.2% of training tokens; 2.9% of validation tokens map to `<OOV>` |
| Padding | `MAX_LEN` = 300, zeros in front (pre-padding), long reviews cut at the end. 6.1% of training reviews are truncated (median length 129 tokens, 95th percentile 318) |
| Class weights | `n / (2 · n_class)` from the training labels: 1.003 (negative) and 0.997 (positive) |

**Exploratory findings (training set).** Negative reviews are longer than positive ones (mean 180 vs 119 tokens), and 95.8% of negative reviews contain a negation word against 59.7% of positive ones, which supports keeping negations in the text. The words most associated with each class are sensible sentiment words (*worst, rude, dirty, broken* vs *enjoyed, perfect, fabulous, loved*). Because length differs between classes, we checked whether length alone could explain the results: a one-threshold length rule, fitted on the training set, reaches only 67.5% validation accuracy (ROC-AUC 0.70; chance is 50%). Length is therefore a weak cue and cannot account for the GRU's performance, although the padding does make review length visible to the model.

## 3. Architecture

```
Input: 300 token ids
  → Embedding(5000, 64)        320,000 parameters
  → GRU(64)                     24,960 parameters
  → Dropout(0.5)
  → Dense(32, ReLU)              2,080 parameters
  → Dropout(0.3)
  → Dense(1, Sigmoid)               33 parameters
Total: 347,073 parameters, all trainable
```

| Layer | Purpose and justification |
|---|---|
| Embedding (dim 64) | Converts each word id into a learned dense vector. A modest dimension was chosen because only 1,276 reviews are available to learn from. No pretrained vectors were used, keeping the comparison to the four architectures. |
| GRU (64 units) | Reads the sequence and returns the final memory state (`tanh` candidate activation, `sigmoid` gates). 64 units is large enough to track sentiment over a review of about 130 words but small enough not to memorise a small training set. |
| Dropout 0.5 | Randomly zeroes half of the GRU outputs during training. Strongest regularisation, placed where memorisation is most likely. |
| Dense (32, ReLU) | One small hidden layer combining the GRU features non-linearly. ReLU is cheap and does not saturate for positive inputs. |
| Dropout 0.3 | Lighter dropout for the small Dense layer. |
| Dense (1, Sigmoid) | Outputs a probability of a positive review; paired with binary cross-entropy. |

The embedding holds 92.2% of all parameters; the recurrent layer and classifier together have 27,073.

## 4. Training configuration

| Setting | Value |
|---|---|
| Optimizer / learning rate | Adam, 1e-3 |
| Loss | Binary cross-entropy |
| Batch size / maximum epochs | 32 / 30 |
| EarlyStopping | Monitor validation loss, patience 5, `restore_best_weights=True` |
| Class weights | As in Section 2 (effectively 1.0, since the classes are balanced) |
| Data used | Training set for weight updates; validation set for monitoring and early stopping; **test set not used** |
| Hardware / software | Apple M4 CPU (no GPU), TensorFlow 2.21, Keras 3.15 |

These values are shared by all four models. No hyperparameter was tuned.

## 5. Training results and learning curves

*Figures: `gru_accuracy_curve.png`, `gru_loss_curve.png`.*

| | Epoch 5 (best, restored) | Epoch 10 (last) |
|---|---|---|
| Training accuracy / loss | 0.984 / 0.063 | 1.000 / 0.003 |
| Validation accuracy / loss | 0.938 / 0.190 | 0.931 / 0.328 |

Training ran for **10 epochs** (early stopping fired five epochs after the validation-loss minimum at epoch 5) in **about 14 s** (13.4 to 14.2 s across identical runs; 1.3 s per epoch).

## 6. Final test-set evaluation

Evaluated once on the 160 unseen test reviews, threshold 0.5; precision, recall and F1 are for the positive class.

| Accuracy | Precision | Recall | F1-score | ROC-AUC | Training time | Trainable parameters |
|---|---|---|---|---|---|---|
| **0.8812** | 0.9178 | 0.8375 | 0.8758 | 0.9367 | ≈ 14 s | 347,073 |

95% bootstrap intervals on the test set: accuracy 0.831 to 0.925, F1 0.813 to 0.925, ROC-AUC 0.893 to 0.973.

*Figures: `gru_confusion_matrix.png`, `gru_roc_curve.png`, `gru_probability_hist.png`.*

| | Predicted negative | Predicted positive |
|---|---|---|
| **True negative (80)** | TN = 74 | FP = 6 |
| **True positive (80)** | FN = 13 | TP = 67 |

Per class: negative precision 0.851, recall 0.925, F1 0.886; positive precision 0.918, recall 0.838, F1 0.876.

## 7. Error analysis

**Overall.** 19 of 160 reviews were misclassified: 6 false positives and 13 false negatives. The GRU is therefore more likely to miss a positive review (recall 0.838) than to wrongly accept a negative one (specificity 0.925). With 19 errors, however, this imbalance is not statistically reliable (a sign test of 13 vs 6 gives p = 0.17).

**Effect of class imbalance.** None is present in this dataset: the test set is exactly 80 / 80 and the class weights are about 1.0, so the asymmetry above cannot be attributed to imbalance. On an imbalanced dataset the weights would matter, because they make errors on the rarer class cost more.

**Confidence.** The model is very sure of itself even when wrong: 13 of the 19 errors had a confidence of at least 0.9 and 5 of at least 0.99 (mean confidence 0.985 on correct predictions vs 0.914 on wrong ones). This is consistent with the near-zero training loss and shows the raw probabilities are over-confident.

**Where the errors are.**

| Group (true class, origin) | Test reviews | Errors | Error rate |
|---|---|---|---|
| Negative, deceptive | 38 | 0 | 0% |
| Negative, truthful | 42 | 6 | 14.3% |
| Positive, deceptive | 42 | 7 | 16.7% |
| Positive, truthful | 38 | 6 | 15.8% |

The `deceptive`/`truthful` field is not a model input; it is used here only to describe errors. The only clear pattern is that fake negative reviews were classified perfectly, which is unlikely to be chance if the other groups' 15.6% error rate applies (probability about 0.2%). Error rate was not clearly related to review length (5.9% for up to 100 tokens, 16.2% for 101 to 200, 10.3% above 200), and only 1 of the 19 errors was a review longer than `MAX_LEN`, so truncation is not a main cause.

**Difficult examples and possible reasons.** The reasons below are interpretations of the misclassified reviews, not tested causes.

1. *Mixed sentiment.* On our reading, most misclassified truthful reviews contain both praise and complaint. A false negative (Omni) is a positive review that complains about the bathroom having only one wastebasket before concluding "what a great hotel!!"; another (James) opens with "I LOVED this hotel" and later calls the restaurant service "horrible" and says "DON'T go there". A false positive (Hard Rock) lists a wrong room, noise and a missing shower door, then ends with enthusiastic praise of the city. Because the GRU summarises the whole review in its final state, later content may weigh heavily, but we did not test this.
2. *Enthusiastic reviews predicted negative.* Seven false negatives are deceptive positive reviews with strongly positive, complaint-free wording (for example "What more could a person ask for?", predicted P(positive) = 0.14). Why the model rejects them is unclear; one possibility is that their marketing-like style differs from the truthful positive reviews it learned from.
3. *Questionable labels.* Two false positives (both Monaco) read as clearly positive ("Excellent Hotel! Rooms and service were great") although labelled negative. Three of the six false positives also contain a pasted "Area recommendations" passage in positive language, a pattern that appears in only one training review.
4. *Borderline cases.* One clearly negative review (Ambassador: "carpeting was very old and worn", "dirty") scored 0.57, just above the threshold.

## 8. Model analysis

| Aspect | Evidence and interpretation |
|---|---|
| **Overfitting** | Yes. Training accuracy reaches 1.000 and training loss 0.003 while validation loss (0.328) stays above its minimum (0.190 at epoch 5). EarlyStopping restored the epoch-5 weights, at which the train-validation accuracy gap is 4.7 points. |
| **Underfitting** | No. Training accuracy exceeds 0.98 by epoch 5. |
| **Convergence** | Fast: validation loss fell from 0.68 to 0.19 in five epochs (about 200 weight updates). |
| **Training stability** | Not smooth. The training loss rose from 0.027 (epoch 6) to 0.154 (epoch 7) and validation loss jumped from 0.190 to 0.437 at epoch 6. A learning rate of 1e-3 may be somewhat high for such a small dataset; we did not tune it, to keep conditions equal across the four models. |
| **Generalisation** | Test accuracy is 0.881 against 0.938 validation accuracy at the selected epoch. Over six random seeds, validation accuracy exceeded test accuracy every time (mean 0.920 vs 0.881), which is consistent with the validation set having been used to pick the best epoch and so giving an optimistic estimate. The test figure is the honest one. |
| **Run-to-run variation** | Six seeds, same split and settings: test accuracy 0.881 ± 0.016, F1 0.877 ± 0.019, ROC-AUC 0.932 ± 0.007 (mean ± std). The headline seed-42 run was fixed in advance and was not selected from these. |
| **Computational efficiency** | About 14 s for the whole training on a CPU; the recurrent layer is small (24,960 parameters). Steps of a recurrent network run sequentially, so a 1D CNN can be expected to be faster per epoch, but that comparison must be made on the same hardware. |
| **Complexity** | 347,073 parameters, of which the embedding accounts for 92.2%. Comparing raw parameter counts between models is therefore dominated by the shared embedding. |

## 9. Strengths and limitations

**Strengths.** Reads text in order, so negation and contrast can be captured; trains in seconds; small recurrent layer; strong ROC-AUC (0.937) meaning positive reviews are usually ranked above negative ones; all results are reproducible from a fixed seed and a fixed, committed split.

**Limitations.** Overfits quickly on 1,276 training reviews and needs early stopping and heavy dropout; over-confident probabilities; the whole review is compressed into one 64-number vector, and long reviews are cut at 300 tokens; embeddings are learned from scratch on little data; very short texts are unreliable, because every training review has at least 25 words and positive reviews are shorter on average, so the model appears to have learned that short means positive (in an informal check with the notebook's demo function, five clearly negative one-line reviews such as "Terrible hotel, dirty room and rude staff." were all labelled positive, with P(positive) between 0.98 and 1.00; multi-sentence reviews of similar sentiment were classified correctly; Section 12 describes an optional mitigation); the test set is small, so results carry roughly ±4.5 points of uncertainty; the dataset is a single-city, curated corpus containing crowd-written fake reviews, so results may not transfer to other hotel-review sources.

## 10. Row for the group comparison table

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training Time | Parameters |
|---|---|---|---|---|---|---|---|
| GRU | 0.8812 | 0.9178 | 0.8375 | 0.8758 | 0.9367 | ≈ 14 s (Apple M4 CPU) | 347,073 |

The row uses the shared dataset, split, preprocessing, vocabulary, padding, class-weight rule, metrics and training settings. No conclusion about the best model can be drawn until all four models are evaluated. When they are, note that a gap of a few points between models is within the test set's uncertainty (Section 6), and that training times are only comparable if measured on the same hardware. Per-review test predictions are saved in `gru_test_predictions.csv` so models can be compared review by review (for example with McNemar's test).

## 11. Critical analysis

The GRU reaches 88% test accuracy on a balanced two-class task, well above the 50% chance level and above the 67.5% obtained from review length alone, so it is learning from the content. Its main weaknesses are rapid overfitting, over-confidence, and difficulty with mixed-sentiment reviews and with enthusiastic reviews written in an unfamiliar style. The reliability of every number here is limited by the size of the test set (160 reviews) and by a small drop from validation to test performance. The results are best read as a fair, reproducible baseline for comparison with the other three models, not as evidence that a GRU is generally better or worse than them.

## 12. Optional extension: robustness to very short reviews (GRU only)

*This extension is not part of the four-model comparison; the group table uses the main GRU of Sections 3 to 6.*

**Problem.** The main GRU labels one-line negative reviews as positive (Section 9). Every training review has at least 25 words and positive reviews are shorter on average, so the model has learned that short text means positive.

**Method.** The same GRU (same architecture, settings and seed) was trained on the 1,276 training reviews plus 2,528 short examples: two random chunks of one or two consecutive sentences cut from each *training* review, labelled with the label of the review they came from (median 22 words, 58% under 25 words). Validation and test data were not changed and no chunk was taken from them. Because a chunk inherits the whole review's label, some chunk labels are noisy.

| | GRU (main) | GRU + short chunks |
|---|---|---|
| Validation accuracy (full reviews) | 0.9375 | 0.9437 |
| Test accuracy / F1 / ROC-AUC | 0.8812 / 0.8758 / 0.9367 | 0.8875 / 0.8889 / 0.9436 |
| Twelve one-line probes correct (negative ones, of 6) | 6 (0) | 11 (5) |
| Twenty new one-liners correct (negative ones, of 10) | 10 (0) | 19 (10) |
| Epochs / training time | 10 / about 14 s | 13 / about 54 s |

The twelve probes were used while comparing variants; the twenty new one-liners were written after the variant was chosen and were not used to pick anything.

**Findings.** The main GRU classified every one of the 16 negative one-liners (6 + 10) as positive, whereas the extension classified 15 of 16 correctly and missed one, "very bad place to visit never ever again visit to this place". The extension also misclassified one positive one-liner ("Spotless room, comfy beds, friendly reception.", P(positive) = 0.37). Test scores on full reviews are practically unchanged: the 0.006 accuracy difference is a single review out of 160, well within the test set's uncertainty (Section 6), so the extension should be described as *not harming* full-review performance, not as improving it. Training took about four times longer because the training set is about three times larger.

**Limits.** One seed and small probe sets; chunk labels are noisy; and the extension still depends on individual words. For example, the miss above contains "visit", which occurs in about twice as many positive as negative training reviews. Using this approach for the group comparison would require all four models to be retrained on the same augmented training set.
