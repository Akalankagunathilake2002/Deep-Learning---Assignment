# GRU: theory and viva preparation

Plain-language explanations for the report and the viva. Results-specific answers (accuracy, overfitting, and so on) are in `GRU_report_section.md`, because they depend on the actual numbers.

---

## 1. What is a GRU?

A **Gated Recurrent Unit** is a type of recurrent neural network layer (Cho et al., 2014). It reads a sequence one element at a time (here, one word at a time) and keeps a **memory vector** `h` that summarises everything read so far. What makes it different from a Simple RNN is that it has two **gates**, small learned switches that decide how much of the old memory to keep and how much new information to let in.

**One-sentence version for the viva:** *"A GRU is a recurrent layer that reads text word by word and uses two gates, update and reset, to decide what to remember and what to forget, so it can carry important information such as a negation across a long review."*

## 2. Why is it suitable for sequential text?

Text is ordered, and order changes meaning: *"the room was **not** clean"* vs *"the room was clean"* contain the same words except for one. A model that only counts words cannot see that. A GRU reads the words in order, and its memory is updated at every word, so the meaning of a later word can depend on what came before. It also handles reviews of different lengths with the same weights (the same GRU cell is reused at every word).

## 3. How does a GRU process a sequence?

For our model, each review is 300 token ids (pre-padded with zeros).

1. **Embedding**: every id becomes a 64-number vector, giving a 300 × 64 table.
2. The GRU starts with an empty memory `h₀ = 0`.
3. At each step `t` it takes the current word vector `xₜ` and the previous memory `hₜ₋₁`, and computes a new memory `hₜ`.
4. It repeats this for all 300 positions. The padding zeros come first, so the last steps are the real words.
5. Only the **final memory** `h₃₀₀` (64 numbers) is passed on, through Dropout → Dense(32, ReLU) → Dropout → Dense(1, Sigmoid) to give P(positive).

## 4. The gates (Keras convention, which is what our code uses)

At every step:

| | Formula | Plain meaning |
|---|---|---|
| **Update gate** | `zₜ = σ(W_z xₜ + U_z hₜ₋₁ + b_z)` | "How much of the **old memory** should I keep?" |
| **Reset gate** | `rₜ = σ(W_r xₜ + U_r hₜ₋₁ + b_r)` | "How much of the **old memory** should I use when *proposing* new content?" |
| **Candidate memory** | `h̃ₜ = tanh(W_h xₜ + rₜ ⊙ (U_h hₜ₋₁) + b_h)` | "What new memory would I write, given this word (and the past, as allowed by the reset gate)?" |
| **New memory** | `hₜ = zₜ ⊙ hₜ₋₁ + (1 − zₜ) ⊙ h̃ₜ` | A blend of old memory and the candidate |

`σ` is the sigmoid (output between 0 and 1) and `⊙` means element-wise multiplication. Each gate has one value **per memory slot**, so different slots can keep or forget independently.

**Update gate `z`:**
* `z ≈ 1` → copy the old memory unchanged (ignore this word for that slot).
* `z ≈ 0` → replace the memory with the new candidate.
* It acts as a learned "keep vs overwrite" dial.

**Reset gate `r`:**
* `r ≈ 0` → build the candidate as if there were no past, which is useful when a new topic starts (for example "*but* the staff were rude...").
* `r ≈ 1` → build the candidate using the full past.

> Some textbooks write the last line as `(1 − z)·hₜ₋₁ + z·h̃ₜ`. It is the same model with the gate's meaning flipped. In Keras, `z` is the "keep" amount. Quote the Keras version and say the convention varies between sources.

## 5. How does a GRU handle long-term dependencies?

A **Simple RNN** overwrites its whole memory at every word: `hₜ = tanh(W hₜ₋₁ + U xₜ)`. During training the error signal travels backwards through all steps and is multiplied at each one by numbers that are usually smaller than 1, so it **shrinks towards zero** (the *vanishing gradient problem*). The network then cannot learn from words that are far back.

A GRU's update gate lets the memory be **copied almost unchanged** across a step (`z ≈ 1`). The error signal can then flow backwards through that step almost untouched, like a shortcut. The network **learns** when to use the shortcut, so an important early cue (for example "*not*") can survive many words.

This makes long dependencies **easier to learn**, not solved. Very long sequences are still hard, and Transformers are better at that.

## 6. Why is a GRU useful for hotel-review sentiment?

* Reviews are 25 to 800 words long (median ≈ 130), which is long enough for a Simple RNN to struggle.
* Sentiment is often carried by **order and negation** ("not clean", "would not recommend") and by **contrast** ("nice location, *but* dirty room"). A GRU can keep, flip or override earlier evidence as it reads.
* It has fewer parameters than an LSTM, which suits our **small training set** (1,276 reviews).

## 7. Simple RNN vs GRU

| | Simple RNN | GRU |
|---|---|---|
| Memory update | Overwritten completely each step | Gated: keep some, write some |
| Gates | None | 2 (update, reset) |
| Long-range learning | Poor (vanishing gradients) | Better |
| Parameters (units=64, input=64) | 8,256 | 24,960 |
| Cost per step | Lowest | About 3× the Simple RNN |
| Typical use | Short sequences, baselines | Longer text where memory matters |

## 8. GRU vs LSTM

| | GRU | LSTM |
|---|---|---|
| States | One (`h`) | Two (hidden `h` and cell `c`) |
| Gates | 2: update, reset | 3: input, forget, output |
| Weight sets | 3 | 4 |
| Parameters (units=64, input=64) | 24,960 | 33,024 |
| Speed | Usually somewhat faster | Somewhat slower |
| Accuracy | Often about the same | Often about the same |

Published comparisons (for example Chung et al., 2014) found neither consistently better. **Do not claim in the report that one wins in general.** State only what *our* experiment shows, and only after all four models are done.

## 9. Computational characteristics

* Step `t` needs `hₜ₋₁`, so the 300 steps **must run one after another**. This limits parallelism. A 1D CNN can process all positions at once, so it is normally faster per epoch.
* Cost per epoch grows with sequence length (`MAX_LEN`) and with the number of units.
* In our model the recurrent layer is small. Most of the parameters are in the **Embedding** (5000 × 64 = 320,000). Comparing "parameter count" between models is therefore dominated by the shared embedding, so report the non-embedding count as well.
* On GPU, Keras uses a fast cuDNN kernel when the GRU uses default settings. Adding `recurrent_dropout` disables it and makes training much slower, which is one reason we do not use it.

## 10. Advantages and limitations

**Advantages**
* Learns order-dependent patterns such as negation.
* Easier to train than a Simple RNN and better at longer dependencies.
* Fewer parameters than an LSTM, so less overfitting risk on small data.
* Small and simple to train, with no pretrained model needed.

**Limitations**
* Sequential, so it cannot parallelise over time and is slower than a CNN.
* Long-range memory is still limited, and very long reviews are truncated at `MAX_LEN`.
* Compresses the whole review into one 64-number vector, so details can be lost.
* No attention, so it cannot look back at specific words directly.
* Embeddings are learned from scratch on a small dataset, so rare words are poorly represented.
* Hard to interpret, because we cannot easily say why it predicted a class.

---

## 11. Likely viva questions and short answers

**Q1. Why did you choose a GRU?**
It is one of the four models assigned to the group. Technically it suits sentiment because reviews are ordered text where negation and contrast matter, and it is lighter than an LSTM, which fits a small dataset.

**Q2. Explain the update gate and reset gate.**
See section 4. Update = keep old memory vs write new. Reset = whether the new candidate should look at the old memory.

**Q3. Why not a Simple RNN?**
It overwrites its memory each step and suffers vanishing gradients, so it struggles to use words far back. That is why it is a baseline in the comparison.

**Q4. What is the difference between GRU and LSTM?**
GRU has one state and two gates. LSTM has a separate cell state and three gates. GRU has about 25% fewer recurrent parameters (24,960 vs 33,024 here) and is often similar in accuracy.

**Q5. Why an Embedding layer?**
Token ids are arbitrary integers with no meaning. The embedding learns a dense vector for each word so that similar words can have similar vectors.

**Q6. Why is the vocabulary built only from the training set?**
To prevent data leakage. If validation or test words shaped the vocabulary, the model would have had indirect access to them. Words that appear only in validation or test are mapped to `<OOV>`, exactly as it would happen on genuinely new reviews.

**Q7. Why is the test set used only once?**
So the test score is an honest estimate of performance on new data. If we tuned on it, the score would be optimistic. We tuned nothing on it; validation guided everything.

**Q8. Why do you use class weights when the classes are balanced?**
The group agreed one procedure for all models so they stay comparable. On this dataset the weights are about 1.0 and change nothing. The step matters for imbalanced data, where it makes errors on the rare class cost more so the model does not just favour the majority class.

**Q9. Why the label is `polarity` and not a rating rule?**
This dataset has no ratings, only a polarity label (positive or negative), so the rating rule cannot be applied. There are no neutral reviews to remove.

**Q10. Why pre-padding?**
With pre-padding the zeros come first and the real words last, so the GRU's final memory is computed right after real words instead of after many empty steps. All four models use the same padding.

**Q11. Why dropout, and why 0.5 then 0.3?**
Dropout randomly switches off units during training so the network cannot rely on any single one, which reduces overfitting. With 1,276 training reviews overfitting is the main risk, so the stronger 0.5 goes after the GRU and the lighter 0.3 after the small Dense layer.

**Q12. Why ReLU in the hidden layer and sigmoid at the output?**
ReLU is simple, cheap and avoids vanishing gradients. Sigmoid squashes the output to (0, 1), which is read as the probability of a positive review, and it pairs with binary cross-entropy.

**Q13. Why binary cross-entropy?**
It measures how far the predicted probability is from the true 0/1 label and penalises confident wrong predictions heavily. It is the standard loss for binary classification with a sigmoid output.

**Q14. Why EarlyStopping on validation loss, and what does `restore_best_weights` do?**
Training loss keeps falling even when the model starts memorising. Validation loss rises when that happens, so we stop after 5 epochs without improvement. `restore_best_weights` rolls the model back to the best epoch instead of keeping the overfitted last one.

**Q15. Why is accuracy not the only metric?**
Accuracy can hide poor performance on one class. Precision, recall and F1 show the trade-off between false positives and false negatives, and ROC-AUC measures how well the model ranks positive above negative reviews regardless of the 0.5 threshold.

**Q16. Can you say the GRU is the best model?**
Not until all four models have been evaluated under identical conditions. Also, with a 160-review test set, small differences between models are within noise (see the bootstrap interval in the report section).

**Q17. What would you improve with more time?**
Pretrained embeddings (GloVe) or a bidirectional GRU, more data, and repeated cross-validation instead of a single split. We did not use them to keep the comparison to the four required architectures under the same conditions.
