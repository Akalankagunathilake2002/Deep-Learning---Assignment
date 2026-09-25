# Simple RNN: Comprehensive Theory & Viva Voce Guide
**Individual Contribution:** Simple RNN (Vanilla Elman Recurrent Neural Network)
**Course:** SE4050 – Deep Learning 2026
**Subject:** Hotel Review Sentiment Classification

---

## PART 1: CORE MATHEMATICAL THEORY

### 1.1 The Recurrent Equation (Elman Network)
A Simple Recurrent Neural Network (Elman, 1990) processes an input sequence $X = (x_1, x_2, \dots, x_T)$ sequentially. At each time step $t \in \{1, \dots, T\}$:
1. **Affine Combination:**
   $$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$
2. **Hidden State Activation:**
   $$h_t = \tanh(a_t) = \frac{e^{a_t} - e^{-a_t}}{e^{a_t} + e^{-a_t}}$$
3. **Classification Output Head:**
   $$z = W_{hy} h_T + b_y$$
   $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
- $x_t \in \mathbb{R}^{d}$ ($d = 64$ embedding dimension)
- $h_t \in \mathbb{R}^{m}$ ($m = 64$ hidden recurrent units)
- $W_{xh} \in \mathbb{R}^{m \times d}$ (input-to-hidden weight matrix, $64 \times 64 = 4,096$)
- $W_{hh} \in \mathbb{R}^{m \times m}$ (hidden-to-hidden recurrent matrix, $64 \times 64 = 4,096$)
- $b_h \in \mathbb{R}^{m}$ (recurrent bias vector, 64)
- $\hat{y} = P(\text{positive} \mid X) \in [0, 1]$

---

### 1.2 Backpropagation Through Time (BPTT) & The Vanishing Gradient Problem

To train the Simple RNN, we minimize the binary cross-entropy loss:
$$L = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

During BPTT, the gradient with respect to the recurrent weight matrix $W_{hh}$ accumulates the gradient across all time steps:
$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}$$

For any step $t$, the gradient with respect to an earlier hidden state $h_k$ ($k < t$) requires applying the chain rule across the temporal sequence:
$$\frac{\partial L}{\partial h_k} = \frac{\partial L}{\partial h_t} \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}$$

Examining the Jacobian matrix $\frac{\partial h_j}{\partial h_{j-1}}$:
$$\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(1 - h_j^2) \, W_{hh}^T$$

Let $J_j = \text{diag}(1 - h_j^2) \, W_{hh}^T$. The temporal gradient product is:
$$\prod_{j=k+1}^t J_j = \prod_{j=k+1}^t \left[ \text{diag}(1 - h_j^2) \, W_{hh}^T \right]$$

#### Why Gradients Vanish:
1. **Derivative of Tanh**: The derivative $\tanh'(z) = 1 - \tanh^2(z) \in (0, 1]$. In practice, activations frequently saturate towards $\pm 1$, causing $\tanh'(z) \ll 1$.
2. **Eigenvalue Spectrum**: If the largest eigenvalue (spectral radius) of $W_{hh}$ satisfies $\rho(W_{hh}) < 1$, then:
   $$\left\| \prod_{j=k+1}^t W_{hh}^T \right\| \le \|W_{hh}\|^{t-k} \xrightarrow{t - k \to \infty} 0$$
   The norm of the gradient shrinks exponentially with the temporal distance $(t - k)$. Consequently, words encountered early in a 300-word review have virtually zero influence on weight updates.

---

## PART 2: ARCHITECTURAL COMPARISON MATRIX

| Feature / Dimension | Simple RNN | LSTM | GRU | 1D CNN |
|---|---|---|---|---|
| **Recurrent Equation** | $h_t = \tanh(W x_t + U h_{t-1} + b)$ | Gated: Forget $f_t$, Input $i_t$, Output $o_t$, Cell $C_t$ | Gated: Reset $r_t$, Update $z_t$, Candidate $\tilde{h}_t$ | Non-recurrent: $y = \text{Conv1D}(X)$ + GlobalPooling |
| **Memory Highway** | None (pure matrix mult) | Additive linear cell state: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ | Convex interpolation: $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ | Feedforward receptive field over local n-grams |
| **Vanishing Gradient** | Severe over $>30$ steps | Effectively solved via constant error carousel (CEC) | Effectively solved via shortcut update gate connection | Immune across depth (shallow convolutional layers) |
| **Recurrent Parameters (units=64)** | **8,256** | **33,024** ($4 \times$ Simple RNN) | **24,960** ($3 \times$ Simple RNN) | Varies (e.g. Filter size $5 \times 64 \times 64 = 20,480$) |
| **Computational Speed** | Fastest per step ($\mathcal{O}(m^2)$) | Slower ($\mathcal{O}(4m^2)$) | Moderate ($\mathcal{O}(3m^2)$) | Highly parallelizable across sequence length |
| **Inductive Bias** | Sequential temporal recency | Long-term sequential retention | Balanced sequential retention | Spatial/local n-gram feature extraction |

---

## PART 3: TOP 25 VIVA VOCE QUESTIONS & MODEL ANSWERS

### Q1: What is a Simple RNN and what role does it play in your group project?
**Answer:** A Simple RNN (Elman network) is the foundational recurrent architecture where the hidden state is updated at each step via an affine transformation of the input and previous hidden state followed by a `tanh` activation. In our group assignment, Simple RNN serves as the baseline sequential model. Comparing it directly to LSTM, GRU, and 1D CNN allows us to quantify the exact performance gain achieved by adding gating mechanisms and convolutional filters.

### Q2: Exactly how many parameters does your Simple RNN model have? Break them down.
**Answer:** The model has exactly **330,369 parameters**, 100% trainable:
- `Embedding(5000, 64)`: $5,000 \times 64 = 320,000$ params (96.86% of total).
- `SimpleRNN(64)`: $(64 \times 64) + (64 \times 64) + 64 = 8,256$ params ($W_{xh} + W_{hh} + b_h$).
- `Dropout(0.5)`: 0 params.
- `Dense(32, relu)`: $(64 \times 32) + 32 = 2,080$ params.
- `Dropout(0.3)`: 0 params.
- `Dense(1, sigmoid)`: $(32 \times 1) + 1 = 33$ params.
- Total = $320,000 + 8,256 + 2,080 + 33 = 330,369$.

### Q3: Why does Simple RNN suffer from the vanishing gradient problem?
**Answer:** During Backpropagation Through Time (BPTT), the gradient of the loss at step $T$ with respect to step $k$ involves the repeated product of the Jacobian matrix $\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(1 - h_j^2) W_{hh}^T$. Because the derivative of `tanh` is bounded by 1 ($\le 1$) and the spectral radius of $W_{hh}$ is typically $< 1$, multiplying these matrices over many time steps ($T - k > 30$) causes the gradient norm to diminish exponentially towards zero.

### Q4: How do LSTM and GRU resolve this vanishing gradient issue?
**Answer:** LSTM introduces an additive cell state $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. Because the gradient can flow back along the cell state with $\frac{\partial C_t}{\partial C_{t-1}} = f_t$, when the forget gate $f_t \approx 1$, the gradient does not decay exponentially. GRU achieves a similar effect using an update gate $z_t$ which acts as a linear shortcut between $h_{t-1}$ and $h_t$.

### Q5: Why is pre-padding (`padding='pre'`) preferred over post-padding for Simple RNN?
**Answer:** Pre-padding places `<PAD>` zeros at the start of the sequence, so the network reads real review tokens at the final time steps before the classification head. If post-padding were used, a 50-token review would be followed by 250 steps of zeros. In Simple RNN, repeated multiplication by $W_{hh}$ over 250 zero inputs would cause the meaningful hidden state from the review to decay back to zero before reaching the output layer.

### Q6: Why did you keep stop-words like "not", "no", and "never"?
**Answer:** In sentiment analysis, negation words are critical sentiment shifters. Removing "not" turns *"The room was not clean"* into *"room clean"*, completely inverting the polarity. Our EDA revealed that 95.8% of negative reviews contain negation words compared to only 59.7% of positive reviews.

### Q7: Why is the vocabulary built strictly on the training set?
**Answer:** To prevent data leakage. If we included the validation or test set when computing token frequencies, words unique to the test set would be indexed into the vocabulary, artificially inflating test set comprehension. Any token in validation or test not in the training vocabulary is mapped to `<OOV>` (ID 1).

### Q8: What is the purpose of the EarlyStopping callback?
**Answer:** EarlyStopping monitors validation loss (`val_loss`) and halts training after a designated `patience` (5 epochs) without improvement. Critically, `restore_best_weights=True` ensures the model weights are rolled back to the checkpoint with the lowest validation loss, effectively mitigating overfitting.

### Q9: Why use Binary Cross-Entropy with a Sigmoid activation?
**Answer:** For a binary classification task ($y \in \{0, 1\}$), the Sigmoid activation squashes the output into $[0, 1]$, representing the Bernoulli posterior probability $P(y=1 \mid x)$. Binary cross-entropy corresponds to the negative log-likelihood under this distribution, providing steep gradients when the model makes confident incorrect predictions.

### Q10: Why did you compute class weights if the dataset is 50/50 balanced?
**Answer:** The training set has 640 positive and 636 negative reviews (near-perfect balance). While the calculated weights are virtually 1.0 (0.997 and 1.003), computing them is an essential methodological requirement across all four group models to guarantee consistency and guard against bias if applied to skewed external corpora.

### Q11: What does the ROC-AUC score measure, and why is it valuable?
**Answer:** ROC-AUC measures the area under the Receiver Operating Characteristic curve. It reflects the probability that the model ranks a randomly chosen positive review higher than a randomly chosen negative review across all possible classification thresholds. Unlike accuracy, it is threshold-independent.

### Q12: Why did you calculate a 95% bootstrap confidence interval on the test set?
**Answer:** Our test set has 160 reviews. Because of small sample variation, point estimates like 85% accuracy carry uncertainty. Generating 2,000 bootstrap resamples yields empirical 95% confidence intervals (e.g. $[81.2\%, 89.4\%]$), confirming whether differences between models are statistically meaningful.

### Q13: Why did you evaluate the test set only once?
**Answer:** The test set must remain completely unseen during model selection and hyperparameter tuning. If we repeatedly evaluated on the test set to adjust architecture, learning rate, or thresholds, the test set would become an implicit training set, invalidating generalizability claims.

### Q14: How does Simple RNN handle word order compared to Bag-of-Words or 1D CNN?
**Answer:** Bag-of-Words treats text as an unordered frequency vector, losing all syntax. 1D CNN captures local word order within small sliding windows (kernel size 3 or 5). Simple RNN processes the sequence strictly in chronological order, allowing it in theory to track sentence structure from beginning to end.

### Q15: Why is Dropout rate 0.5 placed immediately after SimpleRNN?
**Answer:** The 64 hidden units output from SimpleRNN carry the combined semantic representation of the review. Applying 50% dropout during training randomly zeroes activations, forcing the network to learn redundant, robust feature representations rather than memorizing individual training reviews.

### Q16: What is the computational complexity of Simple RNN per step?
**Answer:** Simple RNN computes $W_{xh} x_t$ (size $m \times d$) and $W_{hh} h_{t-1}$ (size $m \times m$). Its complexity per time step is $\mathcal{O}(m \cdot d + m^2)$. For $m=d=64$, this requires approximately $8,192$ floating point operations per step, which is $4\times$ faster than an equivalent LSTM layer.

### Q17: Could gradient clipping help Simple RNN?
**Answer:** Yes. Gradient clipping ($\|g\| \le \text{clip\_norm}$) effectively prevents the **exploding gradient problem** by scaling back abnormally large gradient vectors. However, gradient clipping cannot resolve the **vanishing gradient problem**, because scaling an already near-zero gradient leaves it negligible.

### Q18: What is an OOV token, and what was your OOV rate on the validation set?
**Answer:** `<OOV>` stands for Out-Of-Vocabulary. Any word in the validation or test set that did not appear in the top 5,000 words of the training set is assigned ID 1. On our validation set, the OOV rate was approximately $2.8\%$, demonstrating that a 5,000-word vocabulary provides $>97\%$ coverage on hotel domain text.

### Q19: Why was review length truncated at `MAX_LEN = 300`?
**Answer:** Our EDA showed that the median review length was 128 tokens, and 94% of all reviews were shorter than 300 tokens. Setting `MAX_LEN = 300` limits padding overhead for shorter reviews while preserving the full content of the vast majority of texts.

### Q20: What did your error analysis reveal about Simple RNN's failure modes?
**Answer:** Simple RNN misclassified reviews characterized by:
1. Long review length ($>200$ tokens), where initial complaints were forgotten due to gradient decay.
2. Mixed sentiment reviews containing contrasting clauses (e.g. *"great staff, but filthy bathroom"*), where the model latched onto local positive sentiment near the end.

### Q21: What is the purpose of the multi-seed robustness check?
**Answer:** Neural network training is stochastic due to random weight initialization and minibatch ordering. Training across 6 distinct seeds (`[42, 1, 2, 3, 4, 5]`) and reporting mean $\pm$ standard deviation demonstrates that the model's performance is stable and reproducible.

### Q22: Why did you not use pretrained GloVe or BERT embeddings?
**Answer:** The primary objective of this SE4050 assignment is to isolate and benchmark the algorithmic capabilities of the four recurrent/convolutional architectures. Using heavy pretrained contextual representations like BERT would mask architectural differences, as the pretrained embeddings would perform most of the feature extraction.

### Q23: Why did you use ReLU for the intermediate dense layer?
**Answer:** Rectified Linear Unit ($\text{ReLU}(z) = \max(0, z)$) is computationally efficient and has a constant derivative of 1 for positive inputs, avoiding further gradient saturation in the classification head.

### Q24: What is the difference between Stateful and Stateless RNNs in Keras?
**Answer:** By default, Keras RNNs are stateless (`stateful=False`), meaning the hidden state is reset to zeros at the beginning of each batch/sample. In stateful RNNs, the hidden state is preserved across batches, which is useful for continuous time-series but unsuitable for independent customer reviews.

### Q25: If you had another week to improve the Simple RNN model, what would you try?
**Answer:** 
1. **Bidirectional Simple RNN**: Process sequences forward and backward simultaneously to mitigate unilateral vanishing gradients.
2. **Layer Normalization**: Apply LayerNorm inside the recurrence to stabilize hidden state dynamics.
3. **Residual / Skip Connections**: Add identity shortcuts between time steps to facilitate gradient propagation across long reviews.
