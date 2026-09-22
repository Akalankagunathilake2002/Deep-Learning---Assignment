# LSTM Theory & Viva Preparation Guide

Comprehensive theoretical reference and likely viva examination questions for the **Long Short-Term Memory (LSTM)** model in hotel review sentiment classification (SE4050 Deep Learning).

---

## 1. Core Architecture & Mathematics

### Why was LSTM developed?
Traditional Simple RNNs suffer from the **vanishing and exploding gradient problem** when backpropagating through time (BPTT). Because gradients are multiplied by the recurrent weight matrix $W_{hh}$ at every time step ($t=1 \dots T$), eigenvalues smaller than 1 cause gradients to decay exponentially to zero over long sequences ($>10-20$ steps), preventing the network from learning long-term dependencies.

### How does an LSTM solve this?
The LSTM introduces a linear conveyor belt called the **Cell State** ($C_t$). The update equation:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
is **additive**. During backpropagation, the derivative $\frac{\partial C_t}{\partial C_{t-1}} = f_t$. If the forget gate $f_t \approx 1$, the gradient passes backward through time without exponential attenuation (the Constant Error Carousel).

### The Four Gates / Components
1. **Forget Gate ($f_t \in (0, 1)$):**
   $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
   Controls how much of the previous cell state $C_{t-1}$ is preserved.
2. **Input Gate ($i_t \in (0, 1)$):**
   $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
   Decides which values in the candidate state are updated.
3. **Candidate Cell State ($\tilde{C}_t \in (-1, 1)$):**
   $$\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
   Generates newly proposed information to add to the cell state.
4. **Output Gate ($o_t \in (0, 1)$):**
   $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$
   Selects which parts of the cell state $C_t$ are emitted as the hidden state $h_t$.

---

## 2. Parameter Derivations

### Why does an LSTM have $4 \times$ the parameters of a Simple RNN?
For input size $D=64$ and hidden size $H=64$:
* Simple RNN has **1** affine transformation: $(D + H) \times H + H = (64 + 64) \times 64 + 64 = 8,256$.
* GRU has **3** gates (reset, update, candidate): $3 \times 8,256 = 24,960$ (or 24,768 depending on bias implementation).
* LSTM has **4** gates ($f, i, \tilde{C}, o$): $4 \times 8,256 = \mathbf{33,024}$ parameters.
* Bidirectional LSTM has **2 independent layers** (forward and backward): $2 \times 33,024 = \mathbf{66,048}$ parameters.

---

## 3. Likely Viva Questions & Model Answers

### Q1: Why did you compare Unidirectional and Bidirectional LSTM?
**Answer:** In Lab Sheet 05 (Task 3 / Q3), we observed that Unidirectional LSTM matched or slightly outperformed Bidirectional LSTM in sentiment analysis. To determine whether bidirectional context benefited our Chicago hotel review dataset, we evaluated both architectures on the validation set under identical training conditions. We found that the Unidirectional LSTM achieved equal or superior validation performance while requiring 35k fewer parameters, making it more resistant to overfitting on a small dataset (1,276 training reviews).

### Q2: What is the difference between the Hidden State ($h_t$) and the Cell State ($C_t$)?
**Answer:** The Cell State ($C_t$) acts as long-term internal memory with linear additive updates. The Hidden State ($h_t$) acts as short-term working memory and the external output of the cell at step $t$, obtained by non-linearly squashing the cell state with $\tanh$ and filtering it through the output gate $o_t$.

### Q3: Why is pre-padding (`padding="pre"`) used instead of post-padding for recurrent networks?
**Answer:** With post-padding (`[w_1, w_2, ..., 0, 0, 0]`), the LSTM must step through dozens or hundreds of zero tokens after reading the review, which can dilute the accumulated hidden state. With pre-padding (`[0, 0, ..., w_1, w_2]`), the zero tokens are processed first, and the LSTM finishes reading on the actual final words of the review, leaving the final hidden state fresh with the review's content.

### Q4: Why did you keep stop-words and expand contractions?
**Answer:** Standard stop-word removal strips words such as *"not"*, *"no"*, *"never"*, *"hardly"*, and *"didn't"*. In sentiment analysis, negation words are the primary modifiers that invert semantic polarity. We expanded contractions (*"didn't"* → *"did not"*) to guarantee that negation tokens survive punctuation removal.

### Q5: How did you ensure there was no data leakage?
**Answer:** 
1. Exact duplicates were removed before partitioning.
2. The 80/10/10 split was fixed in advance by row ID.
3. The vocabulary and class weights were derived strictly from the **training set**.
4. The test set was touched only once for the final headline evaluation, after all architecture decisions were fixed.
