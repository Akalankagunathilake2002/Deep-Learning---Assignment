# 1D CNN: Theory and Viva Preparation Guide

**Author:** Dilmith  
**Course:** SE4050 – Deep Learning  
**Topic:** 1D Convolutional Neural Networks for Natural Language Processing & Sentiment Analysis  

---

## 1. What is a 1D CNN and how does it work on text?

In computer vision, a 2D CNN slides a 2D filter across spatial height and width. In Natural Language Processing (NLP), a **1D CNN** slides a one-dimensional kernel along the **temporal sequence dimension** (word order) while spanning the entire embedding depth.

- **Input Representation:** A review is transformed into a matrix of shape $(L, D)$, where $L=300$ is sequence length (words) and $D=64$ is the embedding dimension.
- **1D Convolution Operation:** A filter matrix $W \in \mathbb{R}^{k \times D}$ of window size $k$ (e.g., $k=5$) computes the dot product with contiguous word embedding blocks:
  $$c_i = f\left(\sum_{j=1}^{k} \sum_{d=1}^{D} W_{j,d} \cdot X_{i+j-1, d} + b\right)$$
  where $f$ is a non-linear activation function (ReLU), $X$ is the embedding matrix, and $b$ is the bias scalar.
- **Feature Map:** Sliding the filter across all valid windows produces a feature map vector $c \in \mathbb{R}^{L - k + 1}$. With $L=300$ and $k=5$, each filter outputs $300 - 5 + 1 = 296$ temporal activations.
- **Filter Bank:** Using $F=128$ filters generates a feature map tensor of shape $(296, 128)$.

---

## 2. Why use 1D CNN instead of an RNN, LSTM, or GRU?

| Dimension | 1D CNN | Recurrent Neural Networks (RNN, LSTM, GRU) |
| :--- | :--- | :--- |
| **Computation Model** | Parallel feedforward convolution across time steps | Sequential step-by-step state propagation |
| **Hardware Utilization** | Extremely high GPU/TPU/CPU vectorization | Sequential bottleneck prevents full tensor parallelism |
| **Training Speed** | Fast (~6 s in our benchmark) | Slower (~14 s for GRU, longer for LSTM) |
| **Feature Extraction** | Detects local n-gram patterns (e.g., 3-grams, 5-grams) | Models arbitrary-length sequential dependencies |
| **Gradient Flow** | Direct backpropagation through depth (no BPTT) | Susceptible to vanishing/exploding gradients over time |
| **Inductive Bias** | Translation / position invariance via pooling | Order preservation and temporal recency bias |

**Key Viva Takeaway:** *"For document-level sentiment classification, sentiment is often conveyed by localized phrases such as 'horrible customer service' or 'immaculately clean rooms'. A 1D CNN captures these salient n-grams in parallel without needing to unroll recurrent loops over hundreds of time steps."*

---

## 3. What is the role of Global Max-Pooling (GlobalMaxPooling1D)?

After the convolution layer outputs a $(296, 128)$ feature map, `GlobalMaxPooling1D()` extracts the maximum scalar along the temporal axis for each of the 128 filters:
$$\hat{c}_j = \max_{1 \le i \le 296} c_{i, j}, \quad j = 1, \dots, 128$$

### Why is this crucial?
1. **Dimension Reduction:** Collapses the variable or fixed temporal dimension $(296, 128) \to 128$, providing a fixed-size vector for dense classification layers.
2. **Position Invariance (Translation Invariance):** It detects *whether* a critical phrase appeared anywhere in the review, regardless of whether it was in the first sentence or the last sentence.
3. **Sparsity & Noise Filtering:** Ignores filler words and background text, capturing only the most prominent signal per filter.

---

## 4. Why did we choose Kernel Size $k=5$ and 128 Filters?

- **Kernel Size $k=5$:** In English sentiment analysis, critical evaluative phrases frequently span 3 to 5 words (e.g., *"would not recommend this hotel"*, *"best place I ever stayed"*). A 5-gram window captures both the modifier, negation, and adjective together.
- **128 Filters:** Provides sufficient feature capacity to learn 128 distinct phrase archetypes (positive descriptors, negative complaints, amenities, staff evaluations) without causing parameter explosion.
- **Filter Parameters Calculation:**
  $$\text{Params} = (k \times D \times F) + F = (5 \times 64 \times 128) + 128 = 40,960 + 128 = 41,088$$

---

## 5. What are the Limitations of 1D CNNs?

1. **Receptive Field Constraint:** A single layer with kernel size $k=5$ only sees 5 contiguous words at a time. It cannot natively capture dependencies between words separated by 50 tokens (e.g., *"Although in the beginning we thought ... twenty days later we concluded it was awful"*), unless hierarchical/dilated convolutions are employed.
2. **Lack of Sequential Ordering across Filter Peaks:** Global max-pooling discards the relative order between different detected n-grams.
3. **Sensitivity to OOV Tokens:** If key words in an n-gram are out-of-vocabulary, the filter activation may drop significantly.

---

## 6. Likely Viva Questions & Model Answers

### Q1: "Why does the 1D CNN have higher accuracy than the GRU in your results?"
**Answer:** *"In hotel reviews, sentiment is predominantly sparse and localized rather than distributed over complex long-range temporal syntax. The 1D CNN's 128 filters of size 5 acted as efficient parallel matched filters for key evaluative 5-grams. Combined with global max-pooling, the model extracted peak sentiment signals with minimal noise, reaching 94.38% test accuracy compared to 88.13% for the GRU."*

### Q2: "Can 1D CNN use `mask_zero=True` in the Embedding layer?"
**Answer:** *"No. In Keras, 1D convolution layers do not support boolean masking because sliding convolutional kernels across masked and unmasked elements would create undefined boundary conditions. Therefore, to ensure a strictly fair comparison, masking was disabled across all models in our group study."*

### Q3: "How does 1D CNN address the vanishing gradient problem compared to Simple RNN?"
**Answer:** *"A Simple RNN unrolls backpropagation through time over 300 steps, repeatedly multiplying Jacobian matrices by the recurrent weights $W_{hh}$, leading to exponential decay or explosion of gradients. A 1D CNN has only one convolutional layer and one dense hidden layer; the gradient backpropagates directly through the max-pooling switch and embedding depth in just 3 layer transitions, completely eliminating temporal vanishing gradients."*

### Q4: "What happens if a review is shorter than the kernel size?"
**Answer:** *"Our preprocessing pipeline pads all reviews to `MAX_LEN=300` using zero pre-padding. Because $300 \ge 5$, all reviews have sufficient length. Even if an unpadded review was shorter than $k=5$, padding guarantees valid convolution."*
