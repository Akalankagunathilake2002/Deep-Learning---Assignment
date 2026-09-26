# Final controlled experiment — results

All four models were trained with the identical configuration, on the identical
data pipeline, and evaluated on the identical 160-review test set.

| Model      |   Best Epoch |   Total Epochs |   Best Val Loss |   Test Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |   Params |   Training Time (s) |
|:-----------|-------------:|---------------:|----------------:|----------------:|------------:|---------:|-------:|----------:|---------:|--------------------:|
| Simple RNN |            8 |             13 |        0.17322  |          0.925  |      0.9359 |   0.9125 | 0.9241 |    0.9697 |   330369 |               10.27 |
| LSTM       |            5 |             10 |        0.181187 |          0.8875 |      0.9429 |   0.825  | 0.88   |    0.9372 |   355137 |               11.59 |
| GRU        |            3 |              8 |        0.182083 |          0.9    |      0.9103 |   0.8875 | 0.8987 |    0.9372 |   347073 |               11.79 |
| 1D CNN     |            8 |             13 |        0.146252 |          0.9062 |      0.8736 |   0.95   | 0.9102 |    0.9623 |   365249 |                4.54 |

## Confusion matrices

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Simple RNN | 75 | 5 | 7 | 73 |
| LSTM | 76 | 4 | 14 | 66 |
| GRU | 73 | 7 | 9 | 71 |
| 1D CNN | 69 | 11 | 4 | 76 |

## Architectures

| Model | Layer stack |
|---|---|
| Simple RNN | `Embedding(5000, 64) -> SimpleRNN(64, return_sequences=True) -> GlobalAveragePooling1D -> Dense(32, relu) -> Dropout(0.3) -> Dense(1, sigmoid)` |
| LSTM | `Embedding(5000, 64) -> LSTM(64) -> Dense(32, relu) -> Dropout(0.3) -> Dense(1, sigmoid)` |
| GRU | `Embedding(5000, 64) -> GRU(64) -> Dense(32, relu) -> Dropout(0.3) -> Dense(1, sigmoid)` |
| 1D CNN | `Embedding(5000, 64) -> Conv1D(128, kernel_size=5, relu) -> GlobalMaxPooling1D -> Dense(32, relu) -> Dropout(0.3) -> Dense(1, sigmoid)` |

## Are the differences significant?

All four models predicted on the identical 160 test reviews, so the comparison
is paired and McNemar's exact test applies.

| Model A    | Model B   |   A right, B wrong |   A wrong, B right |   p-value | Significant (0.05)   |
|:-----------|:----------|-------------------:|-------------------:|----------:|:---------------------|
| Simple RNN | LSTM      |                  8 |                  2 |    0.1094 | no                   |
| Simple RNN | GRU       |                 10 |                  6 |    0.4545 | no                   |
| Simple RNN | 1D CNN    |                  8 |                  5 |    0.5811 | no                   |
| LSTM       | GRU       |                  5 |                  7 |    0.7744 | no                   |
| LSTM       | 1D CNN    |                  7 |                 10 |    0.6291 | no                   |
| GRU        | 1D CNN    |                  5 |                  6 |    1      | no                   |

0 of 6 pairs differ significantly at the 0.05 level. On this test set the four architectures are statistically indistinguishable.

## Environment

- Apple M4, 10 cores, 16 GB RAM
- macOS-26.2-arm64-arm-64bit
- Python 3.12.13, TensorFlow 2.21.0, Keras 3.15.1
- Accelerator: none (CPU only)

Training time is reported as an observation only. It is **not** used to rank the
models: wall-clock time depends on hardware, and it also depends on how many
epochs early stopping allowed, which differs legitimately between architectures.
