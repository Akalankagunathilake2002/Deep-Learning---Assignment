// The trained Simple RNN sentiment model, written out in plain JavaScript (no TensorFlow needed).
//
//   Embedding(5000, 64) -> SimpleRNN(64, return_sequences=True, tanh) -> GlobalAveragePooling1D()
//                       -> Dropout -> Dense(32, relu) -> Dropout -> Dense(1, sigmoid)
//
// Dropout only acts during training, so it does nothing here.
// Sequence-averaged Elman recurrence (R3-3):
//   h_t = tanh(x_t * W_kernel + h_{t-1} * W_recurrent + bias)
//   h_pooled = (1 / T) * sum_{t=0}^{T-1} h_t
//
// Matches simple_rnn/results/simple_rnn_model.keras evaluated in the SE4050 benchmark.

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class RnnModel {
  /**
   * @param {object} manifest the manifest.json of the exported Simple RNN model
   * @param {ArrayBuffer} buffer its weights.bin as an ArrayBuffer
   */
  constructor(manifest, buffer) {
    const all = new Float32Array(buffer);
    const get = (name) => {
      const t = manifest.tensors[name];
      return all.subarray(t.offset, t.offset + t.length);
    };
    this.units = manifest.units;
    this.embedDim = manifest.tensors.embedding.shape[1];
    this.hidden = manifest.tensors.dense1_bias.shape[0];
    this.pooling = manifest.pooling || "global_average_pooling_1d";
    this.w = {
      embedding: get("embedding"),
      kernel: get("rnn_kernel"), // [embedDim, units]
      recurrent: get("rnn_recurrent_kernel"), // [units, units]
      bias: get("rnn_bias"), // [units]
      w1: get("dense1_kernel"), // [units, hidden]
      b1: get("dense1_bias"), // [hidden]
      w2: get("dense2_kernel"), // [hidden, 1]
      b2: get("dense2_bias"), // [1]
    };
  }

  /**
   * @param {ArrayLike<number>} ids padded word ids (length = maxLen). Returns P(positive) between 0 and 1.
   * @returns {number} Predicted probability P(positive)
   */
  predict(ids) {
    const U = this.units;
    const E = this.embedDim;
    const T = ids.length;
    const { embedding, kernel, recurrent, bias, w1, b1, w2, b2 } = this.w;

    const h = new Float64Array(U);
    const acc = new Float64Array(U);
    const poolSum = new Float64Array(U);

    for (let t = 0; t < T; t++) {
      const id = ids[t];
      const base = id * E;

      // Initialize with bias
      for (let j = 0; j < U; j++) {
        acc[j] = bias[j];
      }

      // Add x @ kernel
      for (let i = 0; i < E; i++) {
        const x = embedding[base + i];
        if (x === 0) continue;
        const row = i * U;
        for (let j = 0; j < U; j++) {
          acc[j] += x * kernel[row + j];
        }
      }

      // Add h @ recurrent_kernel
      for (let i = 0; i < U; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const row = i * U;
        for (let j = 0; j < U; j++) {
          acc[j] += hi * recurrent[row + j];
        }
      }

      // Activation: tanh and accumulate across timesteps for GlobalAveragePooling1D
      for (let j = 0; j < U; j++) {
        const val = Math.tanh(acc[j]);
        h[j] = val;
        poolSum[j] += val;
      }
    }

    // GlobalAveragePooling1D: arithmetic mean across all T timesteps
    const h_pooled = new Float64Array(U);
    const invT = T > 0 ? 1.0 / T : 1.0;
    for (let j = 0; j < U; j++) {
      h_pooled[j] = poolSum[j] * invT;
    }

    // Dense(32, relu) then Dense(1, sigmoid)
    const H = this.hidden;
    let out = b2[0];
    for (let k = 0; k < H; k++) {
      let a = b1[k];
      for (let i = 0; i < U; i++) {
        a += h_pooled[i] * w1[i * H + k];
      }
      if (a > 0) {
        out += a * w2[k];
      }
    }
    return sigmoid(out);
  }
}
