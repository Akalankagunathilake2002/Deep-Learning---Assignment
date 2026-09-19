// The trained sentiment model, written out in plain JavaScript (no TensorFlow needed).
//
//   Embedding(5000, 64) -> GRU(64) -> Dropout -> Dense(32, relu) -> Dropout -> Dense(1, sigmoid)
//
// Dropout only acts during training, so it does nothing here. The GRU is the Keras 3 version
// (reset_after = true): at every word it computes two gates and a candidate memory,
//   z = sigmoid(Wz x + Uz h + bz)          update gate: how much of the old memory to keep
//   r = sigmoid(Wr x + Ur h + br)          reset gate:  how much of the old memory the candidate may use
//   c = tanh(Wh x + r * (Uh h + bh))       candidate memory
//   h = z * h + (1 - z) * c                new memory
// This is a direct port of numpy_forward() in scripts/export_model.py, which was checked against Keras.

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class GruModel {
  /** @param manifest the manifest.json of an exported model, @param buffer its weights.bin as an ArrayBuffer */
  constructor(manifest, buffer) {
    const all = new Float32Array(buffer);
    const get = (name) => {
      const t = manifest.tensors[name];
      return all.subarray(t.offset, t.offset + t.length);
    };
    this.units = manifest.units;
    this.embedDim = manifest.tensors.embedding.shape[1];
    this.hidden = manifest.tensors.dense1_bias.shape[0];
    this.w = {
      embedding: get("embedding"),
      kernel: get("gru_kernel"), // [embedDim, 3 * units]
      recurrent: get("gru_recurrent_kernel"), // [units, 3 * units]
      bias: get("gru_bias"), // [2, 3 * units]: input bias, then recurrent bias
      w1: get("dense1_kernel"), // [units, hidden]
      b1: get("dense1_bias"),
      w2: get("dense2_kernel"), // [hidden, 1]
      b2: get("dense2_bias"),
    };
  }

  /** @param {ArrayLike<number>} ids padded word ids (length = maxLen). Returns P(positive) between 0 and 1. */
  predict(ids) {
    const U = this.units;
    const E = this.embedDim;
    const G = 3 * U;
    const { embedding, kernel, recurrent, bias, w1, b1, w2, b2 } = this.w;
    const inputBias = bias.subarray(0, G);
    const recurrentBias = bias.subarray(G, 2 * G);

    const h = new Float64Array(U);
    const fromInput = new Float64Array(G);
    const fromState = new Float64Array(G);

    for (let t = 0; t < ids.length; t++) {
      const base = ids[t] * E;
      for (let j = 0; j < G; j++) fromInput[j] = inputBias[j];
      for (let i = 0; i < E; i++) {
        const x = embedding[base + i];
        if (x === 0) continue;
        const row = i * G;
        for (let j = 0; j < G; j++) fromInput[j] += x * kernel[row + j];
      }
      for (let j = 0; j < G; j++) fromState[j] = recurrentBias[j];
      for (let i = 0; i < U; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const row = i * G;
        for (let j = 0; j < G; j++) fromState[j] += hi * recurrent[row + j];
      }
      for (let j = 0; j < U; j++) {
        const z = sigmoid(fromInput[j] + fromState[j]);
        const r = sigmoid(fromInput[U + j] + fromState[U + j]);
        const candidate = Math.tanh(fromInput[2 * U + j] + r * fromState[2 * U + j]);
        h[j] = z * h[j] + (1 - z) * candidate;
      }
    }

    // Dense(32, relu) then Dense(1, sigmoid)
    const H = this.hidden;
    let out = b2[0];
    for (let k = 0; k < H; k++) {
      let a = b1[k];
      for (let i = 0; i < U; i++) a += h[i] * w1[i * H + k];
      if (a > 0) out += a * w2[k];
    }
    return sigmoid(out);
  }
}
