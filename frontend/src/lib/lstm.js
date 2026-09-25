const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class LstmModel {
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
      kernel: get("lstm_kernel"), // [embedDim, 4 * units]
      recurrent: get("lstm_recurrent_kernel"), // [units, 4 * units]
      bias: get("lstm_bias"), // [4 * units]
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
    const G = 4 * U; // 4 gates: i, f, c, o
    const { embedding, kernel, recurrent, bias, w1, b1, w2, b2 } = this.w;

    const h = new Float64Array(U);
    const c = new Float64Array(U);
    const gates = new Float64Array(G);

    for (let t = 0; t < ids.length; t++) {
      const id = ids[t];
      const base = id * E;

      // Initialize with bias
      for (let j = 0; j < G; j++) gates[j] = bias[j];

      // Add x @ kernel
      for (let i = 0; i < E; i++) {
        const x = embedding[base + i];
        if (x === 0) continue;
        const row = i * G;
        for (let j = 0; j < G; j++) gates[j] += x * kernel[row + j];
      }

      // Add h @ recurrent_kernel
      for (let i = 0; i < U; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const row = i * G;
        for (let j = 0; j < G; j++) gates[j] += hi * recurrent[row + j];
      }

      // Gate activations:
      // i: [0, U), f: [U, 2U), c_cand: [2U, 3U), o: [3U, 4U)
      for (let j = 0; j < U; j++) {
        const in_gate = sigmoid(gates[j]);
        const forget_gate = sigmoid(gates[U + j]);
        const cand = Math.tanh(gates[2 * U + j]);
        const out_gate = sigmoid(gates[3 * U + j]);

        c[j] = forget_gate * c[j] + in_gate * cand;
        h[j] = out_gate * Math.tanh(c[j]);
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
