// The trained 1D CNN sentiment model, written in plain JavaScript (no external ML library needed).
//
//   Input(300) -> Embedding(5000, 64) -> Conv1D(128, k=5, relu) -> GlobalMaxPooling1D()
//              -> Dropout -> Dense(32, relu) -> Dropout -> Dense(1, sigmoid)
//
// Matches the Keras 3 model evaluated in cnn1d/results/cnn1d_model.keras.
// Developed by Dilmith for SE4050 Deep Learning.

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class Cnn1dModel {
  /**
   * @param manifest the manifest.json of the exported 1D CNN model
   * @param buffer its weights.bin as an ArrayBuffer
   */
  constructor(manifest, buffer) {
    const all = new Float32Array(buffer);
    const get = (name) => {
      const t = manifest.tensors[name];
      return all.subarray(t.offset, t.offset + t.length);
    };
    this.filters = manifest.filters ?? 128;
    this.kernelSize = manifest.kernel_size ?? 5;
    this.embedDim = manifest.embed_dim ?? 64;
    this.hidden = manifest.hidden ?? 32;
    this.w = {
      embedding: get("embedding"), // [5000, 64]
      convKernel: get("conv1d_kernel"), // [5, 64, 128]
      convBias: get("conv1d_bias"), // [128]
      w1: get("dense1_kernel"), // [128, 32]
      b1: get("dense1_bias"), // [32]
      w2: get("dense2_kernel"), // [32, 1]
      b2: get("dense2_bias"), // [1]
    };
  }

  /**
   * Run forward inference on pre-padded token IDs.
   * @param {ArrayLike<number>} ids padded word IDs (length = 300)
   * @returns {number} Predicted probability P(Positive) in [0, 1]
   */
  predict(ids) {
    const L = ids.length;
    const K = this.kernelSize;
    const E = this.embedDim;
    const F = this.filters;
    const H = this.hidden;
    const { embedding, convKernel, convBias, w1, b1, w2, b2 } = this.w;

    const Lout = L - K + 1; // 296
    const pooled = new Float32Array(F);

    // Global max pooling over temporal 1D convolution
    for (let t = 0; t < Lout; t++) {
      for (let f = 0; f < F; f++) {
        let sum = convBias[f];
        for (let k = 0; k < K; k++) {
          const wordId = ids[t + k];
          if (wordId === 0) continue; // padding token
          const embBase = wordId * E;
          const kBase = (k * E) * F + f;
          for (let d = 0; d < E; d++) {
            const x = embedding[embBase + d];
            if (x !== 0) {
              sum += x * convKernel[kBase + d * F];
            }
          }
        }
        // ReLU & running max
        if (sum > pooled[f]) {
          pooled[f] = sum;
        }
      }
    }

    // Dense(32, ReLU)
    const h = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      let sum = b1[j];
      for (let f = 0; f < F; f++) {
        const pf = pooled[f];
        if (pf > 0) {
          sum += pf * w1[f * H + j];
        }
      }
      h[j] = sum > 0 ? sum : 0;
    }

    // Dense(1, Sigmoid)
    let out = b2[0];
    for (let j = 0; j < H; j++) {
      const hj = h[j];
      if (hj > 0) {
        out += hj * w2[j];
      }
    }

    return sigmoid(out);
  }
}
