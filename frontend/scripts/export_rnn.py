"""Export the trained Simple RNN (Sequence-Averaged R3-3) model weights and manifest into frontend/public/model/rnn/.

Run it from the repository root with:
    uv run --with h5py,numpy,pandas python frontend/scripts/export_rnn.py
"""
import io
import json
from pathlib import Path
import zipfile
import shutil

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "simple_rnn" / "results"
PUBLIC = ROOT / "frontend" / "public"
DIST = ROOT / "frontend" / "dist"
RNN_MODEL_DIR = PUBLIC / "model" / "rnn"
KERAS_PATH = RESULTS / "simple_rnn_model.keras"
VOCAB_PATH = PUBLIC / "model" / "vocab.json"
TEST_PREDS_PATH = RESULTS / "simple_rnn_test_predictions.csv"
DATASET_PATH = ROOT / "dataset" / "deceptive-opinion.csv"


def extract_weights(keras_path):
    """Extract layer weights from the Keras 3 .keras archive."""
    with zipfile.ZipFile(keras_path, "r") as z:
        h5_bytes = z.read("model.weights.h5")

    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        weights = {
            "embedding": np.array(f["layers/embedding/vars/0"], dtype="<f4"),
            "rnn_kernel": np.array(f["layers/simple_rnn/cell/vars/0"], dtype="<f4"),
            "rnn_recurrent_kernel": np.array(f["layers/simple_rnn/cell/vars/1"], dtype="<f4"),
            "rnn_bias": np.array(f["layers/simple_rnn/cell/vars/2"], dtype="<f4"),
            "dense1_kernel": np.array(f["layers/dense/vars/0"], dtype="<f4"),
            "dense1_bias": np.array(f["layers/dense/vars/1"], dtype="<f4"),
            "dense2_kernel": np.array(f["layers/dense_1/vars/0"], dtype="<f4"),
            "dense2_bias": np.array(f["layers/dense_1/vars/1"], dtype="<f4"),
        }
    return weights


def write_weights(model_dir, weights):
    """Write binary weights and JSON manifest."""
    model_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    chunks = []
    offset = 0

    for name, arr in weights.items():
        flat = arr.ravel()
        tensors[name] = {
            "shape": list(arr.shape),
            "offset": offset,
            "length": int(flat.size),
        }
        chunks.append(flat)
        offset += flat.size

    concatenated = np.concatenate(chunks).astype("<f4")
    (model_dir / "weights.bin").write_bytes(concatenated.tobytes())

    manifest = {
        "dtype": "float32",
        "units": int(weights["rnn_kernel"].shape[1]),
        "pooling": "global_average_pooling_1d",
        "tensors": tensors,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Exported {offset} parameters to {model_dir}")

    # Also sync to dist/model/rnn if dist exists
    dist_dir = DIST / "model" / "rnn"
    if dist_dir.parent.exists():
        dist_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_dir / "weights.bin", dist_dir / "weights.bin")
        shutil.copy2(model_dir / "manifest.json", dist_dir / "manifest.json")
        print(f"Synced weights and manifest to {dist_dir}")


def verify_predictions(weights):
    """Self-check: verify forward pass matches test predictions."""
    import sys
    sys.path.insert(0, str(ROOT))
    from shared import data_pipeline as P

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    df = pd.read_csv(DATASET_PATH)
    test_preds = pd.read_csv(TEST_PREDS_PATH)

    emb = weights["embedding"]
    kernel = weights["rnn_kernel"].astype(np.float64)
    recurrent = weights["rnn_recurrent_kernel"].astype(np.float64)
    bias = weights["rnn_bias"].astype(np.float64)
    w1 = weights["dense1_kernel"].astype(np.float64)
    b1 = weights["dense1_bias"].astype(np.float64)
    w2 = weights["dense2_kernel"].astype(np.float64)
    b2 = weights["dense2_bias"].astype(np.float64)

    max_diff = 0.0
    for _, row in test_preds.iterrows():
        row_id = int(row["row_id"])
        text = df.loc[row_id, "text"]
        tokens = P.tokenize(text)
        encoded = P.encode(tokens, vocab)
        padded = P.pad([encoded])[0]

        seq_emb = emb[padded].astype(np.float64)
        h = np.zeros(kernel.shape[1], dtype=np.float64)
        h_seq = []

        # Sequence-averaged forward pass (SimpleRNN with return_sequences=True + GlobalAveragePooling1D)
        for t in range(len(padded)):
            h = np.tanh(seq_emb[t] @ kernel + h @ recurrent + bias)
            h_seq.append(h)

        h_pooled = np.mean(h_seq, axis=0)

        h_dense1 = np.maximum(0, h_pooled @ w1 + b1)
        logits = h_dense1 @ w2 + b2
        p = 1.0 / (1.0 + np.exp(-logits[0]))

        diff = abs(p - row["prob_positive"])
        if diff > max_diff:
            max_diff = diff

    print(f"Verification against all {len(test_preds)} test reviews passed. Max |diff| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"Prediction mismatch: max diff {max_diff}"


def main():
    print(f"Loading weights from {KERAS_PATH}...")
    weights = extract_weights(KERAS_PATH)

    total_params = sum(w.size for w in weights.values())
    print(f"Total extracted parameters: {total_params}")
    assert total_params == 330369, f"Expected 330369 params, got {total_params}"

    write_weights(RNN_MODEL_DIR, weights)

    if VOCAB_PATH.exists() and TEST_PREDS_PATH.exists() and DATASET_PATH.exists():
        verify_predictions(weights)
    else:
        print("Skipping prediction verification (dataset/vocab not found)")


if __name__ == "__main__":
    main()
