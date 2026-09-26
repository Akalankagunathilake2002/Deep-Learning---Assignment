"""Export the trained GRU models and everything the browser needs into frontend/public and frontend/src/lib/__fixtures__.

Run it from the repository root with a Python that has TensorFlow (see requirements.txt):

    python frontend/scripts/export_model.py

The exported files are committed, so you only need to run this again if the models are retrained.
It uses the shared data pipeline (../shared) only to rebuild the training vocabulary and the reference outputs.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tensorflow import keras  # noqa: E402

from shared import config as C  # noqa: E402
from shared import data_pipeline as P  # noqa: E402

RESULTS = ROOT / "gru" / "results"
PUBLIC = ROOT / "frontend" / "public"
MODEL_DIR = PUBLIC / "model"
FIXTURES = ROOT / "frontend" / "src" / "lib" / "__fixtures__" / "fixtures.json"

MODELS = {"main": RESULTS / "gru_model.keras"}
FIGURES = ["gru_accuracy_curve.png", "gru_loss_curve.png", "gru_confusion_matrix.png", "gru_roc_curve.png",
           "gru_probability_hist.png"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def extract_weights(model):
    """Weights in the order the browser needs them (Keras 3 GRU, reset_after=True, gates z, r, h)."""
    emb, gru, d1, d2 = model.layers[0], model.layers[1], model.layers[3], model.layers[5]
    assert emb.__class__.__name__ == "Embedding" and gru.__class__.__name__ == "GRU"
    assert d1.__class__.__name__ == "Dense" and d2.__class__.__name__ == "Dense"
    (e,), (k, rk, b), (w1, b1), (w2, b2) = (emb.get_weights(), gru.get_weights(), d1.get_weights(), d2.get_weights())
    assert b.shape == (2, k.shape[1]), "expected reset_after=True bias of shape (2, 3*units)"
    return {"embedding": e, "gru_kernel": k, "gru_recurrent_kernel": rk, "gru_bias": b,
            "dense1_kernel": w1, "dense1_bias": b1, "dense2_kernel": w2, "dense2_bias": b2}


def numpy_forward(w, ids):
    """Reference implementation of the model in plain numpy. The browser code is a port of exactly this."""
    units = w["gru_recurrent_kernel"].shape[0]
    emb = w["embedding"][ids].astype("float64")
    h = np.zeros((ids.shape[0], units))
    bi, br = w["gru_bias"][0].astype("float64"), w["gru_bias"][1].astype("float64")
    for t in range(ids.shape[1]):
        mx = emb[:, t, :] @ w["gru_kernel"] + bi
        mi = h @ w["gru_recurrent_kernel"] + br
        z = sigmoid(mx[:, :units] + mi[:, :units])
        r = sigmoid(mx[:, units:2 * units] + mi[:, units:2 * units])
        hh = np.tanh(mx[:, 2 * units:] + r * mi[:, 2 * units:])
        h = z * h + (1 - z) * hh
    a = np.maximum(h @ w["dense1_kernel"] + w["dense1_bias"], 0)
    return sigmoid(a @ w["dense2_kernel"] + w["dense2_bias"]).ravel()


def write_weights(name, w):
    out = MODEL_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    tensors, chunks, offset = {}, [], 0
    for key, arr in w.items():
        flat = np.asarray(arr, dtype="<f4").ravel()
        tensors[key] = {"shape": list(arr.shape), "offset": offset, "length": int(flat.size)}
        chunks.append(flat)
        offset += flat.size
    (out / "weights.bin").write_bytes(np.concatenate(chunks).tobytes())
    (out / "manifest.json").write_text(json.dumps(
        {"dtype": "float32", "units": int(w["gru_recurrent_kernel"].shape[0]), "tensors": tensors}, indent=1))


def main():
    data = P.prepare_data()
    vocab = data.vocab
    models = {name: keras.models.load_model(path) for name, path in MODELS.items()}
    weights = {name: extract_weights(m) for name, m in models.items()}

    # ---- self-checks: the exported weights + numpy maths reproduce Keras, and the vocabulary matches the model
    for name, m in models.items():
        keras_p = m.predict(data.X_test, verbose=0).ravel()
        np_p = numpy_forward(weights[name], data.X_test)
        assert np.abs(keras_p - np_p).max() < 1e-4, f"{name}: numpy port differs from Keras"
        print(f"{name}: numpy port vs Keras on the test set, max |diff| = {np.abs(keras_p - np_p).max():.2e}")
    main_acc = float(((models["main"].predict(data.X_test, verbose=0).ravel() >= 0.5) == data.y_test).mean())
    reported = json.loads((RESULTS / "gru_metrics.json").read_text())["accuracy"]
    assert abs(main_acc - reported) < 1e-9, "vocabulary/pipeline does not reproduce the reported test accuracy"

    # ---- model files
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, w in weights.items():
        write_weights(name, w)
    (MODEL_DIR / "vocab.json").write_text(json.dumps(vocab, separators=(",", ":")))
    min_words = int(data.train_df["tokens"].map(len).min())
    (MODEL_DIR / "config.json").write_text(json.dumps({
        "maxLen": C.MAX_LEN, "vocabSize": C.VOCAB_SIZE, "padToken": C.PAD_TOKEN, "oovToken": C.OOV_TOKEN,
        "padding": C.PADDING, "truncating": C.TRUNCATING, "threshold": C.THRESHOLD, "minTrainWords": min_words,
        "models": [{"id": "main", "name": "GRU", "dir": "main"}]}, indent=1))

    # ---- figures and results shown on the "Results" tab
    (PUBLIC / "figures").mkdir(parents=True, exist_ok=True)
    for f in FIGURES:
        shutil.copy(RESULTS / f, PUBLIC / "figures" / f)
    main_m = json.loads((RESULTS / "gru_metrics.json").read_text())
    hist = pd.read_csv(RESULTS / "gru_history.csv")
    seeds = pd.read_csv(RESULTS / "gru_seed_robustness.csv")
    keep = ["accuracy", "precision", "recall", "f1", "roc_auc", "training_time_sec", "trainable_params", "epochs_run",
            "best_epoch", "confusion_matrix"]
    results = {
        "main": {**{k: main_m[k] for k in keep}, "bootstrap_95ci": main_m["bootstrap_95ci"],
                 "validation_accuracy": float(hist.loc[main_m["best_epoch"] - 1, "val_accuracy"])},
        "robustness": {"runs": int(len(seeds)), **{f"{c}_{s}": float(getattr(seeds[c], s)()) for c in
                       ("test_acc", "test_f1", "test_auc") for s in ("mean", "std")}},
        "split": {"train": int(len(data.y_train)), "validation": int(len(data.y_val)), "test": int(len(data.y_test))},
    }
    (PUBLIC / "results.json").write_text(json.dumps(results, indent=1))

    # ---- reference outputs for the JavaScript tests
    examples = [e["text"] for e in json.loads((ROOT / "frontend" / "src" / "examples.json").read_text())]
    edge = ["", "   ", "???", "12345", "<b>Great</b> hotel, see http://example.com or www.example.org now!",
            "I can't say it wasn't good; won't you come? They'll say we've been; she'd said I'm sure it's fine. Cannot complain.",
            "We didn’t like it, and it isn’t cheap. THE ROOM WAS DIRTY!!!", "Café très bien, naïve staff – nice.",
            "\U0001F600 nice stay\U0001F600 but the bed was hard.\tSecond line\nthird line", "5-star, 10/10, room 305, $173 per night",
            "constructor toString __proto__ hasOwnProperty valueOf", "Loved it", "word " * 1000,
            "İstanbul hotel was GÜZEL and the KK sign", "It's the hotel's best. The guests' rooms weren't bad."]
    rng = np.random.default_rng(0)
    df = P.load_labelled_data()
    sampled = list(df.loc[rng.choice(len(df), size=60, replace=False), "text"])
    texts = examples + edge + sampled
    ids_list = [P.encode(P.tokenize(t), vocab)[:C.MAX_LEN] for t in texts]
    X = P.pad([P.encode(P.tokenize(t), vocab) for t in texts])
    probs = {name: m.predict(X, verbose=0).ravel() for name, m in models.items()}
    FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES.write_text(json.dumps([
        {"text": t, "clean": P.clean_text(t), "ids": ids, "main": float(probs["main"][i])}
        for i, (t, ids) in enumerate(zip(texts, ids_list))], separators=(",", ":")))
    print(f"exported {len(texts)} test vectors, vocabulary of {len(vocab)} ids, models: {', '.join(MODELS)}")


if __name__ == "__main__":
    main()
