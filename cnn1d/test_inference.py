"""
Interactive and batch inference tester for the 1D CNN hotel review sentiment classification model.
Trained on the common dataset (dataset/deceptive-opinion.csv) under the SE4050 assignment protocol.

Usage:
    # 1. Single review test from CLI:
    python cnn1d/test_inference.py "The hotel was absolutely amazing, spotless rooms and friendly staff!"

    # 2. Interactive prompt (run without arguments):
    python cnn1d/test_inference.py
"""

import json
import os
import sys
from pathlib import Path

# Set up paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow as tf
from shared.data_pipeline import clean_text, encode, pad

CNN_MODEL_PATH = REPO_ROOT / "cnn1d" / "results" / "cnn1d_model.keras"
VOCAB_PATH = REPO_ROOT / "frontend" / "public" / "model" / "vocab.json"


def load_vocab():
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocabulary JSON not found at {VOCAB_PATH}")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"1D CNN model not found at {CNN_MODEL_PATH}")
    return tf.keras.models.load_model(CNN_MODEL_PATH)


def predict_review(text: str, cnn_model, vocab, max_len: int = 300):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    encoded_seq = encode(tokens, vocab)
    padded_seq = pad([encoded_seq], max_len=max_len)

    cnn_prob = float(cnn_model.predict(padded_seq, verbose=0)[0][0])
    cnn_label = "Positive" if cnn_prob >= 0.5 else "Negative"
    cnn_conf = cnn_prob if cnn_prob >= 0.5 else (1.0 - cnn_prob)

    return {
        "text": text,
        "cleaned": cleaned,
        "tokens": len(tokens),
        "cnn": {
            "probability": cnn_prob,
            "label": cnn_label,
            "confidence": cnn_conf,
        },
    }


def print_result_card(res: dict):
    print("\n" + "─" * 68)
    print(f"INPUT REVIEW: \"{res['text']}\"")
    print(f"CLEANED     : \"{res['cleaned']}\" ({res['tokens']} tokens)")
    print("─" * 68)

    # 1D CNN result
    cnn = res["cnn"]
    cnn_bar = "█" * int(cnn["probability"] * 25) + "░" * (25 - int(cnn["probability"] * 25))
    print(f"  [1D CNN]    Prediction: {cnn['label'].upper():<8} | P(Pos): {cnn['probability']:.4f}  [{cnn_bar}]  ({cnn['confidence']*100:.1f}% conf)")
    print("─" * 68)


def main():
    print("=" * 68)
    print(" 1D CNN Hotel Review Sentiment Inference Tester")
    print(" Implemented by: Dilmith (SE4050 Deep Learning)")
    print(" Common Dataset: dataset/deceptive-opinion.csv")
    print("=" * 68)
    print("Loading model and vocabulary...")
    vocab = load_vocab()
    cnn_model = load_model()
    print("Model loaded successfully!\n")

    # If argument provided, test it
    if len(sys.argv) > 1:
        review_text = " ".join(sys.argv[1:])
        res = predict_review(review_text, cnn_model, vocab)
        print_result_card(res)
        return

    # Sample reviews
    demo_samples = [
        "The hotel was absolutely amazing, spotless rooms and friendly staff!",
        "The room smelled terrible, AC was broken, and customer service refused to help.",
        "The view was okay, but the shower was not clean at all.",
        "I was skeptical at first, but the bed was comfortable and breakfast was delicious.",
    ]

    print("Running 4 Benchmark Reviews:")
    for sample in demo_samples:
        res = predict_review(sample, cnn_model, vocab)
        print_result_card(res)

    print("\nHow to test your own reviews:")
    print("  python cnn1d/test_inference.py \"Type your hotel review here\"")
    print("  Or test live in your browser at: http://localhost:5173/ (Click '1D CNN')\n")


if __name__ == "__main__":
    main()
