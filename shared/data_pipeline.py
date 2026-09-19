"""Shared data pipeline: load -> label -> de-duplicate -> split -> clean -> vocabulary -> pad.

Every model (Simple RNN, LSTM, GRU, 1D CNN) calls `prepare_data()` so that the
dataset, split, preprocessing, vocabulary, padding and class weights are identical.

Leakage rule: the vocabulary and class weights are built from the TRAINING rows only.
Validation and test rows are only ever *encoded* with the training vocabulary.
"""
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shared import config as C

# --------------------------------------------------------------------------- #
# 1. Load, label, de-duplicate
# --------------------------------------------------------------------------- #
def load_labelled_data() -> pd.DataFrame:
    """Read the CSV, map polarity -> 0/1 and drop exact duplicate reviews.

    Duplicates must go before splitting, otherwise a copy of a training review could
    land in the test set and the test set would no longer be unseen.
    `row_id` is the row number in the original CSV, so the split is easy to audit.
    """
    df = pd.read_csv(C.DATA_PATH)
    df["row_id"] = np.arange(len(df))
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df[C.LABEL_COLUMN].map(C.LABEL_MAP)
    assert df["label"].notna().all(), "unexpected label value"
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    return df[["row_id", "text", "label", "hotel", "deceptive", "source"]]


# --------------------------------------------------------------------------- #
# 2. Stratified 80/10/10 split (saved to disk so every teammate gets the same rows)
# --------------------------------------------------------------------------- #
def make_split(df: pd.DataFrame) -> pd.DataFrame:
    train_ids, rest_ids, _, rest_y = train_test_split(
        df["row_id"], df["label"], test_size=C.VAL_FRAC + C.TEST_FRAC,
        stratify=df["label"], random_state=C.SEED)
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
        stratify=rest_y, random_state=C.SEED)
    split = pd.concat([
        pd.DataFrame({"row_id": train_ids, "split": "train"}),
        pd.DataFrame({"row_id": val_ids, "split": "val"}),
        pd.DataFrame({"row_id": test_ids, "split": "test"}),
    ]).sort_values("row_id").reset_index(drop=True)
    return split


def load_or_create_split(df: pd.DataFrame) -> pd.DataFrame:
    """Use the committed split file if it exists; create it once if it does not."""
    if C.SPLIT_PATH.exists():
        split = pd.read_csv(C.SPLIT_PATH)
        if set(split["row_id"]) != set(df["row_id"]):
            raise ValueError(
                f"{C.SPLIT_PATH.name} does not match the dataset. Was the CSV changed or "
                "de-duplicated differently? Do NOT regenerate it on your own - ask the group.")
        return split
    split = make_split(df)
    split.to_csv(C.SPLIT_PATH, index=False)
    return split


# --------------------------------------------------------------------------- #
# 3. Text cleaning (identical for all models; a pure per-review function, nothing fitted)
# --------------------------------------------------------------------------- #
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_CONTRACTIONS = [                       # keep negation: "didn't" -> "did not", not "didn t"
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"\bcannot\b", "can not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"),
    (r"'d\b", " would"), (r"'m\b", " am"), (r"'s\b", ""),
]
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """lowercase -> strip HTML -> strip URLs -> expand contractions -> keep letters only.

    Stop-words are deliberately NOT removed: words such as "not", "no" and "never"
    flip the sentiment of a sentence.
    """
    text = text.lower().replace("’", "'")
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    for pattern, replacement in _CONTRACTIONS:
        text = re.sub(pattern, replacement, text)
    text = _NON_LETTER.sub(" ", text)           # digits, punctuation, symbols
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


# --------------------------------------------------------------------------- #
# 4. Vocabulary (TRAIN ONLY), encoding and padding
# --------------------------------------------------------------------------- #
def build_vocab(train_token_lists: list[list[str]], vocab_size: int = C.VOCAB_SIZE) -> dict[str, int]:
    """Most frequent training words -> ids. 0 = <PAD>, 1 = <OOV>, real words start at 2."""
    counts = Counter(tok for tokens in train_token_lists for tok in tokens)
    words = [w for w, _ in counts.most_common(vocab_size - 2)]
    vocab = {C.PAD_TOKEN: 0, C.OOV_TOKEN: 1}
    vocab.update({w: i + 2 for i, w in enumerate(words)})
    return vocab


def encode(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    oov = vocab[C.OOV_TOKEN]
    return [vocab.get(t, oov) for t in tokens]


def decode(ids, vocab: dict[str, int], skip_pad: bool = True) -> list[str]:
    """Inverse of `encode`: ids -> words. Unknown words come back as <OOV> (the original is lost)."""
    inverse = {i: w for w, i in vocab.items()}
    return [inverse[int(i)] for i in ids if not (skip_pad and int(i) == 0)]


def pad(sequences: list[list[int]], max_len: int = C.MAX_LEN) -> np.ndarray:
    out = np.zeros((len(sequences), max_len), dtype="int32")
    for i, seq in enumerate(sequences):
        seq = seq[:max_len] if C.TRUNCATING == "post" else seq[-max_len:]
        if C.PADDING == "pre":
            out[i, max_len - len(seq):] = seq
        else:
            out[i, :len(seq)] = seq
    return out


# --------------------------------------------------------------------------- #
# 5. Class weights (TRAIN ONLY) - same formula as sklearn's class_weight="balanced"
# --------------------------------------------------------------------------- #
def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """weight_c = n_samples / (n_classes * n_samples_in_class_c)"""
    classes, counts = np.unique(y_train, return_counts=True)
    return {int(c): float(len(y_train) / (len(classes) * n)) for c, n in zip(classes, counts)}


# --------------------------------------------------------------------------- #
# 6. One call that returns everything a model needs
# --------------------------------------------------------------------------- #
@dataclass
class Data:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    vocab: dict
    class_weight: dict
    train_df: pd.DataFrame     # raw + cleaned text, kept for EDA / error analysis
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def prepare_data() -> Data:
    df = load_labelled_data()
    split = load_or_create_split(df)
    df = df.merge(split, on="row_id")
    df["clean"] = df["text"].map(clean_text)
    df["tokens"] = df["clean"].str.split()

    parts = {name: g.reset_index(drop=True) for name, g in df.groupby("split")}
    vocab = build_vocab(parts["train"]["tokens"].tolist())          # <- TRAIN ONLY

    def to_xy(part: pd.DataFrame):
        seqs = [encode(tokens, vocab) for tokens in part["tokens"]]
        return pad(seqs), part["label"].to_numpy(dtype="int32")

    X_train, y_train = to_xy(parts["train"])
    X_val, y_val = to_xy(parts["val"])
    X_test, y_test = to_xy(parts["test"])
    return Data(X_train, y_train, X_val, y_val, X_test, y_test, vocab,
                compute_class_weights(y_train), parts["train"], parts["val"], parts["test"])
