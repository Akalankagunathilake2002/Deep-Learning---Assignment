"""
Data Preprocessing, Cleaning, Splitting, and Tokenization Module
Author: Dilmith
Course: SE4050 - Deep Learning
Component: 1D CNN
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==============================================================================
# 1. DATASET LOADING & INSPECTION
# ==============================================================================

def load_and_inspect_dataset(
    filepath: str,
    text_col: str = "text",
    label_col: str = "label"
) -> pd.DataFrame:
    """
    Loads dataset from CSV and verifies essential columns.

    Parameters:
        filepath (str): Path to CSV file.
        text_col (str): Name of text column.
        label_col (str): Name of binary label column (0=Genuine, 1=Fake).

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please place your dataset CSV there "
            f"or update DATASET_PATH."
        )

    df = pd.read_csv(filepath)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataset columns: {list(df.columns)}")

    # If label_col not found, check for standard fake review label column names
    if label_col not in df.columns:
        for candidate in ["label", "deceptive", "is_fake", "fake", "target"]:
            if candidate in df.columns:
                print(f"[INFO] Specified label column '{label_col}' not found. Using detected column '{candidate}'.")
                label_col = candidate
                break
        else:
            raise ValueError(f"Column '{label_col}' not found in dataset columns: {list(df.columns)}")

    # Standardize string labels to binary 0 (Genuine/Truthful) and 1 (Fake/Deceptive)
    unique_vals = set(df[label_col].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        mapping = {
            "truthful": 0, "genuine": 0, "real": 0, "0": 0,
            "deceptive": 1, "fake": 1, "1": 1
        }
        # Check lowercase representation
        str_mapping = {}
        for val in unique_vals:
            val_lower = str(val).strip().lower()
            if val_lower in mapping:
                str_mapping[val] = mapping[val_lower]
        if len(str_mapping) == len(unique_vals):
            df[label_col] = df[label_col].map(str_mapping)
            print(f"[INFO] Successfully mapped label values {unique_vals} to binary {0, 1}.")
        else:
            # Fallback factorize if custom 2 classes
            if len(unique_vals) == 2:
                sorted_vals = sorted(list(unique_vals))
                fallback_map = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                df[label_col] = df[label_col].map(fallback_map)
                print(f"[INFO] Mapped 2-class labels {fallback_map} to binary {0, 1}.")

    df[label_col] = df[label_col].astype(int)
    if "label" not in df.columns:
        df["label"] = df[label_col]
    return df


def get_dataset_inspection_dict(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label"
) -> dict:
    """
    Computes summary inspection statistics for the raw dataset.
    """
    total_records = len(df)
    missing_text = int(df[text_col].isna().sum())
    missing_label = int(df[label_col].isna().sum())
    duplicate_reviews = int(df.duplicated(subset=[text_col]).sum())
    class_counts = df[label_col].value_counts().to_dict()
    genuine_count = int(class_counts.get(0, 0))
    fake_count = int(class_counts.get(1, 0))

    return {
        "total_records": total_records,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_text": missing_text,
        "missing_label": missing_label,
        "duplicate_reviews": duplicate_reviews,
        "genuine_count": genuine_count,
        "fake_count": fake_count,
        "genuine_percentage": (genuine_count / total_records * 100) if total_records > 0 else 0.0,
        "fake_percentage": (fake_count / total_records * 100) if total_records > 0 else 0.0,
    }


# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

def compute_review_length_stats(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label"
) -> pd.DataFrame:
    """
    Calculates character and word length statistics overall and by class.
    """
    df_temp = df.copy()
    # Compute character length and word counts
    df_temp["char_length"] = df_temp[text_col].astype(str).apply(len)
    df_temp["word_count"] = df_temp[text_col].astype(str).apply(lambda t: len(t.split()))

    stats_list = []
    # Overall
    stats_list.append({
        "Category": "Overall",
        "Review Count": len(df_temp),
        "Mean Words": df_temp["word_count"].mean(),
        "Median Words": df_temp["word_count"].median(),
        "Std Words": df_temp["word_count"].std(),
        "Min Words": df_temp["word_count"].min(),
        "Max Words": df_temp["word_count"].max(),
        "Mean Characters": df_temp["char_length"].mean(),
    })

    # By Class: 0 = Genuine, 1 = Fake
    for label, label_name in [(0, "Genuine (0)"), (1, "Fake (1)")]:
        sub = df_temp[df_temp[label_col] == label]
        if len(sub) > 0:
            stats_list.append({
                "Category": label_name,
                "Review Count": len(sub),
                "Mean Words": sub["word_count"].mean(),
                "Median Words": sub["word_count"].median(),
                "Std Words": sub["word_count"].std(),
                "Min Words": sub["word_count"].min(),
                "Max Words": sub["word_count"].max(),
                "Mean Characters": sub["char_length"].mean(),
            })

    return pd.DataFrame(stats_list)


def plot_eda_distributions(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    save_dir: str = "results/figures"
) -> None:
    """
    Generates and saves the three core EDA visualizations:
    1. Class distribution
    2. Review length distribution (words)
    3. Review length comparison between Genuine and Fake
    """
    os.makedirs(save_dir, exist_ok=True)
    df_temp = df.copy()
    df_temp["word_count"] = df_temp[text_col].astype(str).apply(lambda t: len(t.split()))
    df_temp["Class_Name"] = df_temp[label_col].map({0: "Genuine (0)", 1: "Fake (1)"})

    palette = {"Genuine (0)": "#2b5c8f", "Fake (1)": "#d95f02"}

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(7, 5))
    counts = df_temp["Class_Name"].value_counts().sort_index()
    bars = plt.bar(counts.index, counts.values, color=["#2b5c8f", "#d95f02"], width=0.5, edgecolor="black")
    plt.title("Class Distribution of Online Reviews (SE4050)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Review Class", fontsize=11, labelpad=8)
    plt.ylabel("Number of Reviews", fontsize=11, labelpad=8)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + (max(counts.values) * 0.015),
                 f"{int(height)} ({height/len(df_temp)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_class_distribution.png"), dpi=300)
    plt.close()

    # 2. Review Length Distribution Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df_temp["word_count"], bins=40, kde=True, color="#1b7837", edgecolor="black")
    plt.axvline(df_temp["word_count"].mean(), color="red", linestyle="--", label=f"Mean: {df_temp['word_count'].mean():.1f} words")
    plt.axvline(df_temp["word_count"].median(), color="blue", linestyle=":", label=f"Median: {df_temp['word_count'].median():.1f} words")
    plt.title("Review Length Distribution (Word Count)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Review Length (Words)", fontsize=11, labelpad=8)
    plt.ylabel("Frequency", fontsize=11, labelpad=8)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_review_length_distribution.png"), dpi=300)
    plt.close()

    # 3. Review Length Comparison by Class (Boxplot)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Class_Name", y="word_count", data=df_temp, hue="Class_Name", palette=palette, legend=False, width=0.45, fliersize=3)
    plt.title("Review Length Comparison by Class", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Review Class", fontsize=11, labelpad=8)
    plt.ylabel("Review Length (Words)", fontsize=11, labelpad=8)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_review_length_by_class.png"), dpi=300)
    plt.close()


# ==============================================================================
# 3. TEXT CLEANING
# ==============================================================================

def clean_text(text: str) -> str:
    """
    Cleans raw review text for neural text classification.

    Design rationale:
    1. Lowercase normalization: Unifies vocabulary tokens (e.g. 'Hotel' vs 'hotel').
    2. HTML tag removal: Web-scraped reviews often retain tags (e.g., '<br />', '<p>').
    3. URL removal: Web links add noise and high-entropy unique tokens.
    4. Whitespace normalization: Cleans consecutive tabs, newlines, and spaces.
    5. Selective punctuation normalization: Preserves standard contractions and word boundaries
       while removing non-informative symbols/control characters. Stopwords and punctuation
       like exclamation marks are NOT stripped aggressively, because excessive punctuation
       and stylistic stopwords often provide strong indicators of deception.

    Parameters:
        text (str): Raw input text.

    Returns:
        str: Cleaned review string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags (<p>, <br/>, etc.)
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Replace newlines, tabs, and multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Clean unprintable control characters, keeping letters, numbers, and standard punctuation
    text = re.sub(r"[^\w\s.,!?'\"-]", "", text)

    return text.strip()


def remove_duplicate_reviews(
    df: pd.DataFrame,
    text_col: str = "cleaned_text"
) -> pd.DataFrame:
    """
    Identifies and removes duplicate reviews prior to train/val/test splitting.

    Parameters:
        df (pd.DataFrame): Dataset dataframe.
        text_col (str): Column name containing review text.

    Returns:
        pd.DataFrame: Deduplicated dataframe with reset index.
    """
    initial_count = len(df)
    duplicates_count = int(df.duplicated(subset=[text_col]).sum())
    df_dedup = df.drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
    retained_count = len(df_dedup)

    print(f"[Deduplication] Initial reviews: {initial_count:,}")
    print(f"[Deduplication] Duplicate reviews removed: {duplicates_count:,}")
    print(f"[Deduplication] Unique reviews retained: {retained_count:,}")

    return df_dedup


def verify_zero_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "cleaned_text"
) -> None:
    """
    Performs rigorous assertion checks to confirm zero text overlap
    and zero index overlap across train, validation, and test sets.

    Parameters:
        train_df (pd.DataFrame): Training subset.
        val_df (pd.DataFrame): Validation subset.
        test_df (pd.DataFrame): Testing subset.
        text_col (str): Text column to check for overlap.
    """
    train_texts = set(train_df[text_col].dropna())
    val_texts = set(val_df[text_col].dropna())
    test_texts = set(test_df[text_col].dropna())

    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)

    # Strict assertion checks to ensure zero data leakage
    assert len(train_val_overlap) == 0, (
        f"Data Leakage Error: {len(train_val_overlap)} reviews overlap between Train and Validation!"
    )
    assert len(train_test_overlap) == 0, (
        f"Data Leakage Error: {len(train_test_overlap)} reviews overlap between Train and Test!"
    )
    assert len(val_test_overlap) == 0, (
        f"Data Leakage Error: {len(val_test_overlap)} reviews overlap between Validation and Test!"
    )

    # Verify index uniqueness
    assert len(set(train_df.index).intersection(set(val_df.index))) == 0, "Index overlap between Train and Val!"
    assert len(set(train_df.index).intersection(set(test_df.index))) == 0, "Index overlap between Train and Test!"
    assert len(set(val_df.index).intersection(set(test_df.index))) == 0, "Index overlap between Val and Test!"

    print("[Verification Passed] Zero text overlap confirmed across Train, Val, and Test sets:")
    print(f"  - Train ∩ Val:  {len(train_val_overlap)} common reviews (Assertion: 0)")
    print(f"  - Train ∩ Test: {len(train_test_overlap)} common reviews (Assertion: 0)")
    print(f"  - Val ∩ Test:   {len(val_test_overlap)} common reviews (Assertion: 0)")


# ==============================================================================
# 4. STRATIFIED TRAIN / VALIDATION / TEST SPLIT (80 / 10 / 10)
# ==============================================================================

def create_stratified_splits(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_col: str = "label",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    save_dir: str = "data/processed"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates stratified Train (80%), Validation (10%), and Test (10%) splits.
    Saves the splits to CSV and saves split indices to JSON so that group members
    (Simple RNN, LSTM, GRU) can load the exact same observations for fair comparison.

    Parameters:
        df (pd.DataFrame): Dataset with cleaned text.
        text_col (str): Text feature column.
        label_col (str): Target label column.
        train_ratio (float): Ratio for training (default: 0.8).
        val_ratio (float): Ratio for validation (default: 0.1).
        test_ratio (float): Ratio for test (default: 0.1).
        seed (int): Reproducible random seed (42).
        save_dir (str): Directory to save processed splits.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Step 1: Separate Train from Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df[label_col],
        random_state=seed
    )

    # Step 2: Split Temp into Validation and Test (50/50 of the remaining 20% = 10% each)
    val_share_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        stratify=temp_df[label_col],
        random_state=seed
    )

    # Verify zero overlap before saving
    verify_zero_overlap(train_df, val_df, test_df, text_col=text_col)

    # Save to disk for group sharing & fair comparison
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=True)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=True)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=True)

    # Save indices to JSON
    split_indices = {
        "train_indices": list(train_df.index),
        "val_indices": list(val_df.index),
        "test_indices": list(test_df.index),
        "random_seed": seed,
        "total_unique_records": len(df),
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        }
    }
    with open(os.path.join(save_dir, "split_indices.json"), "w") as f:
        json.dump(split_indices, f, indent=2)

    return train_df, val_df, test_df


# ==============================================================================
# 5. TOKENIZATION & FIXED-LENGTH PADDING
# ==============================================================================

def fit_tokenizer_on_train(
    train_texts: pd.Series,
    vocab_size: int = 10000,
    oov_token: str = "<OOV>"
) -> Tokenizer:
    """
    Fits Keras Tokenizer STRICTLY on training data to prevent data leakage.

    Parameters:
        train_texts (pd.Series): Training review strings.
        vocab_size (int): Max vocabulary size.
        oov_token (str): Token used for out-of-vocabulary words.

    Returns:
        Tokenizer: Fitted Keras Tokenizer.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def transform_texts_to_padded_sequences(
    tokenizer: Tokenizer,
    texts: pd.Series,
    max_length: int = 200,
    padding: str = "post",
    truncating: str = "post"
) -> np.ndarray:
    """
    Transforms text series into fixed-length padded integer sequences.

    Parameters:
        tokenizer (Tokenizer): Fitted Keras Tokenizer.
        texts (pd.Series): Review texts.
        max_length (int): Maximum sequence length.
        padding (str): 'post' or 'pre' padding.
        truncating (str): 'post' or 'pre' truncating.

    Returns:
        np.ndarray: Padded integer sequences of shape (num_samples, max_length).
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding, truncating=truncating)
    return padded
