"""
Utility Functions for 1D CNN Hotel Review Sentiment Classification
Author: Dilmith
Course: SE4050 - Deep Learning
"""

import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd


def set_seeds(seed: int = 42) -> None:
    """
    Sets reproducible random seeds across Python, NumPy, and TensorFlow.

    Parameters:
        seed (int): The integer seed to apply. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Enable deterministic operations where available in TensorFlow
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def ensure_directories(base_dir: str = ".") -> None:
    """
    Ensures all required project directories exist.
    """
    directories = [
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "notebooks"),
        os.path.join(base_dir, "src"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "results", "figures"),
        os.path.join(base_dir, "results", "metrics"),
        os.path.join(base_dir, "results", "experiments"),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes}m {rem_seconds:.1f}s"


def resolve_dataset_path(base_dir: str = ".") -> str:
    """
    Resolves the absolute or relative path to the common dataset:
    dataset/deceptive-opinion.csv

    Parameters:
        base_dir (str): Base directory to search from.

    Returns:
        str: Verified path to deceptive-opinion.csv.
    """
    candidates = [
        os.path.join(base_dir, "dataset", "deceptive-opinion.csv"),
        os.path.join(base_dir, "..", "dataset", "deceptive-opinion.csv"),
        os.path.join(base_dir, "..", "..", "dataset", "deceptive-opinion.csv"),
        "dataset/deceptive-opinion.csv",
        "../dataset/deceptive-opinion.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Common dataset 'dataset/deceptive-opinion.csv' not found. "
        "Please ensure deceptive-opinion.csv exists in the dataset/ directory."
    )

