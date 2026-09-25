"""
Utility Functions for 1D CNN Fake Review Detection
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


def create_dryrun_dataset(filepath: str, n_samples: int = 600) -> pd.DataFrame:
    """
    Creates a sample dry-run dataset for testing and verification purposes.
    Only used when the final user dataset is not yet provided at data/raw/fake_reviews.csv.

    Parameters:
        filepath (str): Destination CSV path.
        n_samples (int): Number of synthetic review samples to generate.

    Returns:
        pd.DataFrame: Generated sample dataframe.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    genuine_samples = [
        "The hotel room was clean and spacious, but the air conditioning was a bit noisy during the night.",
        "We ordered the seafood pasta and a margherita pizza. The crust was crispy and service was prompt.",
        "Great location right next to the metro station. However, breakfast options were somewhat limited.",
        "The battery life on this laptop lasts about 6 hours of normal office work. Decent keyboard feel.",
        "Customer support took two days to respond to my ticket, but they resolved the billing issue.",
        "Average stay. The bed was comfortable, but check-in had a long queue due to only one receptionist.",
        "The camera quality in daylight is excellent, but low light photos show noticeable digital noise.",
        "Nice quiet cafe with fast Wi-Fi and good espresso. A little pricey compared to other places nearby.",
        "The product arrived with slight packaging damage, but the item itself was completely intact.",
        "Decent sound quality for the price. The ear cushions get warm after two hours of continuous use.",
        "Staff was polite and helped store our luggage after checkout. Room amenities were standard.",
        "The display is sharp and vivid. Battery charges to 80% in about 40 minutes with the supplied adapter."
    ]

    fake_samples = [
        "ABSOLUTELY THE BEST HOTEL IN THE ENTIRE UNIVERSE! EVERYTHING WAS PERFECT AND FLAWLESS 10/10 MUST VISIT!",
        "THIS IS A MIRACLE PRODUCT! CURED ALL MY PROBLEMS IN JUST 24 HOURS! BUY IT IMMEDIATELY DO NOT HESITATE!",
        "WORST SCAM EVER! TERRIBLE FAKE COMPANY STOLE MY MONEY AND RUINED MY LIFE DO NOT BUY FROM THEM EVER!",
        "UNBELIEVABLY AMAZING EXPERIENCE! FIVE STARS! BEST SERVICE EVER IN THE WORLD BEST STAFF BEST EVERYTHING!",
        "A truly heavenly haven of luxury and unmatched perfection. Every single detail was a masterpiece of grandeur.",
        "Horrible horrible horrible! Disgusting garbage product that broke immediately! Total fraud and rip-off!",
        "Best purchase of my entire lifetime! Superior excellence beyond human imagination! Truly revolutionary!",
        "Do not trust anyone who rates this lower than 5 stars! This product will change your destiny forever!",
        "MAGICAL EXPERIENCE! Wonderful beyond words, completely flawless, sublime perfection from start to finish!",
        "SCAM ALERT! Cheap Chinese knockoff, completely unusable, threw it directly in the garbage trash!"
    ]

    records = []
    rng = random.Random(42)
    for i in range(n_samples // 2):
        base_g = rng.choice(genuine_samples)
        # Add slight natural variation
        records.append({
            "text": f"{base_g} Overall review index {i} experience was realistic.",
            "label": 0
        })
    for i in range(n_samples // 2):
        base_f = rng.choice(fake_samples)
        records.append({
            "text": f"{base_f} Review ID {i} unbelievable feedback!",
            "label": 1
        })

    rng.shuffle(records)
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    return df
