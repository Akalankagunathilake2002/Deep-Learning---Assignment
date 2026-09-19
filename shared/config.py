"""Shared experiment settings.

Simple RNN, LSTM, GRU and 1D CNN must ALL import these values unchanged.
Only the model architecture is allowed to differ between the four models.
If the group changes a value here, every model has to be re-run.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "deceptive-opinion.csv"
SPLIT_PATH = ROOT / "shared" / "split_assignment.csv"   # fixed row -> train/val/test map

# ---- reproducibility -------------------------------------------------------
SEED = 42

# ---- labels: this dataset has no star rating, so `polarity` is the label ----
LABEL_COLUMN = "polarity"
LABEL_MAP = {"negative": 0, "positive": 1}

# ---- split: 80 / 10 / 10, stratified on the label ---------------------------
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1

# ---- text -> integer sequences ---------------------------------------------
VOCAB_SIZE = 5000          # total ids, including <PAD>=0 and <OOV>=1
MAX_LEN = 300              # tokens per review after cleaning
PAD_TOKEN, OOV_TOKEN = "<PAD>", "<OOV>"
PADDING = "pre"            # zeros in front, so a recurrent layer ends on real words
TRUNCATING = "post"        # long reviews keep their first MAX_LEN tokens

# ---- training (identical for all four models) -------------------------------
EMBED_DIM = 64
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5               # EarlyStopping patience, monitoring validation loss
MONITOR = "val_loss"
THRESHOLD = 0.5            # probability cut-off used for every model's test metrics
