"""The Palmer Penguins dataset and the one split every model in this app shares.

The CSV is exported from the ``palmerpenguins`` package and committed, so the app has no
runtime dependency on that package and the numbers are reproducible for a grader.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = Path(__file__).resolve().parent / "data" / "penguins.csv"

TARGET = "species"
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex", "year"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE = 0.3
SEED = 0


@lru_cache(maxsize=1)
def load():
    """Complete-case penguins. Treat the returned frame as read-only."""
    frame = pd.read_csv(DATA_FILE)
    # A year is a label here, not a quantity to scale.
    frame["year"] = frame["year"].astype(str)
    return frame.dropna().reset_index(drop=True)


@lru_cache(maxsize=1)
def split():
    """One fixed stratified split, so every accuracy on the page is comparable."""
    frame = load()
    return train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=frame[TARGET],
    )


def species():
    return sorted(load()[TARGET].unique())


def summary():
    """What the dataset panel shows."""
    raw = pd.read_csv(DATA_FILE)
    frame = load()
    _, X_test, _, _ = split()
    return {
        "n_total": len(raw),
        "n_rows": len(frame),
        "n_dropped": len(raw) - len(frame),
        "n_test": len(X_test),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "distribution": frame[TARGET].value_counts().sort_index().to_dict(),
    }


def example(row):
    """One penguin, by position in the complete-case frame."""
    frame = load()
    index = max(0, min(int(row), len(frame) - 1))
    return index, frame.iloc[index]
