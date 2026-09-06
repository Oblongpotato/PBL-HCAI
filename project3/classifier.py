"""Task 1: the classifier trained on every available label.

This is the baseline the human-AI team has to match or beat, so it is deliberately a strong
one for the budget: TF-IDF with bigrams into multinomial logistic regression. It also
supplies the class probabilities that both deferral strategies need.
"""

from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from . import data


@lru_cache(maxsize=1)
def baseline():
    """Fit on the full training set and report how it does per topic."""
    parts = data.features()
    model = LogisticRegression(max_iter=2000, C=4.0)
    model.fit(parts["X_train"], parts["y_train"])

    probabilities = {
        "train": model.predict_proba(parts["X_train"]),
        "test": model.predict_proba(parts["X_test"]),
    }
    predictions = {split: array.argmax(axis=1) for split, array in probabilities.items()}

    truth = parts["y_test"]
    per_topic = {
        topic: float(accuracy_score(truth[truth == code], predictions["test"][truth == code]))
        for topic, code in data.CODES.items()
    }

    return {
        "model": model,
        "probabilities": probabilities,
        "predictions": predictions,
        "accuracy": float(accuracy_score(truth, predictions["test"])),
        "per_topic": per_topic,
        "confusion": confusion_matrix(truth, predictions["test"]).tolist(),
    }


def correctness(split):
    """1 where the classifier is right, 0 where it is wrong."""
    truth = data.features()["y_train" if split == "train" else "y_test"]
    return (baseline()["predictions"][split] == truth).astype(int)


def confidence(split):
    """The classifier's own certainty, which confidence-based rejection thresholds."""
    return baseline()["probabilities"][split].max(axis=1)


def report():
    """Serialisable summary for the page."""
    result = baseline()
    return {
        "accuracy": round(result["accuracy"], 4),
        "per_topic": {topic: round(value, 4) for topic, value in result["per_topic"].items()},
        "confusion": result["confusion"],
        "topics": data.TOPICS,
        "mean_confidence": round(float(np.mean(confidence("test"))), 4),
    }
