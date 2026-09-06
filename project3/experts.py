"""Task 2: simulated experts with region-specific competence.

An expert here is a user model in the sense of lecture 9: a distribution p(a | s, theta) over
the label they would give, conditioned on the true topic. Competence is per topic, so the
expert is reliable in some regions of the input space and close to guessing in others. That
is the situation deferral is for. An expert who is uniformly better than the classifier makes
the problem trivial, and one who is uniformly worse makes it pointless.

The specialist is set up to be complementary to the classifier: strong exactly where the
baseline is weakest (Business and Sci/Tech) and poor where the baseline is strongest (Sports).
Its overall accuracy is well below the classifier's, so any gain has to come from deferring
selectively rather than from deferring often.
"""

from functools import lru_cache

import numpy as np

from . import data

PROFILES = {
    "specialist": {
        "label": "Specialist",
        "description": (
            "Reads business and technology news closely, skims the rest. Strong exactly where "
            "the classifier is weakest."
        ),
        "competence": {"World": 0.35, "Sports": 0.30, "Business": 0.95, "Sci/Tech": 0.95},
    },
    "generalist": {
        "label": "Generalist",
        "description": "Moderately reliable everywhere, with no particular speciality.",
        "competence": {"World": 0.70, "Sports": 0.70, "Business": 0.70, "Sci/Tech": 0.70},
    },
}

SEED = 7


@lru_cache(maxsize=8)
def predictions(name, split):
    """The label this expert gives for every article in a split.

    Correct with probability competence[true topic], otherwise a uniformly drawn wrong label.
    Seeded per (expert, split), so the whole page is reproducible.
    """
    truth = data.features()["y_train" if split == "train" else "y_test"]
    competence = np.array(
        [PROFILES[name]["competence"][topic] for topic in data.TOPICS], dtype=float
    )

    rng = np.random.default_rng(abs(hash((SEED, name, split))) % (2**32))
    correct = rng.random(len(truth)) < competence[truth]

    # A wrong answer is one of the other three topics, drawn uniformly.
    offsets = rng.integers(1, len(data.TOPICS), size=len(truth))
    return np.where(correct, truth, (truth + offsets) % len(data.TOPICS))


def correctness(name, split):
    truth = data.features()["y_train" if split == "train" else "y_test"]
    return (predictions(name, split) == truth).astype(int)


def report(name):
    """Measured behaviour of the expert, not the numbers it was configured with."""
    truth = data.features()["y_test"]
    right = correctness(name, "test")

    from . import classifier

    baseline = classifier.correctness("test")
    per_topic = {}
    for topic, code in data.CODES.items():
        mask = truth == code
        per_topic[topic] = {
            "expert": round(float(right[mask].mean()), 4),
            "classifier": round(float(baseline[mask].mean()), 4),
            "expert_better": bool(right[mask].mean() > baseline[mask].mean()),
        }

    return {
        "key": name,
        "label": PROFILES[name]["label"],
        "description": PROFILES[name]["description"],
        "accuracy": round(float(right.mean()), 4),
        "per_topic": per_topic,
        "beats_classifier_on": [t for t, row in per_topic.items() if row["expert_better"]],
    }
