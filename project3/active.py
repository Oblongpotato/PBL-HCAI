"""Task 4: finding out what the expert is good at, without being told.

Task 3 assumed we already knew, for every training article, whether the expert would have
been right. That is the expensive assumption: collecting it means asking a human about all
8000 articles. Here we start with none of it and may only ask about a few.

The thing being learned is the expert's competence profile, so the query strategy should
target the decision that profile feeds: defer or not. That decision is hardest where the two
parties are closest, so the informativeness term is

    u_info(x) = -| P(expert right | x) - P(classifier right | x) |

which is largest when the estimated probabilities nearly tie. On its own that happily spends
the budget on outliers, so following lecture 6 it is combined with a representativeness term,
the mean similarity of x to the rest of the pool.

Two baselines: querying at random, and querying where the classifier is least confident,
which is the obvious thing to do if you forget that the point is to learn about the *expert*.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import classifier, data, defer, experts

BUDGET = 1000
ROUND = 50
ANCHORS = 600
SEED = 0


def _density(Z, rng):
    """Mean cosine similarity to a random sample of the pool."""
    normalised = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    anchors = normalised[rng.choice(len(Z), size=min(ANCHORS, len(Z)), replace=False)]
    return (normalised @ anchors.T).mean(axis=1)


def _competence_model(Z, queried, answers):
    """P(expert right | x), fitted only on the articles we actually asked about."""
    if len(set(answers)) < 2:
        constant = float(np.mean(answers)) if len(answers) else 0.5
        return lambda features: np.full(len(features), constant)

    model = LogisticRegression(max_iter=1000)
    model.fit(Z[queried], answers)
    return lambda features: model.predict_proba(features)[:, 1]


def _system_accuracy(expert_right_test, classifier_right_test, defer_mask):
    return float(np.where(defer_mask, expert_right_test, classifier_right_test).mean())


def run_strategy(name, expert_name, seed=SEED):
    """One active-learning run, measured on the test set after every round."""
    parts = data.features()
    Z_train, Z_test = defer.deferral_features()

    expert_right_train = experts.correctness(expert_name, "train")
    expert_right_test = experts.correctness(expert_name, "test")
    classifier_right_test = classifier.correctness("test")

    # What the classifier itself thinks its chances are, honest on train, plain on test.
    p_classifier_train = classifier.out_of_fold_probabilities().max(axis=1)
    p_classifier_test = classifier.confidence("test")

    rng = np.random.default_rng(seed)
    density = _density(Z_train, rng)

    queried = []
    available = np.ones(len(Z_train), dtype=bool)
    pool = np.flatnonzero(available)
    curve = []

    while len(queried) < BUDGET:
        if not queried or name == "random":
            picks = rng.choice(pool, size=ROUND, replace=False)
        elif name == "classifier_uncertainty":
            picks = pool[np.argsort(-p_classifier_train[pool])[-ROUND:]]
        else:
            # Only this strategy reads the competence model, so only this branch fits one.
            estimate = _competence_model(Z_train, queried, expert_right_train[queried])
            closeness = -np.abs(estimate(Z_train[pool]) - p_classifier_train[pool])
            picks = pool[np.argsort(closeness * density[pool])[-ROUND:]]

        queried.extend(int(index) for index in picks)
        available[picks] = False
        pool = np.flatnonzero(available)

        estimate = _competence_model(Z_train, queried, expert_right_train[queried])
        p_expert_test = estimate(Z_test)
        accuracy = _system_accuracy(
            expert_right_test, classifier_right_test, p_expert_test > p_classifier_test
        )
        curve.append({"queries": len(queried), "system_accuracy": round(accuracy, 4)})

    # How well the learned profile matches the truth it never saw.
    estimate = _competence_model(Z_train, queried, expert_right_train[queried])
    p_expert_test = estimate(Z_test)
    truth = parts["y_test"]
    learned = {
        topic: round(float(p_expert_test[truth == code].mean()), 3)
        for topic, code in data.CODES.items()
    }
    actual = {
        topic: round(float(expert_right_test[truth == code].mean()), 3)
        for topic, code in data.CODES.items()
    }

    return {
        "strategy": name,
        "curve": curve,
        "competence": [
            {"topic": topic, "learned": learned[topic], "actual": actual[topic]}
            for topic in data.TOPICS
        ],
    }


def run(expert_name):
    """All three strategies, plus the ceiling from knowing every expert label."""
    parts = data.features()
    Z_train, Z_test = defer.deferral_features()

    strategies = [
        run_strategy(name, expert_name)
        for name in ("proposed", "classifier_uncertainty", "random")
    ]

    # What the same policy achieves with the whole training set labelled by the expert.
    everything = _competence_model(
        Z_train, list(range(len(Z_train))), experts.correctness(expert_name, "train")
    )
    full = _system_accuracy(
        experts.correctness(expert_name, "test"),
        classifier.correctness("test"),
        everything(Z_test) > classifier.confidence("test"),
    )

    return {
        "expert": expert_name,
        "strategies": strategies,
        "full_supervision": round(full, 4),
        "baseline_accuracy": round(float(classifier.correctness("test").mean()), 4),
        "budget": BUDGET,
        "round": ROUND,
    }
