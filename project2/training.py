"""The pool of candidate models, and the complexity-penalised choice between them.

Lecture 4's argument, made operational: fit a range of models at different regularisation
strengths, then let the user trade accuracy against complexity with a single weight.

Two different regularisers are in play and the interface must not confuse them:

* ``max_leaf_nodes`` / ``C`` constrain the model *while it is being fitted*;
* lambda weighs the complexity term *after* fitting, when picking one model from the pool.
"""

from functools import lru_cache

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import data

FAMILIES = {"tree": "Decision tree", "logistic": "Logistic regression"}

COMPLEXITY_LABEL = {"tree": "leaves", "logistic": "features used"}
FIT_REGULARISER = {"tree": "max_leaf_nodes", "logistic": "C"}

LEAF_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
C_GRID = [round(value, 5) for value in np.logspace(-2.5, 1.5, 24)]

ZERO = 1e-6

# Each family's accuracy/complexity frontier turns over in a different place, so a single
# range leaves one of them with a dead slider. A tree switches model at lambda 0.005 and
# 0.010; logistic regression only at 0.050 and 0.128.
LAMBDA_MAX = {"tree": 0.02, "logistic": 0.15}


def _preprocessor(family):
    """Trees are scale-invariant, so leaving their thresholds in millimetres and grams
    keeps the rendered tree readable. The logistic model needs scaling for its weights
    to be comparable and for the L1 penalty to act evenly across features."""
    numeric = "passthrough" if family == "tree" else StandardScaler()
    return ColumnTransformer(
        [
            ("numeric", numeric, data.NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), data.CATEGORICAL_FEATURES),
        ]
    )


def _estimator(family, value):
    if family == "tree":
        return DecisionTreeClassifier(max_leaf_nodes=int(value), random_state=data.SEED)
    # l1_ratio=1 is a pure L1 penalty, which is what drives coefficients to exactly zero
    # and makes "number of features used" a meaningful complexity measure.
    return LogisticRegression(solver="saga", l1_ratio=1, C=value, max_iter=5000)


def grid(family):
    return LEAF_GRID if family == "tree" else C_GRID


def lambda_max(family):
    return LAMBDA_MAX[family]


def source_features(pipeline):
    """Map each encoded column back to the original feature it came from."""
    encoded = pipeline.named_steps["prepare"].get_feature_names_out()
    origins = []
    for name in encoded:
        _, _, rest = name.partition("__")
        origins.append(next(f for f in data.FEATURES if rest == f or rest.startswith(f + "_")))
    return origins


def used_features(pipeline):
    """Original features the logistic model still relies on."""
    origins = source_features(pipeline)
    weights = np.abs(pipeline.named_steps["model"].coef_).max(axis=0)
    return sorted({origin for origin, weight in zip(origins, weights) if weight > ZERO},
                  key=data.FEATURES.index)


def complexity(family, pipeline):
    """Omega(f): leaves for a tree, features actually used for logistic regression."""
    if family == "tree":
        return int(pipeline.named_steps["model"].get_n_leaves())
    return len(used_features(pipeline))


@lru_cache(maxsize=2)
def pool(family):
    """Every candidate model for a family, fitted once per process."""
    X_train, X_test, y_train, y_test = data.split()
    entries = []
    for value in grid(family):
        pipeline = Pipeline([("prepare", _preprocessor(family)), ("model", _estimator(family, value))])
        pipeline.fit(X_train, y_train)
        entries.append(
            {
                "value": value,
                "pipeline": pipeline,
                "accuracy": float(accuracy_score(y_test, pipeline.predict(X_test))),
                "complexity": complexity(family, pipeline),
            }
        )
    return entries


def objective(entry, lam):
    return entry["accuracy"] - lam * entry["complexity"]


def select(family, lam):
    """The maximiser of acc_test - lambda * Omega(f), preferring the simpler model on ties."""
    entries = sorted(pool(family), key=lambda entry: entry["complexity"])
    return max(entries, key=lambda entry: objective(entry, lam))
