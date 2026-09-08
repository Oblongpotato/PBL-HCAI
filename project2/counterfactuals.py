"""Counterfactual explanations by local sampling, following lecture 3.

Sample points around x, keep the ones the model assigns to the desired class, and rank
them by MAD-weighted L1 distance so that "close" means close in units the data itself
defines, not in raw millimetres versus grams.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data

SIGMAS = (0.5, 1.0, 2.0, 4.0)
START_N = 2000
MAX_N = 16000
FLIP_PROBABILITY = 0.25
TOP_K = 5


@lru_cache(maxsize=1)
def mad():
    """Median absolute deviation per numeric feature, guarded against zero."""
    frame = data.load()
    deviations = {}
    for feature in data.NUMERIC_FEATURES:
        column = frame[feature]
        value = float((column - column.median()).abs().median())
        if value <= 0:
            value = float(column.std()) or 1.0
        deviations[feature] = value
    return deviations


@lru_cache(maxsize=1)
def _categories():
    frame = data.load()
    return {feature: sorted(frame[feature].unique()) for feature in data.CATEGORICAL_FEATURES}


def distance(x, candidates):
    """MAD-weighted L1 over numeric features, plus one unit per differing category."""
    deviations = mad()
    total = np.zeros(len(candidates))
    for feature in data.NUMERIC_FEATURES:
        total += (candidates[feature] - x[feature]).abs().to_numpy() / deviations[feature]
    for feature in data.CATEGORICAL_FEATURES:
        total += (candidates[feature] != x[feature]).to_numpy().astype(float)
    return total


def sample_around(x, size, sigma, rng):
    """Noise numeric features on their own scale; resample categories occasionally.

    Categorical and binary features cannot be nudged, so they are redrawn from the values
    the dataset actually contains, and only for a minority of the samples.
    """
    deviations = mad()
    columns = {}
    for feature in data.NUMERIC_FEATURES:
        columns[feature] = x[feature] + rng.normal(0, sigma * deviations[feature], size)

    for feature, options in _categories().items():
        drawn = rng.choice(options, size=size)
        keep = rng.random(size) > FLIP_PROBABILITY
        columns[feature] = np.where(keep, x[feature], drawn)

    return pd.DataFrame(columns)[data.FEATURES]


def generate(pipeline, x, target, k=TOP_K, seed=0):
    """The closest k samples the model assigns to ``target``.

    Widens the search until something is found or the budget runs out, as the brief asks:
    a target class may simply not be reachable near this penguin.
    """
    rng = np.random.default_rng(seed)
    found = []

    for sigma in SIGMAS:
        size = START_N
        while size <= MAX_N:
            candidates = sample_around(x, size, sigma, rng)
            hits = candidates[pipeline.predict(candidates) == target]
            if not hits.empty:
                hits = hits.copy()
                hits["distance"] = distance(x, hits)
                found.append(hits)
                break
            size *= 2

        if found and sum(len(frame) for frame in found) >= k:
            break

    if not found:
        return pd.DataFrame(columns=list(data.FEATURES) + ["distance"])

    return (
        pd.concat(found, ignore_index=True)
        .sort_values("distance")
        .drop_duplicates(subset=data.FEATURES)
        .head(k)
        .reset_index(drop=True)
    )


def as_rows(x, counterfactuals):
    """Render for the template: each cell knows whether it changed and by how much."""
    rows = []
    for _, candidate in counterfactuals.iterrows():
        cells = []
        for feature in data.FEATURES:
            original, changed = x[feature], candidate[feature]
            if feature in data.NUMERIC_FEATURES:
                delta = float(changed) - float(original)
                cells.append(
                    {
                        "value": f"{float(changed):.1f}",
                        "changed": abs(delta) > 1e-9,
                        "delta": f"{delta:+.1f}",
                    }
                )
            else:
                cells.append({"value": changed, "changed": changed != original, "delta": ""})
        rows.append({"cells": cells, "distance": candidate["distance"]})
    return rows
