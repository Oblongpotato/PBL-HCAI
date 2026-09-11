"""Feature effect curves, computed here rather than taken from a library.

PDP marginalises the other features; ALE accumulates *local* differences and so stays
honest when features are correlated, which on this dataset they very much are.

The lecture asks for which model the partial derivative can be computed exactly. That is
answered by ``ale_exact``: a multinomial logistic model has a closed-form derivative,
whereas a decision tree is piecewise constant, its derivative zero almost everywhere with
jumps at the split thresholds, so only the discretised estimate exists for it.
"""

import numpy as np

from . import data

GRID_POINTS = 40
INTERVALS = 20


def _probabilities(pipeline, frame):
    """predict_proba as a dict keyed by species."""
    predicted = pipeline.predict_proba(frame)
    return {name: predicted[:, index] for index, name in enumerate(pipeline.classes_)}


def grid_for(feature, points=GRID_POINTS):
    column = data.load()[feature]
    return np.linspace(column.quantile(0.02), column.quantile(0.98), points)


def pdp(pipeline, feature, grid):
    """PDP_c(v) = mean over the dataset of P(class = c | x_j := v)."""
    frame = data.load()[data.FEATURES]
    curves = {name: [] for name in pipeline.classes_}

    for value in grid:
        replaced = frame.copy()
        replaced[feature] = value
        for name, probability in _probabilities(pipeline, replaced).items():
            curves[name].append(float(probability.mean()))
    return {name: np.asarray(values) for name, values in curves.items()}


def _intervals(feature, count=INTERVALS):
    column = data.load()[feature]
    edges = np.unique(np.quantile(column, np.linspace(0, 1, count + 1)))
    return edges


def ale_finite(pipeline, feature):
    """Accumulated local effects by finite differences inside each quantile interval.

    Within interval k, every point is evaluated at both edges and the differences are
    averaged; accumulating them and centring gives the ALE curve. This works for any
    model, including one with no usable derivative.
    """
    frame = data.load()[data.FEATURES]
    column = frame[feature].to_numpy()
    edges = _intervals(feature)

    curves = {name: np.zeros(len(edges)) for name in pipeline.classes_}
    for k in range(1, len(edges)):
        inside = (column > edges[k - 1]) & (column <= edges[k])
        if k == 1:
            inside |= column <= edges[0]
        if not inside.any():
            for name in curves:
                curves[name][k] = curves[name][k - 1]
            continue

        lower, upper = frame[inside].copy(), frame[inside].copy()
        lower[feature] = edges[k - 1]
        upper[feature] = edges[k]

        low, high = _probabilities(pipeline, lower), _probabilities(pipeline, upper)
        for name in curves:
            curves[name][k] = curves[name][k - 1] + float((high[name] - low[name]).mean())

    return edges, _centre(edges, curves, column)


def ale_exact(pipeline, feature):
    """ALE from the closed-form derivative of the multinomial logistic model.

    For softmax probabilities, dP_c/dz_j = P_c (w_cj - sum_k P_k w_kj). The model is fitted
    on standardised features, so the chain rule divides by the scaler's scale for j.
    Returns None for models without a usable derivative.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None, None

    prepare = pipeline.named_steps["prepare"]
    scaler = prepare.named_transformers_["numeric"]
    if not hasattr(scaler, "scale_"):
        return None, None

    position = data.NUMERIC_FEATURES.index(feature)
    scale = float(scaler.scale_[position])
    encoded = list(prepare.get_feature_names_out())
    column_index = encoded.index(f"numeric__{feature}")
    weights = model.coef_[:, column_index] / scale

    frame = data.load()[data.FEATURES]
    column = frame[feature].to_numpy()
    edges = _intervals(feature)

    curves = {name: np.zeros(len(edges)) for name in model.classes_}
    for k in range(1, len(edges)):
        inside = (column > edges[k - 1]) & (column <= edges[k])
        if k == 1:
            inside |= column <= edges[0]
        if not inside.any():
            for name in curves:
                curves[name][k] = curves[name][k - 1]
            continue

        midpoint = frame[inside].copy()
        midpoint[feature] = (edges[k - 1] + edges[k]) / 2
        probability = _probabilities(pipeline, midpoint)

        stacked = np.vstack([probability[name] for name in model.classes_])
        shared = (stacked * weights[:, None]).sum(axis=0)
        width = edges[k] - edges[k - 1]
        for index, name in enumerate(model.classes_):
            derivative = probability[name] * (weights[index] - shared)
            curves[name][k] = curves[name][k - 1] + float(derivative.mean()) * width

    return edges, _centre(edges, curves, column)


def _centre(edges, curves, column):
    """Subtract the data-weighted mean so the curve reads as a deviation from average."""
    positions = np.clip(np.searchsorted(edges, column, side="left"), 0, len(edges) - 1)
    return {
        name: values - float(values[positions].mean())
        for name, values in curves.items()
    }
