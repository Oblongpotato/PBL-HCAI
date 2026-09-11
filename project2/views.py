import math
from functools import lru_cache

from django.shortcuts import render

from . import counterfactuals, data, effects, plots, training

PAGE_TITLE = "Project 2: Explainability"


def _selection(request):
    """Read the linked state out of the query string, falling back to sane defaults."""
    family = request.GET.get("family", "tree")
    if family not in training.FAMILIES:
        family = "tree"

    try:
        lam = float(request.GET.get("lam", 0.0))
    except (TypeError, ValueError):
        lam = 0.0
    # nan survives both comparisons below, so it has to be caught before them.
    if not math.isfinite(lam):
        lam = 0.0
    lam = min(max(lam, 0.0), training.lambda_max(family))

    return family, round(lam, 4)


def _pipeline(family, value):
    return next(entry for entry in training.pool(family) if entry["value"] == value)["pipeline"]


@lru_cache(maxsize=64)
def _counterfactuals_for(family, value, row, target):
    """Counterfactual rows for one fully specified state.

    The search is the slowest thing on the page and the same state is re-requested every
    time the user moves any other control, so the result is kept rather than recomputed.
    """
    _, x = data.example(row)
    found = counterfactuals.generate(_pipeline(family, value), x, target)
    return counterfactuals.as_rows(x, found)


@lru_cache(maxsize=64)
def _effect_curves(family, value, feature):
    """PDP and both ALE estimates, cached for the same reason. Figures are drawn fresh."""
    pipeline = _pipeline(family, value)
    grid = effects.grid_for(feature)
    edges, finite = effects.ale_finite(pipeline, feature)
    _, exact = effects.ale_exact(pipeline, feature)
    return grid, effects.pdp(pipeline, feature, grid), edges, finite, exact


def _counterfactual_context(request, family, selected):
    """Section 4, driven by the same model the sections above are showing."""
    frame = data.load()
    try:
        row = int(request.GET.get("row", 0))
    except (TypeError, ValueError):
        row = 0
    index, x = data.example(row)

    predicted = selected["pipeline"].predict(frame.iloc[[index]][data.FEATURES])[0]
    target = request.GET.get("target")
    if target not in data.species():
        target = next(name for name in data.species() if name != predicted)

    return {
        "row": index,
        "row_choices": data.row_choices(),
        "species": data.species(),
        "features": data.FEATURES,
        "target": target,
        "predicted": predicted,
        "actual": x[data.TARGET],
        "original_cells": [x[feature] for feature in data.FEATURES],
        "counterfactuals": _counterfactuals_for(family, selected["value"], index, target),
    }


def _effects_context(request, family, selected):
    """Section 5, on the same model again."""
    feature = request.GET.get("feature")
    if feature not in data.NUMERIC_FEATURES:
        feature = data.NUMERIC_FEATURES[0]

    grid, pdp, edges, finite, exact = _effect_curves(family, selected["value"], feature)

    return {
        "feature": feature,
        "pdp_plot": plots.effect_curves(grid, pdp, feature, f"Partial dependence on {feature}"),
        "ale_plot": plots.effect_curves(
            edges, finite, feature, f"Accumulated local effects of {feature}", extra=exact
        ),
        "exact_available": exact is not None,
    }


def index(request):
    """One page: the model selected here drives everything shown below it."""
    family, lam = _selection(request)
    selected = training.select(family, lam)

    context = {
        "page_title": PAGE_TITLE,
        "summary": data.summary(),
        "family": family,
        "families": training.FAMILIES.items(),
        "family_label": training.FAMILIES[family],
        "lam": lam,
        "lambda_max": training.lambda_max(family),
        "selected": selected,
        "complexity_label": training.COMPLEXITY_LABEL[family],
        "fit_regulariser": training.FIT_REGULARISER[family],
        "objective": training.objective(selected, lam),
        "tradeoff_plot": plots.tradeoff(family, lam, selected),
        "candidates": [
            {
                "value": entry["value"],
                "complexity": entry["complexity"],
                "accuracy": entry["accuracy"],
                "objective": training.objective(entry, lam),
                "chosen": entry["value"] == selected["value"],
            }
            for entry in sorted(training.pool(family), key=lambda e: e["complexity"])
        ],
    }

    context.update(_counterfactual_context(request, family, selected))
    context.update(_effects_context(request, family, selected))

    if family == "tree":
        context["model_plot"] = plots.decision_tree(selected["pipeline"])
    else:
        context["used_features"] = training.used_features(selected["pipeline"])
        # With no surviving coefficients there is nothing to draw; the template says so.
        if selected["complexity"]:
            context["model_plot"] = plots.coefficients(selected["pipeline"])

    return render(request, "project2/index.html", context)
