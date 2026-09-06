from django.shortcuts import render

from . import counterfactuals, data, effects, plots, training


def _selection(request):
    """Read the linked state out of the query string, falling back to sane defaults."""
    family = request.GET.get("family", "tree")
    if family not in training.FAMILIES:
        family = "tree"

    try:
        lam = float(request.GET.get("lam", 0.0))
    except (TypeError, ValueError):
        lam = 0.0
    lam = min(max(lam, 0.0), training.lambda_max(family))

    return family, round(lam, 4)


def _counterfactual_context(request, pipeline):
    """Section 4, driven by the same model the sections above are showing."""
    frame = data.load()
    try:
        row = int(request.GET.get("row", 0))
    except (TypeError, ValueError):
        row = 0
    index, x = data.example(row)

    predicted = pipeline.predict(frame.iloc[[index]][data.FEATURES])[0]
    target = request.GET.get("target")
    if target not in data.species():
        target = next(name for name in data.species() if name != predicted)

    found = counterfactuals.generate(pipeline, x, target)
    return {
        "row": index,
        "species": data.species(),
        "features": data.FEATURES,
        "target": target,
        "predicted": predicted,
        "original_cells": [x[feature] for feature in data.FEATURES],
        "counterfactuals": counterfactuals.as_rows(x, found),
    }


def _effects_context(request, pipeline):
    """Section 5, on the same model again."""
    feature = request.GET.get("feature")
    if feature not in data.NUMERIC_FEATURES:
        feature = data.NUMERIC_FEATURES[0]

    grid = effects.grid_for(feature)
    pdp_plot = plots.effect_curves(
        grid, effects.pdp(pipeline, feature, grid), feature,
        f"Partial dependence on {feature}",
    )

    edges, finite = effects.ale_finite(pipeline, feature)
    exact_edges, exact = effects.ale_exact(pipeline, feature)
    ale_plot = plots.effect_curves(
        edges, finite, feature, f"Accumulated local effects of {feature}", extra=exact,
    )

    return {
        "feature": feature,
        "pdp_plot": pdp_plot,
        "ale_plot": ale_plot,
        "exact_available": exact is not None,
    }


def index(request):
    """One page: the model selected here drives everything shown below it."""
    family, lam = _selection(request)
    selected = training.select(family, lam)

    context = {
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

    context.update(_counterfactual_context(request, selected["pipeline"]))
    context.update(_effects_context(request, selected["pipeline"]))

    if family == "tree":
        context["model_plot"] = plots.decision_tree(selected["pipeline"])
    else:
        context["model_plot"] = plots.coefficients(selected["pipeline"])
        context["used_features"] = training.used_features(selected["pipeline"])

    return render(request, "project2/index.html", context)
