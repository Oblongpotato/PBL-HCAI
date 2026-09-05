from django.shortcuts import render

from . import data, plots, training


def _selection(request):
    """Read the linked state out of the query string, falling back to sane defaults."""
    family = request.GET.get("family", "tree")
    if family not in training.FAMILIES:
        family = "tree"

    try:
        lam = float(request.GET.get("lam", 0.0))
    except (TypeError, ValueError):
        lam = 0.0
    lam = min(max(lam, 0.0), training.LAMBDA_MAX)

    return family, round(lam, 4)


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
        "lambda_max": training.LAMBDA_MAX,
        "selected": selected,
        "complexity_label": training.COMPLEXITY_LABEL[family],
        "fit_regulariser": training.FIT_REGULARISER[family],
        "objective": training.objective(selected, lam),
    }

    if family == "tree":
        context["model_plot"] = plots.decision_tree(selected["pipeline"])
    else:
        context["model_plot"] = plots.coefficients(selected["pipeline"])
        context["used_features"] = training.used_features(selected["pipeline"])

    return render(request, "project2/index.html", context)
