from django.shortcuts import render

from . import experiments


def index(request):
    """Display the precomputed experiment results.

    Nothing is trained here. `manage.py run_project3` produces results.json and the figures,
    both of which are committed, so this view only reads them.
    """
    results = experiments.load()
    return render(
        request,
        "project3/index.html",
        {
            "results": results,
            "stale": bool(results) and experiments.is_stale(results),
        },
    )
