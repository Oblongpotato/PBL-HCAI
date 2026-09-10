from io import BytesIO

from django.http import FileResponse
from django.shortcuts import redirect, render

from . import experiments, report

PAGE_TITLE = "Project 3 — Learning to Defer"


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
            "page_title": PAGE_TITLE,
            "results": results,
            "stale": bool(results) and experiments.is_stale(results),
        },
    )


def report_pdf(request):
    """The report the brief asks for, built from the committed results on each request."""
    try:
        body = report.build()
    except FileNotFoundError:
        # Nothing to report on yet. The index says what to run, so send the reader there
        # rather than to a stack trace.
        return redirect("project3:index")

    return FileResponse(
        BytesIO(body),
        as_attachment=True,
        filename="project3-report.pdf",
        content_type="application/pdf",
    )
