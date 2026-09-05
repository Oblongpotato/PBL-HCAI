from django.shortcuts import redirect, render

from utils import datasets

from .forms import DatasetUploadForm
from .models import Dataset

SESSION_KEY = "project1_dataset"


def _current_dataset(request):
    """The dataset the user is working on, or None."""
    return Dataset.objects.filter(pk=request.session.get(SESSION_KEY)).first()


def _store(request, upload, task_override):
    """Validate, describe and persist an uploaded CSV. Returns (dataset, summary)."""
    frame = datasets.read_csv(upload)
    frame, dropped = datasets.drop_id_columns(frame)
    summary = datasets.describe(frame, dropped)

    upload.seek(0)
    dataset = Dataset.objects.create(
        name=upload.name,
        file=upload,
        target=summary["target"],
        task=task_override or summary["task"],
        n_rows=summary["n_rows"],
        n_features=summary["n_features"],
        dropped_columns=dropped,
    )
    request.session[SESSION_KEY] = dataset.pk
    return dataset


def _context(dataset, **extra):
    """Shared page context: everything the interface needs about the active dataset."""
    context = {"dataset": dataset, "upload_form": DatasetUploadForm()}
    if dataset:
        frame = dataset.load()
        summary = datasets.describe(frame, dataset.dropped_columns)
        summary["task"] = dataset.task
        context["summary"] = summary
        context["warnings"] = datasets.quality_warnings(frame, dataset.task)
    context.update(extra)
    return context


def index(request):
    """Upload a dataset and inspect what the app understood about it."""
    if request.method != "POST":
        return render(request, "project1/index.html", _context(_current_dataset(request)))

    form = DatasetUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(
            request,
            "project1/index.html",
            _context(_current_dataset(request), upload_form=form),
        )

    try:
        _store(request, request.FILES["file"], form.cleaned_data["task"])
    except ValueError as error:
        form.add_error("file", str(error))
        return render(
            request,
            "project1/index.html",
            _context(_current_dataset(request), upload_form=form),
        )

    return redirect("project1:index")
