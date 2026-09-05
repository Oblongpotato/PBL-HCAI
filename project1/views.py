from django.shortcuts import redirect, render

from utils import datasets

from . import plots
from .forms import DatasetUploadForm, VisualizationForm
from .models import Dataset

SESSION_KEY = "project1_dataset"


def _current_dataset(request):
    """The dataset the user is working on, or None."""
    return Dataset.objects.filter(pk=request.session.get(SESSION_KEY)).first()


def _store(request, upload, task_override):
    """Validate, describe and persist an uploaded CSV."""
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


def _context(dataset, frame=None, **extra):
    """Page context: everything the interface shows about the active dataset."""
    context = {"dataset": dataset, "upload_form": DatasetUploadForm()}
    if dataset:
        frame = dataset.load() if frame is None else frame
        summary = datasets.describe(frame, dataset.dropped_columns)
        summary["task"] = dataset.task
        context["summary"] = summary
        context["warnings"] = datasets.quality_warnings(frame, dataset.task)
        context.setdefault("visualization_form", VisualizationForm(summary["numeric_features"]))
    context.update(extra)
    return context


def index(request):
    """Upload a dataset and inspect what the app understood about it."""
    if request.method != "POST":
        return render(request, "project1/index.html", _context(_current_dataset(request)))

    form = DatasetUploadForm(request.POST, request.FILES)
    if form.is_valid():
        try:
            _store(request, request.FILES["file"], form.cleaned_data["task"])
            return redirect("project1:index")
        except ValueError as error:
            form.add_error("file", str(error))

    return render(request, "project1/index.html", _context(_current_dataset(request), upload_form=form))


def visualize(request):
    """Render one plot of the loaded dataset, chosen by the user."""
    dataset = _current_dataset(request)
    if dataset is None or request.method != "POST":
        return redirect("project1:index")

    frame = dataset.load()
    numeric = datasets.describe(frame, dataset.dropped_columns)["numeric_features"]
    form = VisualizationForm(numeric, request.POST)

    plot = None
    if form.is_valid():
        url, caption = plots.build(
            frame,
            dataset.target,
            dataset.task,
            form.cleaned_data["kind"],
            form.cleaned_data["x"],
            form.cleaned_data["y"],
        )
        plot = {"url": url, "caption": caption}

    return render(
        request,
        "project1/index.html",
        _context(dataset, frame=frame, visualization_form=form, plot=plot),
    )
