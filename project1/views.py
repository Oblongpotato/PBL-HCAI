from django.shortcuts import redirect, render

from utils import datasets

from . import ml, plots
from .forms import DatasetUploadForm, TrainingForm, VisualizationForm
from .models import Dataset, TrainingRun

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
        context.setdefault("training_form", TrainingForm(dataset.task))
        context["model_specs"] = ml.models_for(dataset.task).values()
        context["history"] = dataset.runs.all()[:10]
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


def train(request):
    """Run the lecture's pipeline with the settings the user chose."""
    dataset = _current_dataset(request)
    if dataset is None or request.method != "POST":
        return redirect("project1:index")

    frame = dataset.load()
    form = TrainingForm(dataset.task, request.POST)
    if not form.is_valid():
        return render(request, "project1/index.html", _context(dataset, frame=frame, training_form=form))

    choice = form.cleaned_data
    result = _sweep(dataset, frame, choice["model"], choice["grid"], choice["test_size"],
                    choice["random_state"], choice["scoring"], automated=False)
    return render(
        request,
        "project1/index.html",
        _context(dataset, frame=frame, training_form=form, result=result),
    )


def _sweep(dataset, frame, model_key, values, test_size, random_state, scoring, automated):
    """Train, record the run and render the score curve."""
    result = ml.run_sweep(frame, dataset.target, dataset.task, model_key, values,
                          test_size, random_state, scoring)
    TrainingRun.objects.create(
        dataset=dataset,
        model_key=model_key,
        hyperparameter=result["hyperparameter"],
        values=result["values"],
        scores=result["scores"],
        test_size=test_size,
        random_state=random_state,
        scoring=scoring,
        best_value=result["best_value"],
        best_score=result["best_score"],
        automated=automated,
    )
    result["curve"] = plots.score_curve(result["values"], result["scores"],
                                        result["hyperparameter"], result["score_label"],
                                        result["best_value"])
    result["automated"] = automated
    return result


def automl(request):
    """Run the same pipeline with no user input, to expose what automation costs."""
    dataset = _current_dataset(request)
    if dataset is None or request.method != "POST":
        return redirect("project1:index")

    frame = dataset.load()
    choice = ml.automatic_choice(dataset.task)
    result = _sweep(dataset, frame, choice["model_key"], choice["values"], choice["test_size"],
                    choice["random_state"], choice["scoring"], automated=True)
    return render(request, "project1/index.html", _context(dataset, frame=frame, result=result))
