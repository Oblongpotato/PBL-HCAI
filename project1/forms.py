from django import forms

from . import ml, plots
from .models import TASK_CHOICES


class DatasetUploadForm(forms.Form):
    """Upload a CSV whose first row holds feature names and last column the target."""

    file = forms.FileField(label="CSV file")
    task = forms.ChoiceField(
        label="Problem type",
        required=False,
        choices=[("", "Detect automatically")] + TASK_CHOICES,
        help_text="The app detects the task from the target column; override it if you disagree.",
    )


class VisualizationForm(forms.Form):
    """Pick what to look at. Choices depend on the dataset that is currently loaded."""

    kind = forms.ChoiceField(label="Plot", choices=plots.KINDS)
    x = forms.ChoiceField(label="Feature")
    y = forms.ChoiceField(label="Second feature", required=False)

    def __init__(self, features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        choices = [(feature, feature) for feature in features]
        self.fields["x"].choices = choices
        self.fields["y"].choices = choices

    def clean(self):
        cleaned = super().clean()
        if cleaned.get("kind") == "pair" and not cleaned.get("y"):
            self.add_error("y", "Pick a second feature for this plot.")
        return cleaned


class TrainingForm(forms.Form):
    """The choices the lecture leaves to the human: model, range, split and score."""

    model = forms.ChoiceField(label="Model")
    values = forms.CharField(
        label="Hyperparameter values",
        required=False,
        help_text="Comma-separated. Empty means the model's default range.",
    )
    test_size = forms.FloatField(label="Test fraction", min_value=0.1, max_value=0.5, initial=0.3)
    random_state = forms.IntegerField(label="Random seed", initial=42)
    scoring = forms.ChoiceField(label="Score")

    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model"].choices = [(key, spec["label"]) for key, spec in ml.models_for(task).items()]
        self.fields["scoring"].choices = [(key, spec["label"]) for key, spec in ml.scores_for(task).items()]

    def clean(self):
        cleaned = super().clean()
        key = cleaned.get("model")
        if not key:
            return cleaned

        spec = ml.MODELS[key]
        raw = (cleaned.get("values") or "").strip()
        if not raw:
            cleaned["grid"] = spec["grid"]
            return cleaned

        try:
            grid = [float(value) for value in raw.replace(";", ",").split(",") if value.strip()]
        except ValueError:
            self.add_error("values", "Use numbers separated by commas.")
            return cleaned

        if spec["integer"]:
            grid = [int(value) for value in grid]
        if not grid or any(value <= 0 for value in grid):
            self.add_error("values", f"{spec['hyperparameter']} must be a list of positive numbers.")
        elif len(grid) > 20:
            self.add_error("values", "Twenty values at most, to keep training responsive.")
        else:
            cleaned["grid"] = sorted(dict.fromkeys(grid))
        return cleaned
