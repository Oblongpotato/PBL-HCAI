from django import forms

from . import plots
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
