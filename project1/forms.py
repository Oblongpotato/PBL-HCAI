from django import forms

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
