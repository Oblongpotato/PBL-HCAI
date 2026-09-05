import uuid

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.db import models

from utils import datasets

# Uploads live outside MEDIA_ROOT: that directory is served publicly by pbl/urls.py,
# while a dataset belongs to the session that uploaded it.
private_storage = FileSystemStorage(location=settings.BASE_DIR / "private_uploads")


def upload_path(instance, filename):
    """Discard the caller's filename; it is displayed from Dataset.name instead."""
    return f"{uuid.uuid4().hex}.csv"

TASK_CHOICES = [("classification", "Classification"), ("regression", "Regression")]


class Dataset(models.Model):
    """A CSV uploaded by the user, kept so visualisation and training can reuse it."""

    name = models.CharField(max_length=200)
    file = models.FileField(upload_to=upload_path, storage=private_storage)
    target = models.CharField(max_length=200)
    task = models.CharField(max_length=20, choices=TASK_CHOICES)
    n_rows = models.PositiveIntegerField()
    n_features = models.PositiveIntegerField()
    dropped_columns = models.JSONField(default=list)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.name} ({self.task})"

    def load(self):
        """Return the cleaned dataframe backing this dataset."""
        frame, _ = datasets.drop_id_columns(datasets.read_csv(self.file.path))
        return frame


class TrainingRun(models.Model):
    """One hyperparameter sweep, kept as an audit trail of what the user tried."""

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="runs")
    model_key = models.CharField(max_length=50)
    hyperparameter = models.CharField(max_length=50)
    values = models.JSONField()
    scores = models.JSONField()
    test_size = models.FloatField()
    random_state = models.IntegerField()
    scoring = models.CharField(max_length=30)
    best_value = models.FloatField()
    best_score = models.FloatField()
    automated = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.model_key} @ {self.hyperparameter}={self.best_value}"
