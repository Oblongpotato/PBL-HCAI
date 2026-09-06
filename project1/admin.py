from django.contrib import admin

from .models import Dataset, TrainingRun


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "task", "target", "n_rows", "n_features", "uploaded_at")


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("dataset", "model_key", "hyperparameter", "best_value", "best_score", "automated", "created_at")
    list_filter = ("automated", "model_key")
