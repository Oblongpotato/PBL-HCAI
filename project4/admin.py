from django.contrib import admin

from .models import ElicitationTask, Participant, Response


@admin.register(Participant)
class ParticipantAdmin(admin.ModelAdmin):
    list_display = ("token", "first_condition", "started_at", "consented_at", "finished_at")
    list_filter = ("first_condition",)


@admin.register(ElicitationTask)
class ElicitationTaskAdmin(admin.ModelAdmin):
    list_display = ("participant", "condition", "block", "position", "presented_at")
    list_filter = ("condition",)


@admin.register(Response)
class ResponseAdmin(admin.ModelAdmin):
    list_display = ("task", "duration_ms", "submitted_at")
