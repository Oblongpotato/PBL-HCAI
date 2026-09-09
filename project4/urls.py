from django.urls import path

from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("report/", views.report_pdf, name="report"),
    path("consent/", views.consent, name="consent"),
    path("task/", views.task, name="task"),
    path("results/", views.results, name="results"),
]
