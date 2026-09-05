from django.shortcuts import render

# Group members shown on the home page (task 1: defined in python, not in the template).
STUDENTS = [
    {"name": "Sahil Sajwan", "matriculation": "674409"},
]

PROJECTS = [
    {"name": "Project 1 — Supervised Learning Interface", "url_name": "project1:index"},
    {"name": "Project 2 — Explainability", "url_name": "project2:index"},
    {"name": "Project 3 — Active Learning for Learning-to-Defer", "url_name": "project3:index"},
    {"name": "Project 4 — Preference Elicitation", "url_name": "project4:index"},
]


def index(request):
    return render(request, "home/index.html", {"students": STUDENTS, "projects": PROJECTS})
