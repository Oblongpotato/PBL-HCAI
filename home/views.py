from django.shortcuts import render

# Group identity shown on the home page (task 1: defined in python, not in the template).
GROUP = "39"

STUDENTS = [
    {"name": "Sahil Sajwan", "matriculation": "674409"},
]

PROJECTS = [
    {
        "name": "Supervised Learning Interface",
        "url_name": "project1:index",
        "note": "Upload a dataset, look at it, and train a model on your own terms.",
    },
    {
        "name": "Explainability",
        "url_name": "project2:index",
        "note": "Trade accuracy against complexity, then read the model that comes out.",
    },
    {
        "name": "Active Learning for Learning-to-Defer",
        "url_name": "project3:index",
        "note": "When a classifier should hand an article to a human expert instead.",
    },
    {
        "name": "Preference Elicitation",
        "url_name": "project4:index",
        "note": "A user study comparing two ways of asking what someone likes.",
    },
]


def index(request):
    return render(
        request,
        "home/index.html",
        {
            "group": GROUP,
            "students": STUDENTS,
            "projects": PROJECTS,
            "page_title": "Project hub",
        },
    )
