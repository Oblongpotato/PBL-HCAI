# HCAI project

Group 39, Sahil Sajwan (674409).

Django project for the Human-Centric Artificial Intelligence course. Four apps,
one per project, all linked from the home page.

## Setup

Python 3.12.

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python manage.py migrate

On Linux or macOS the activate line is `source .venv/bin/activate`.

## Running it

    python manage.py runserver

Then open http://127.0.0.1:8000/ and pick a project from the home page.

Settings are read from a `.env` file if one exists. Copy `.env.example` to
`.env` if you want to change them. It is not needed for running locally.

## Tests

    python manage.py test

For a single app:

    python manage.py test project2

## Project 3 results

Project 3 does not train anything while a page loads. The experiments are run
once and the output is committed, so the page only reads it. To run them again:

    python manage.py run_project3

This takes a few minutes. It rewrites `project3/results/results.json` and the
figures in `project3/static/project3/figures/`. If the committed results were
produced by a different version of the code, the page says so at the top.

## Clearing old uploads

Project 1 stores every uploaded CSV in `private_uploads/`. To delete old ones:

    python manage.py prune_datasets --days 7

Add `--dry-run` to list them without deleting.

## Layout

    home/       home page with the project links
    project1/   supervised learning interface
    project2/   explainability
    project3/   active learning for learning-to-defer
    project4/   preference elicitation study
    utils/      shared helpers for CSV loading, plots and PDF reports
    demos/      reference app from the course skeleton, left unchanged
    docs/       notes on which lecture material went where
