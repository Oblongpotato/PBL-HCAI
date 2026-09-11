# Human-Centric Artificial Intelligence — project work

Coursework for the HCAI module. One Django project holding four apps, one per project, all
reachable from a hub page at `/home/`.

Group 39 — Sahil Sajwan, 674409

## The four projects

| | Route | What it does |
|---|---|---|
| 1 | `/project1/` | Upload a CSV, look at it, then train a model on it. You choose the problem type, the model, the hyperparameter range, the split and the score; the app handles encoding, scaling, fitting and reports every decision it made. A one-click AutoML run is there to compare against. |
| 2 | `/project2/` | Palmer Penguins. A slider trades accuracy against model complexity, and everything below it — the tree or the weights, the counterfactuals, the PDP and ALE curves — describes whichever model that choice lands on. The effect curves are written by hand rather than taken from a library. |
| 3 | `/project3/` | AG News. A topic classifier that can hand an article to a simulated human expert instead of answering, and active learning to find out what that expert is good at without being told. Results and a PDF report. |
| 4 | `/project4/` | A user study comparing two ways of asking someone what films they like: pairwise choice against ranking ten. The participant interface is implemented and the design is written up as a PDF. The study itself is not run. |

## Running it

Built with Python 3.12.

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python manage.py migrate
    python manage.py runserver

On macOS or Linux the second line is `source .venv/bin/activate`.

Then open <http://127.0.0.1:8000/>.

Settings are read from a `.env` file if one exists, and `.env.example` lists the keys. It is
optional: the defaults are fine for running locally.

## Tests

    python manage.py test

## Regenerating the project 3 results

Project 3 trains nothing while a page is loading. The experiments run once, offline, and the
results are committed so the view only has to read them:

    python manage.py run_project3

That rewrites `project3/results/results.json` and the figures under
`project3/static/project3/figures/`, and takes a couple of minutes. The page prints a warning
if the committed results were produced by different code than is checked out.

## Housekeeping

Uploaded datasets pile up under `private_uploads/`. To clear out the old ones:

    python manage.py prune_datasets --days 7 --dry-run

Drop `--dry-run` to actually delete.

## Where the lecture material shows up

`docs/README.md` goes through the projects one at a time and records which lecture each idea
came from and where it ended up in the code.
