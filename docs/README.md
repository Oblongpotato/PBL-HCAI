# HCAI PBL — lecture content applied per project

One Django project, four apps (`project1`–`project4`), reachable from the home page at
`/home/`. This document records, project by project, **which lecture material was applied
and how it shows up in the code**. It is written after a feature branch has been stress
tested and reviewed.

Shared surface used by all projects:

| Path | Role |
|---|---|
| `pbl/settings.py`, `pbl/urls.py` | app registration and routing (all four apps registered up front) |
| `home/views.py` | group members and the project links, defined in Python (project 1, task 1) |
| `templates/base.html`, `static/style.css` | global layout and style |
| `static/<app>/`, `templates/<app>/` | mandated per-app paths for CSS and templates |
| `utils/datasets.py`, `utils/plotting.py` | project-agnostic CSV loading / inspection and matplotlib→media rendering |
| `demos/` | instructor's reference app — read, never modified |

---

## Project 1 — Supervised Learning Interface

**App:** `project1/` · **Route:** `/project1/` · **Dataset:** any CSV (first row = feature
names, last column = target); developed against `Resources/iris.csv`.

The page is a single scrolling workflow with five numbered blocks — *Data → What the app
understood → Visualise → Train → Automation* — which is deliberately the shape of the ML
pipeline the course opens with.

### Lecture 1 — Introduction: the ML pipeline and AutoML

Lecture 1's pipeline (choose a hypothesis class → split → fit for several hyperparameter
values → score) is the literal control flow of `project1/ml.py:run_sweep()`:

- **Hypothesis class.** `ml.MODELS` is a registry of eight estimators, four per task, each
  entry naming its single exposed hyperparameter, a default grid, and a one-line trade-off
  note that is surfaced to the user in the "Train a model" table. Ridge regression is
  included because it is the lecture's worked example of a linear model with quadratic
  penalisation.
- **Train/test split and penalisation.** `train_test_split` with a user-chosen test
  fraction and seed; stratification is enabled automatically for classification whenever
  every class has at least two rows.
- **Sweep over the hyperparameter.** One `Pipeline` is fit per grid value, and
  `plots.score_curve()` draws score-against-hyperparameter — the curve the lecture uses to
  argue that a hyperparameter is chosen by evaluation, not by fitting.
- **Score selection.** `ml.SCORES` carries a `maximise` flag so that error scores (MSE,
  MAE) select the *smallest* value while accuracy-type scores select the largest.

The lecture's **AutoML pros/cons** slide ("the user is not in control", "does not exploit
domain expertise") is turned into an interactive argument rather than a paragraph:
`ml.automatic_choice()` fixes model, grid, split and score with no user input, `views.automl`
runs it, and the run history table labels every row **"decided by: you / the app"** so the
two regimes can be compared side by side on the same data. This is section 5 of the page.

Lecture 1's point that **a human is hiding behind every step** drives
`utils/datasets.quality_warnings()` — class imbalance, missing values, tiny sample size,
non-numeric columns and rare classes are reported *before* any score is shown, instead of
being silently handled.

### Lecture 8 — Interactive ML: what a good IML interface contains

Lecture 8 lists four elements of a good interactive-ML interface. Each one is a block on
the page, in that order:

| Lecture 8 element | Where |
|---|---|
| Data visualisation | section 3, `project1/plots.py` — four plot kinds: feature pair coloured by target, feature against target, per-class distribution, correlation matrix |
| Pre-processing | section 2 — detected task, dropped identifier columns, data-quality warnings, 10-row preview; scaling and one-hot encoding are automated in `ml._preprocessor()` and declared to the user |
| Model selection and tuning | section 4 — model, hyperparameter range, test fraction, seed and score are all user-controlled form fields (`project1/forms.py:TrainingForm`) |
| Results visualisation | section 4 — score curve, best value, train/test sizes, and a confusion matrix for the selected classifier |

The split between **parametric** interaction (the user moves a parameter: grid, test
fraction, seed) and what stays automatic (encoding, scaling, fitting) is the answer to
project task 4's "what should the user be in control of?", and the page states it in prose
above the form.

### Lecture 2 — Explainability: the accuracy↔interpretability trade-off

Not a task of project 1, but it decides the model menu. Each entry in `ml.MODELS` carries a
`note` shown in the model table: logistic regression is "one weight per feature, so the
model can be read directly", random forest is "usually accurate, but no longer readable by
a human". Choosing a model in this app is therefore framed as a trade-off, not as a
ranking by score.

### Lecture 12 — Fairness: bias visible before the score

The class-imbalance warning in `quality_warnings()` names the majority and minority class
with their counts and states that *accuracy will flatter the majority class* — the
representation-bias and metric-choice argument of lecture 12, placed where it changes
behaviour (before training) rather than in a report.

### Persistence

`project1/models.py` defines `Dataset` (the uploaded CSV plus what was inferred about it)
and `TrainingRun` (one sweep: model, grid, scores, split, seed, score name, and whether it
was automated). `TrainingRun` is what makes the human-vs-AutoML comparison in section 5
possible, and doubles as an audit trail of what the user tried.

---

## Project 2 — Explainability

*To be completed when `feature/project-2` is reviewed.* Planned lecture coverage: **L2**
(interpretable models, accuracy↔interpretability), **L4** (Rashomon set, regularisation for
interpretability), **L3** (counterfactual explanations with MAD-weighted L1; PDP and ALE,
hand-implemented).

## Project 3 — Active Learning for Learning-to-Defer

*To be completed when `feature/project-3` is reviewed.* Planned lecture coverage: **L5**
(rejection/deferral loss, (K+1)-class scorer, CSS surrogate), **L6** (pool-based active
learning and utility functions), **L9** (the simulated expert as a user model), **L1**
(baseline classifier), **L12** (expert-bias analysis in the report).

## Project 4 — Preference Elicitation User Study

*To be completed when `feature/project-4` is reviewed.* Planned lecture coverage: **L9**
(Luce models ⇒ Bradley–Terry ranking), **L7** (user-study design, between/within subjects,
piloting, ethics and consent), **L8** and **L6** (study interface and adaptive item
selection), **L12** (participant-data ethics).
