# Notes: where the lecture material shows up in the code

One Django project with four apps (`project1`–`project4`), all reachable from the home page
at `/home/`. These notes record, for each project, which lecture material I used and where
it ended up in the code.

Shared across all projects:

| Path | Role |
|---|---|
| `pbl/settings.py`, `pbl/urls.py` | app registration and routing; all four apps are registered up front |
| `home/views.py` | group members and project links, defined in Python (project 1, task 1) |
| `templates/base.html`, `static/style.css` | global layout and style |
| `static/<app>/`, `templates/<app>/` | the per-app paths required by the assignment |
| `utils/datasets.py`, `utils/plotting.py` | CSV loading and inspection, and rendering matplotlib figures into media |
| `demos/` | the reference app that came with the skeleton; I read it but did not change it |

## Datasets

| Project | Data | Where it came from |
|---|---|---|
| 1 | any uploaded CSV | first row holds the feature names, last column is the target; developed against the course iris file |
| 2 | `project2/data/penguins.csv` | exported once from the `palmerpenguins` package and committed, so the numbers are reproducible and the app needs no download at runtime |

The tests do not read the course material, since it is not committed. `project1/tests.py`
builds its iris frame from `sklearn.datasets.load_iris`, which is the same data.

---

## Project 1 — Supervised Learning Interface

App `project1/`, route `/project1/`. Upload a CSV, look at it, train a model on it.

The page is one scrolling workflow of five blocks: Data, What the app understood, Visualise,
Train, Automation. That is the order of the pipeline the course opens with.

### Lecture 1: the ML pipeline

The lecture's four steps (pick a hypothesis class, split the data, fit for several
hyperparameter values, score) are the control flow of `project1/ml.py:run_sweep()`.

`ml.MODELS` is a registry of eight estimators, four per task type. Each entry names the one
hyperparameter the app exposes, a default range for it, and a short note on the trade-off,
which the "Train a model" table shows to the user. Ridge regression is in the list because
it is the lecture's own worked example of a linear model with a quadratic penalty.

The split took some thought. An earlier version picked the hyperparameter on the test set
and then reported that same score, which flatters the model: on a pure-noise dataset it
claimed 0.50 accuracy. Now the test set is held out first, the hyperparameter is chosen by
5-fold cross-validation inside the training set only, the winning value is refit on the full
training set, and the untouched test score is what the page reports. Both numbers are shown,
because the gap between them is the interesting part. On the same noise dataset the reported
score is now 0.389.

`ml.SCORES` carries a `maximise` flag, so error scores such as MSE and MAE select the
smallest value while accuracy-type scores select the largest.

### Lecture 1: AutoML, and who is in control

The lecture's objection to AutoML is that the user is not in control and their domain
expertise goes unused. Rather than describe that, the app lets you compare the two.
`ml.automatic_choice()` fixes the model, the range, the split and the score with no user
input, `views.automl` runs it, and the run history table marks every row with who decided:
you or the app. Both appear side by side on the same data.

The lecture's other point, that a human is hiding behind every step, is why
`utils/datasets.quality_warnings()` exists. Class imbalance, missing values, a small sample,
non-numeric columns and rare classes are reported before any score is shown.

### Lecture 8: what an interactive ML interface needs

Lecture 8 lists four things such an interface should offer. Each is a block on the page:

| Element | Where |
|---|---|
| Data visualisation | block 3, `project1/plots.py`: feature pair coloured by target, feature against target, per-class distribution, correlation matrix |
| Pre-processing | block 2: detected task, dropped identifier columns, quality warnings, a 10-row preview. Scaling and one-hot encoding happen in `ml._preprocessor()` and are stated on the page |
| Model selection and tuning | block 4: model, hyperparameter range, test fraction, seed and score are all form fields (`project1/forms.py:TrainingForm`) |
| Results visualisation | block 4: the cross-validation curve, the chosen value, the test score, train and test sizes, and a confusion matrix |

This is also my answer to task 4's question about what the user should control. The user
moves parameters: the range, the test fraction, the seed, the score. The app handles
encoding, scaling, splitting and fitting, and says so above the form.

### Lecture 2: accuracy against interpretability

Not one of project 1's tasks, but it decides what goes in the model menu. Every entry in
`ml.MODELS` carries a note shown in the table: logistic regression gives one weight per
feature and can be read directly, a random forest is usually more accurate but no longer
readable. Picking a model is presented as a trade-off rather than a ranking by score.

### Lecture 12: bias and privacy

The class-imbalance warning names the majority and minority class with their counts and says
that accuracy will flatter the majority class. It appears before training rather than in a
footnote.

Uploaded files are also a privacy question. `MEDIA_ROOT` is served publicly by `pbl/urls.py`,
so `Dataset.file` writes to a separate `private_uploads/` directory under a random filename,
and the caller's own filename is discarded.

### What is stored

`project1/models.py` defines `Dataset` (the uploaded file and what was inferred about it) and
`TrainingRun` (one sweep: model, range, cross-validation scores, split, seed, score name, the
selection score, the test score, and whether it was automated). `TrainingRun` is what makes
the comparison in block 5 possible, and it doubles as a record of what was tried.

---

## Project 2 — Explainability

App `project2/`, route `/project2/`. Palmer Penguins, 333 complete rows out of 344,
predicting `species` from four measurements plus island, sex and year. 100 rows are held out
for testing.

Everything on the page describes one model, and the choice of that model lives in the query
string (`?family=&lam=&row=&target=&feature=`). Choosing a different model class or a
different λ changes the tree, the counterfactuals and the effect plots together, which is
what the assignment asks for.

### Lectures 2 and 4: interpretable models and a complexity penalty

Two families are offered, a decision tree and L1-penalised logistic regression. Both are
fitted at a range of regularisation strengths, and Ω(f) measures how complex the result is:
leaves for the tree, and the number of original features with a non-zero coefficient for
logistic regression. The λ slider picks the model maximising `accuracy − λ·Ω(f)`.

Two different regularisers are involved and the page keeps them apart. `max_leaf_nodes` and
`C` constrain a model while it is being fitted. λ weighs complexity afterwards, when choosing
between models that have already been fitted.

The trade-off curve is where lecture 4 shows up. Accuracy flattens after about three leaves,
so the region in which a much simpler model is as good as a complicated one is visible rather
than asserted. Each family has its own λ range, because their frontiers turn over in
different places; one shared range left one family with a slider that did nothing. Pushed far
enough, the logistic model loses every coefficient and predicts a single species, which is
what demanding maximum simplicity actually costs.

Trees are fitted without scaling. They are scale-invariant anyway, and unscaled thresholds
read as `flipper_length_mm <= 207.5` instead of `<= 0.459`.

### Lecture 3: counterfactuals

`project2/counterfactuals.py` follows the method from the lecture. Sample points around the
chosen penguin, keep the ones the selected model assigns to the target species, and rank them
by MAD-weighted L1 distance, so a millimetre of bill and a gram of body mass are compared on
the scale the data itself sets. Numeric features are noised on their own scale. Island, sex
and year cannot be nudged, so they are redrawn from the values that actually occur. If
nothing is found the search widens before giving up.

### Lecture 3: PDP and ALE

`project2/effects.py` implements both by hand, as required. PDP replaces the chosen feature
with each grid value across the whole dataset and averages the predicted probabilities; the
three curves sum to 1 at every point. ALE accumulates local differences inside quantile
intervals, using only the penguins that fall in each interval, which keeps it usable when
features are correlated.

The lecture asks which model has an exact partial derivative. A multinomial logistic model
does: ∂P_c/∂x_j = P_c(w_cj − Σ_k P_k w_kj), corrected for the standardisation applied before
fitting. That version is drawn over the finite-difference estimate and the two agree to
0.0024, which is the check that the derivation is right. A decision tree is piecewise
constant, so its derivative is zero almost everywhere with jumps at the split thresholds, and
only the discretised estimate exists for it. The page says which case applies.

---

## Project 3 — Active Learning for Learning-to-Defer

Not implemented yet. It will cover lecture 5 (rejection and deferral losses, the (K+1)-class
scorer, the cost-sensitive softmax surrogate), lecture 6 (pool-based active learning and
query strategies), lecture 9 (the simulated expert as a user model) and lecture 1 (the
baseline classifier).

## Project 4 — Preference Elicitation User Study

Not implemented yet. It will cover lecture 9 (Luce models and the Bradley-Terry extension to
rankings), lecture 7 (study design, between- and within-subjects, piloting, consent) and
lecture 8 (the study interface).
