"""Plots offered by the visualisation panel.

Every function returns the media URL of a rendered PNG, following the approach shown in
the demos app: matplotlib figures cannot be streamed to a template directly.
"""

from matplotlib import pyplot as plt

from utils.plotting import save_figure

FIGSIZE = (6.4, 4.4)
KINDS = [
    ("pair", "Two features, coloured by target"),
    ("target", "One feature against the target"),
    ("distribution", "Distribution of one feature"),
    ("correlation", "Correlation between features"),
]


def _axes(title, xlabel, ylabel=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return fig, ax


def _pair(frame, target, task, x, y):
    fig, ax = _axes(f"{y} against {x}", x, y)
    if task == "classification":
        for label, group in frame.groupby(target, observed=True):
            ax.scatter(group[x], group[y], s=24, alpha=0.8, label=str(label))
        ax.legend(title=target, fontsize=8)
    else:
        points = ax.scatter(frame[x], frame[y], c=frame[target], cmap="viridis", s=24)
        fig.colorbar(points, ax=ax, label=target)
    return fig, f"Feature pair coloured by {target}."


def _target(frame, target, task, x, _y):
    fig, ax = _axes(f"{target} against {x}", x, target)
    if task == "classification":
        categories = sorted(frame[target].astype(str).unique())
        positions = {label: index for index, label in enumerate(categories)}
        ax.scatter(frame[x], frame[target].astype(str).map(positions), s=24, alpha=0.7)
        ax.set_yticks(range(len(categories)), categories)
    else:
        ax.scatter(frame[x], frame[target], s=24, alpha=0.7)
    return fig, f"How {x} relates to the label to predict."


def _distribution(frame, target, task, x, _y):
    fig, ax = _axes(f"Distribution of {x}", x, "count")
    if task == "classification":
        for label, group in frame.groupby(target, observed=True):
            ax.hist(group[x], bins=20, alpha=0.6, label=str(label))
        ax.legend(title=target, fontsize=8)
    else:
        ax.hist(frame[x], bins=20)
    return fig, f"Spread of {x}; overlapping classes are hard to separate."


def _correlation(frame, _target, _task, _x, _y):
    numeric = frame.select_dtypes("number")
    matrix = numeric.corr()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.columns)
    ax.set_title("Correlation between numeric columns")
    fig.colorbar(image, ax=ax)
    return fig, "Strongly correlated features carry redundant information."


BUILDERS = {"pair": _pair, "target": _target, "distribution": _distribution, "correlation": _correlation}


def build(frame, target, task, kind, x=None, y=None):
    """Render the requested plot and return (image url, caption)."""
    fig, caption = BUILDERS[kind](frame, target, task, x, y)
    return save_figure(fig, f"p1-{kind}"), caption


def score_curve(values, scores, hyperparameter, score_label, best_value, test_score):
    """Selection score against hyperparameter, with the untouched test score marked."""
    fig, ax = _axes(f"{score_label} against {hyperparameter}", hyperparameter, score_label)
    ax.plot(range(len(values)), scores, marker="o", label="cross-validation (training set)")
    ax.set_xticks(range(len(values)), [str(value) for value in values])
    ax.axvline(values.index(best_value), color="tab:red", linestyle="--", linewidth=1, label="selected")
    ax.plot([values.index(best_value)], [test_score], marker="*", markersize=14,
            color="tab:green", linestyle="none", label="test score of the selected model")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return save_figure(fig, "p1-curve")
