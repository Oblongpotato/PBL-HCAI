"""Figures for the explainability interface, rendered to PNGs under MEDIA_ROOT."""

from matplotlib import pyplot as plt
from sklearn.tree import plot_tree

from utils.plotting import save_figure

from . import data, training


def decision_tree(pipeline):
    """The model itself, which for a tree is the explanation.

    The figure is sized from the leaf count and shown at its natural size in a scrolling
    box, because a tree shrunk to the width of the page cannot be read.
    """
    model = pipeline.named_steps["model"]
    width = max(9, min(22, model.get_n_leaves() * 1.7))
    fig, ax = plt.subplots(figsize=(width, width * 0.48))
    plot_tree(
        model,
        feature_names=training.source_features(pipeline),
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=10,
        ax=ax,
    )
    return save_figure(fig, "p2-tree")


def coefficients(pipeline):
    """One bar per class per surviving feature: the logistic model read directly."""
    model = pipeline.named_steps["model"]
    origins = training.source_features(pipeline)
    encoded = pipeline.named_steps["prepare"].get_feature_names_out()
    keep = [i for i, origin in enumerate(origins) if origin in training.used_features(pipeline)]

    labels = [encoded[i].partition("__")[2] for i in keep]
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32 * len(keep) + 1.4)))
    height = 0.8 / len(model.classes_)
    for offset, (species, row) in enumerate(zip(model.classes_, model.coef_)):
        positions = [i + offset * height for i in range(len(keep))]
        ax.barh(positions, [row[i] for i in keep], height=height, label=species)

    ax.set_yticks([i + 0.4 - height / 2 for i in range(len(keep))], labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("weight (on standardised features)")
    ax.set_title("Weights of the selected model")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    return save_figure(fig, "p2-coefficients")


def tradeoff(family, lam, selected):
    """Accuracy against complexity across the whole pool, with the choice marked."""
    entries = sorted(training.pool(family), key=lambda entry: entry["complexity"])
    omegas = [entry["complexity"] for entry in entries]
    accuracies = [entry["accuracy"] for entry in entries]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(omegas, accuracies, marker="o", label="test accuracy")
    ax.plot(omegas, [training.objective(entry, lam) for entry in entries], marker="s",
            linestyle="--", label=f"accuracy - {lam:g} x complexity")
    ax.scatter([selected["complexity"]], [selected["accuracy"]], s=160, marker="*",
               color="tab:red", zorder=5, label="selected")

    ax.set_xlabel(f"complexity: {training.COMPLEXITY_LABEL[family]}")
    ax.set_ylabel("test accuracy")
    ax.set_title("What complexity buys")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return save_figure(fig, "p2-tradeoff")


def effect_curves(grid, curves, feature, title, extra=None):
    """One curve per species, optionally overlaid with a second estimate."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for index, species in enumerate(data.species()):
        colour = f"C{index}"
        ax.plot(grid, curves[species], color=colour, label=species)
        if extra:
            ax.plot(grid, extra[species], color=colour, linestyle=":", linewidth=2.2)

    if extra:
        ax.plot([], [], color="grey", linestyle=":", label="exact derivative")
    ax.set_xlabel(feature)
    ax.set_ylabel("effect on predicted probability")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return save_figure(fig, "p2-effect")
