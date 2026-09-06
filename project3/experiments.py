"""Runs every experiment once and writes what the page displays.

Training a deferral model and running an active-learning loop take far too long to happen
inside a request, so nothing here is called from a view. `manage.py run_project3` executes
this module, writes `results.json` and the figures, and both are committed. The page then
only reads them, which also means a reader sees the results without running anything.

Figures go under `static/`, not `MEDIA_ROOT`: media is gitignored and would not survive a
clone.
"""

import json
from pathlib import Path

from matplotlib import pyplot as plt

from . import active, classifier, data, defer, experts

RESULTS_FILE = Path(__file__).resolve().parent / "results" / "results.json"
FIGURE_DIR = Path(__file__).resolve().parent / "static" / "project3" / "figures"


def figure(name):
    """Save the current figure into the committed figure directory."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", dpi=110)
    plt.close()
    return f"project3/figures/{name}.png"


def _confusion_figure(matrix, topics, name, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(topics)), topics, rotation=30, ha="right")
    ax.set_yticks(range(len(topics)), topics)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(topics)):
        for j in range(len(topics)):
            ax.text(j, i, matrix[i][j], ha="center", va="center", fontsize=8,
                    color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "black")
    fig.colorbar(image, ax=ax)
    return figure(name)


def _expert_figure(reports, baseline_accuracy):
    topics = data.TOPICS
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    width = 0.8 / (len(reports) + 1)

    positions = range(len(topics))
    ax.bar([p - 0.4 + width / 2 for p in positions],
           [reports[0]["per_topic"][t]["classifier"] for t in topics],
           width=width, label="classifier")
    for index, report in enumerate(reports, start=1):
        ax.bar([p - 0.4 + width / 2 + index * width for p in positions],
               [report["per_topic"][t]["expert"] for t in topics],
               width=width, label=report["label"].lower())

    ax.axhline(baseline_accuracy, linestyle="--", linewidth=1, color="grey",
               label="classifier, overall")
    ax.set_xticks(list(positions), topics)
    ax.set_ylabel("accuracy on that topic")
    ax.set_title("Who is better, and where")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return figure("experts")


def _coverage_figure(result):
    """Accuracy against how often the human is asked, for both strategies."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for points, label, style in (
        (result["css_curve"], "learned deferral (CSS)", "-o"),
        (result["confidence_curve"], "confidence-based rejection", "--s"),
    ):
        ordered = sorted(points, key=lambda row: row["deferral_rate"])
        ax.plot([row["deferral_rate"] for row in ordered],
                [row["system_accuracy"] for row in ordered], style, markersize=4, label=label)

    ax.axhline(result["baseline_accuracy"], color="grey", linestyle=":", linewidth=1,
               label="classifier alone")
    ax.axhline(result["css"]["oracle_ceiling"], color="green", linestyle=":", linewidth=1,
               label="perfect routing")
    ax.set_xlabel("fraction of articles sent to the expert")
    ax.set_ylabel("system accuracy")
    ax.set_title(f"What deferring buys ({result['expert']})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return figure(f"coverage_{result['expert']}")


def _targeting_figure(result):
    """Where each strategy spends the same deferral budget."""
    topics = list(result["deferral_by_topic"])
    css = [result["deferral_by_topic"][t]["css"] for t in topics]
    conf = [result["deferral_by_topic"][t]["confidence"] for t in topics]
    better = [result["deferral_by_topic"][t]["expert_better"] for t in topics]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    positions = range(len(topics))
    ax.bar([p - 0.2 for p in positions], css, width=0.4, label="learned deferral")
    ax.bar([p + 0.2 for p in positions], conf, width=0.4, label="confidence-based")
    break_line = chr(10)
    labels = [
        topic + break_line + ("(expert better)" if flag else "(classifier better)")
        for topic, flag in zip(topics, better)
    ]
    ax.set_xticks(list(positions), labels, fontsize=8)
    ax.set_ylabel("fraction of that topic deferred")
    ax.set_title("Where the same deferral budget is spent")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return figure(f"targeting_{result['expert']}")


LABELS = {
    "proposed": "uncertain deferral decision x representativeness",
    "classifier_uncertainty": "classifier uncertainty only",
    "random": "random queries",
}


def _active_figure(result):
    """System accuracy against how many questions the expert was asked."""
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for strategy in result["strategies"]:
        curve = strategy["curve"]
        ax.plot([row["queries"] for row in curve],
                [row["system_accuracy"] for row in curve],
                marker="o", markersize=3, label=LABELS[strategy["strategy"]])

    ax.axhline(result["baseline_accuracy"], color="grey", linestyle=":", linewidth=1,
               label="classifier alone")
    ax.axhline(result["full_supervision"], color="green", linestyle=":", linewidth=1,
               label="every expert label known")
    ax.set_xlabel("expert labels collected")
    ax.set_ylabel("system accuracy")
    ax.set_title(f"Learning the expert's competence ({result['expert']})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return figure(f"active_{result['expert']}")


def run_all():
    """Produce every number and figure the page shows."""
    results = {"data": data.summary()}

    baseline = classifier.report()
    baseline["confusion_figure"] = _confusion_figure(
        baseline["confusion"], data.TOPICS, "baseline_confusion",
        "Baseline classifier on the test set",
    )
    results["baseline"] = baseline

    reports = [experts.report(name) for name in ("specialist", "generalist")]
    results["experts"] = {
        "reports": reports,
        "figure": _expert_figure(reports, baseline["accuracy"]),
    }

    deferral = []
    for name in ("specialist", "generalist"):
        result = defer.run(name)
        result["coverage_figure"] = _coverage_figure(result)
        result["targeting_figure"] = _targeting_figure(result)
        result.pop("css_curve"), result.pop("confidence_curve")
        deferral.append(result)
    results["deferral"] = deferral

    learning = active.run("specialist")
    learning["figure"] = _active_figure(learning)
    learning["strategy_labels"] = LABELS
    results["active"] = learning

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def load():
    """Read the committed results, or None if they have not been generated."""
    if not RESULTS_FILE.exists():
        return None
    return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
