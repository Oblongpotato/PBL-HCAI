"""Rendering of matplotlib figures into the media directory.

Django cannot display a figure directly, so every plot is written to MEDIA_ROOT and
served through MEDIA_URL. Unlike the demo in ``demos/views.py``, filenames are unique
so concurrent users never overwrite each other's plots.
"""

import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from django.conf import settings
from matplotlib import pyplot as plt

PLOT_SUBDIR = "plots"
KEEP_PLOTS = 300


def _prune(directory, keep=KEEP_PLOTS):
    """Drop the oldest figures once the directory grows past ``keep``.

    Every page view of projects 1 and 2 writes new PNGs and nothing else ever removes
    them, so without this the directory grows for as long as the server runs. The files
    are pure output: anything deleted here is redrawn the next time it is asked for.
    """
    figures = sorted(directory.glob("*.png"), key=lambda path: path.stat().st_mtime)
    for stale in figures[:-keep]:
        stale.unlink(missing_ok=True)


def save_figure(fig, prefix="plot"):
    """Save ``fig`` under MEDIA_ROOT/plots and return the URL to display it."""
    directory = Path(settings.MEDIA_ROOT) / PLOT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)

    filename = f"{prefix}-{uuid.uuid4().hex[:12]}.png"
    fig.savefig(directory / filename, bbox_inches="tight", dpi=110)
    plt.close(fig)
    _prune(directory)

    return f"{settings.MEDIA_URL}{PLOT_SUBDIR}/{filename}"
