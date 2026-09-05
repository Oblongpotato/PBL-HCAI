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


def save_figure(fig, prefix="plot"):
    """Save ``fig`` under MEDIA_ROOT/plots and return the URL to display it."""
    directory = Path(settings.MEDIA_ROOT) / PLOT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)

    filename = f"{prefix}-{uuid.uuid4().hex[:12]}.png"
    fig.savefig(directory / filename, bbox_inches="tight", dpi=110)
    plt.close(fig)

    return f"{settings.MEDIA_URL}{PLOT_SUBDIR}/{filename}"
