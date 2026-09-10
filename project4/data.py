"""Task 1: the film catalogue and the feature vector a preference is expressed over.

The utility model is U(x) = w.x, so the features decide what a preference can even be about.
Two things follow from that, and they drive every choice here.

First, each feature should be something a person can plausibly have a taste about: a genre, an
era, a running time, how celebrated or how obscure a film is. A feature nobody has an opinion
on contributes a weight nobody can interpret.

Second, the vector has to stay short. A participant will give perhaps thirty to fifty
comparisons, so a representation with hundreds of dimensions cannot be estimated from one
session. Roughly forty is the budget, which is why eras are decades rather than years and why
only the common certificates get their own column.

Everything numeric is standardised so the learned weights are comparable to each other; without
that, a weight on body count in minutes and a weight on an IMDb score would not be on the same
scale and the fitted w would be unreadable.

budget and gross are deliberately excluded. Both are heavily missing, and the values are
denominated in whatever currency the production reported, without a unit column. Including them
would let the model learn data-collection artefacts and present them as taste.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent / "data" / "movie_metadata.csv"

# The eight certificates that between them cover 99% of the catalogue; the rest collapse.
COMMON_RATINGS = ("R", "PG-13", "PG", "Unrated", "Not Rated", "G", "Approved", "X")
OTHER_RATING = "Other"

# Films before 1980 are thin on the ground, so they share a bucket rather than each decade
# getting a column that a participant would rarely see filled.
EARLIEST_DECADE = 1980
PRE_DECADE = "pre-1980"


@lru_cache(maxsize=1)
def catalogue():
    """The cleaned film catalogue, indexed 0..n-1. Treat the frame as read-only."""
    frame = pd.read_csv(DATA_FILE)

    # Every title carries a trailing non-breaking space in this dataset.
    frame["movie_title"] = frame["movie_title"].str.replace("\xa0", "", regex=False).str.strip()

    frame = frame.dropna(subset=["title_year", "genres", "imdb_score", "duration"])
    frame = frame.drop_duplicates(subset=["movie_title", "title_year"])

    frame["content_rating"] = frame["content_rating"].fillna("Unrated")
    frame["language"] = frame["language"].fillna("Unknown")
    frame["country"] = frame["country"].fillna("Unknown")
    for column in ("director_facebook_likes", "cast_total_facebook_likes", "num_voted_users"):
        frame[column] = frame[column].fillna(0)

    frame["title_year"] = frame["title_year"].astype(int)
    frame["decade"] = np.where(
        frame["title_year"] < EARLIEST_DECADE,
        PRE_DECADE,
        ((frame["title_year"] // 10) * 10).astype(str),
    )
    frame["rating_bucket"] = np.where(
        frame["content_rating"].isin(COMMON_RATINGS), frame["content_rating"], OTHER_RATING
    )
    frame["genre_list"] = frame["genres"].str.split("|")

    return frame.reset_index(drop=True)


@lru_cache(maxsize=1)
def genres():
    return tuple(sorted({genre for row in catalogue()["genre_list"] for genre in row}))


def _standardise(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() or 1.0)


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def eras():
    """Every era bucket present in the catalogue, oldest first."""
    decades = set(catalogue()["decade"]) - {PRE_DECADE}
    return (PRE_DECADE, *sorted(decades, key=int))


def features():
    """The feature matrix and the name of every column, in order."""
    frame = catalogue()
    columns, names = [], []

    for genre in genres():
        columns.append(frame["genre_list"].apply(lambda row, g=genre: float(g in row)).to_numpy())
        names.append(f"genre:{genre}")

    # Read the decades off the data rather than listing them, so the block stays a true
    # one-hot if the catalogue is ever refreshed with newer films.
    for decade in eras():
        columns.append((frame["decade"] == decade).astype(float).to_numpy())
        names.append(f"era:{decade}")

    for rating in (*COMMON_RATINGS, OTHER_RATING):
        columns.append((frame["rating_bucket"] == rating).astype(float).to_numpy())
        names.append(f"rating:{rating}")

    columns.append(_standardise(frame["duration"]))
    names.append("length")

    columns.append(_standardise(frame["imdb_score"]))
    names.append("acclaim")

    columns.append(_standardise(np.log1p(frame["director_facebook_likes"])))
    names.append("director fame")

    columns.append(_standardise(np.log1p(frame["cast_total_facebook_likes"])))
    names.append("cast fame")

    # How widely seen a film is, which separates a blockbuster from a cult favourite even when
    # the two are equally well reviewed.
    columns.append(_standardise(np.log1p(frame["num_voted_users"])))
    names.append("reach")

    columns.append((frame["language"] == "English").astype(float).to_numpy())
    names.append("english language")

    columns.append((frame["country"] == "USA").astype(float).to_numpy())
    names.append("us production")

    return np.column_stack(columns), tuple(names)


def matrix():
    return features()[0]


def feature_names():
    return features()[1]


def describe(index):
    """What the interface shows a participant about one film."""
    row = catalogue().iloc[int(index)]
    return {
        "index": int(index),
        "title": row["movie_title"],
        "year": int(row["title_year"]),
        "genres": ", ".join(row["genre_list"][:3]),
        "duration": int(row["duration"]),
        "rating": row["content_rating"],
        "score": float(row["imdb_score"]),
        "director": row["director_name"] if pd.notna(row["director_name"]) else "unknown",
    }


@lru_cache(maxsize=1)
def summary():
    """Facts about the catalogue, for the landing page and the report."""
    frame = catalogue()
    design, names = features()
    raw = len(pd.read_csv(DATA_FILE))
    return {
        "n_films": len(frame),
        "n_raw": raw,
        "n_dropped": raw - len(frame),
        "n_features": design.shape[1],
        "n_genres": len(genres()),
        "genres": list(genres()),
        "feature_names": list(names),
        "year_range": [int(frame["title_year"].min()), int(frame["title_year"].max())],
    }
