"""The AG News subsample and the feature views built from it.

The committed CSVs are a stratified subsample of `fancyzhx/ag_news`, drawn once with a fixed
seed: 2000 training and 500 test articles per topic. Keeping them in the repository means
the numbers on the page are reproducible and nothing is downloaded at runtime.

Two feature views are needed. The baseline classifier works on sparse TF-IDF, which is what
text classifiers are good at. The deferral and competence models need dense vectors of
manageable width, so the same TF-IDF is reduced with truncated SVD.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent / "data"

TOPICS = ["World", "Sports", "Business", "Sci/Tech"]
CODES = {topic: index for index, topic in enumerate(TOPICS)}

SVD_COMPONENTS = 300
SEED = 0


@lru_cache(maxsize=1)
def load():
    """The raw subsample as (train, test) frames of text and topic."""
    train = pd.read_csv(DATA_DIR / "agnews_train.csv")
    test = pd.read_csv(DATA_DIR / "agnews_test.csv")
    return train, test


@lru_cache(maxsize=1)
def features():
    """Vectorise once per process. Treat the returned arrays as read-only."""
    train, test = load()

    tfidf = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"
    )
    X_train = tfidf.fit_transform(train["text"])
    X_test = tfidf.transform(test["text"])

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
    Z_train = svd.fit_transform(X_train)
    Z_test = svd.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "Z_train": Z_train,
        "Z_test": Z_test,
        "y_train": train["topic"].map(CODES).to_numpy(),
        "y_test": test["topic"].map(CODES).to_numpy(),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "vocabulary": len(tfidf.vocabulary_),
    }


def summary():
    """What the data panel shows."""
    train, test = load()
    parts = features()
    return {
        "n_train": len(train),
        "n_test": len(test),
        "topics": TOPICS,
        "per_topic_train": train["topic"].value_counts().reindex(TOPICS).to_dict(),
        "vocabulary": parts["vocabulary"],
        "svd_components": SVD_COMPONENTS,
        "explained_variance": round(parts["explained_variance"], 3),
        "example": train.iloc[0]["text"][:220],
        "example_topic": train.iloc[0]["topic"],
    }


def topic_names(codes):
    return np.asarray(TOPICS, dtype=object)[np.asarray(codes)]
