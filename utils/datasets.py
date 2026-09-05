"""CSV loading and inspection helpers shared by all projects.

Convention from the project description: the first row holds the feature names and the
last column is the target. Identifier columns, when present, are filtered out.
"""

import pandas as pd
from pandas.api import types as ptypes

ID_NAMES = {"id", "index", "unnamed: 0", "no", "sno", "s.no"}
MAX_CLASSES = 20
CLASS_RATIO = 0.05


def read_csv(file):
    """Read an uploaded CSV into a DataFrame, raising ValueError on unusable input."""
    try:
        df = pd.read_csv(file, skipinitialspace=True)
    except UnicodeDecodeError as exc:
        raise ValueError("The file is not UTF-8 text; please upload a plain CSV.") from exc
    except Exception as exc:
        raise ValueError(f"Could not parse the file as CSV: {exc}") from exc

    df.columns = [str(column).strip() for column in df.columns]
    if df.empty or df.shape[1] < 2:
        raise ValueError("The CSV needs at least one feature column and one target column.")
    return df


def drop_id_columns(df):
    """Remove identifier-like columns. Returns the cleaned frame and the dropped names."""
    dropped = [
        column
        for column in df.columns[:-1]
        if column.lower() in ID_NAMES
        or (ptypes.is_integer_dtype(df[column]) and df[column].is_unique and len(df) > 2)
    ]
    return df.drop(columns=dropped), dropped


def split_features_target(df):
    """Split into features and target, the target being the last column."""
    return df.iloc[:, :-1], df.iloc[:, -1]


def infer_task(y):
    """Guess whether the target describes a classification or a regression problem."""
    if not ptypes.is_numeric_dtype(y) or ptypes.is_bool_dtype(y):
        return "classification"

    unique = y.nunique(dropna=True)
    if ptypes.is_integer_dtype(y) and unique <= max(MAX_CLASSES, CLASS_RATIO * len(y)):
        return "classification"
    return "classification" if unique <= 2 else "regression"


def describe(df, dropped=()):
    """Summarise a dataset for display: shape, columns, missing values, target profile."""
    X, y = split_features_target(df)
    task = infer_task(y)
    summary = {
        "n_rows": len(df),
        "n_features": X.shape[1],
        "features": list(X.columns),
        "target": y.name,
        "task": task,
        "dropped_columns": list(dropped),
        "missing": int(df.isna().sum().sum()),
        "numeric_features": [c for c in X.columns if ptypes.is_numeric_dtype(X[c])],
        "preview_columns": list(df.columns),
        "preview_rows": df.head(10).values.tolist(),
    }
    if task == "classification":
        summary["target_distribution"] = y.value_counts().sort_index().to_dict()
    else:
        summary["target_range"] = (float(y.min()), float(y.max()), float(y.mean()))
    return summary


def quality_warnings(df, task):
    """Surface data problems the user should know about before trusting any model.

    Follows lecture 1: the human is hiding behind the data, so the interface must not
    silently hide collection and representation issues.
    """
    warnings = []
    X, y = split_features_target(df)

    missing = int(df.isna().sum().sum())
    if missing:
        warnings.append(f"{missing} missing value(s) found; rows with gaps are dropped before training.")

    if len(df) < 50:
        warnings.append(f"Only {len(df)} rows: test scores will be very unstable.")

    non_numeric = [c for c in X.columns if not ptypes.is_numeric_dtype(X[c])]
    if non_numeric:
        warnings.append(f"Non-numeric feature(s) {', '.join(non_numeric)} will be one-hot encoded.")

    if task == "classification":
        counts = y.value_counts()
        if len(counts) < 2:
            warnings.append("The target has a single class: no classification is possible.")
        elif counts.min() * 3 < counts.max():
            warnings.append(
                f"Class imbalance: '{counts.idxmax()}' has {counts.max()} rows against "
                f"{counts.min()} for '{counts.idxmin()}'. Accuracy will flatter the majority class."
            )
        if counts.min() < 5:
            warnings.append("At least one class has fewer than 5 rows; the split may leave it untested.")

    return warnings
