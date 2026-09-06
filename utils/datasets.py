"""CSV loading and inspection helpers shared by all projects.

Convention from the project description: the first row holds the feature names and the
last column is the target. Identifier columns, when present, are filtered out.
"""

import pandas as pd
from pandas.api import types as ptypes

ID_NAMES = {"id", "index", "unnamed: 0", "no", "sno", "s.no"}
MAX_CLASSES = 20
CLASS_RATIO = 0.05
MIN_TRAINING_ROWS = 5
MIN_ROWS_FOR_ID_GUESS = 10


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


def _looks_like_a_serial(column):
    """A running index: unique integers increasing in steps of one."""
    if not ptypes.is_integer_dtype(column) or not column.is_unique:
        return False
    ordered = column.sort_values()
    return bool((ordered.diff().dropna() == 1).all())


def drop_id_columns(df):
    """Remove identifier columns. Returns the cleaned frame and the dropped names.

    Only names we recognise, or columns that are literally a running index, count. A
    unique integer column is not enough on its own: salaries, counts and years are all
    unique integers and dropping them silently destroys the dataset.
    """
    dropped = [
        column
        for column in df.columns[:-1]
        if column.lower() in ID_NAMES
        or (len(df) >= MIN_ROWS_FOR_ID_GUESS and _looks_like_a_serial(df[column]))
    ]
    if len(dropped) == len(df.columns) - 1:
        dropped = [column for column in dropped if column.lower() in ID_NAMES]
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


def describe(df, dropped=(), task=None):
    """Summarise a dataset for display: shape, columns, missing values, target profile.

    ``task`` overrides the detected one so the profile matches what the user chose.
    """
    X, y = split_features_target(df)
    task = task or infer_task(y)
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
    if task == "regression" and ptypes.is_numeric_dtype(y):
        summary["target_range"] = (float(y.min()), float(y.max()), float(y.mean()))
    else:
        summary["target_distribution"] = y.value_counts().sort_index().to_dict()
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


def validate_trainable(df, task):
    """Raise ValueError if this frame cannot be trained on under the requested task.

    Called at upload time so an unusable dataset never reaches the training views. The
    checks mirror what scikit-learn would otherwise raise deep inside a pipeline.
    """
    X, y = split_features_target(df)

    if X.shape[1] == 0:
        raise ValueError("No feature columns are left once identifier columns are removed.")

    usable = len(df.dropna())
    if usable < MIN_TRAINING_ROWS:
        raise ValueError(
            f"At least {MIN_TRAINING_ROWS} complete rows are needed to train and test; "
            f"this file has {usable}."
        )

    if task == "classification":
        if not ptypes.is_numeric_dtype(y) or ptypes.is_bool_dtype(y):
            pass
        elif infer_task(y) == "regression":
            raise ValueError(
                f"'{y.name}' holds {y.nunique()} distinct numbers and looks continuous. "
                "Choose Regression, or use a target with a small number of classes."
            )
        if y.nunique(dropna=True) < 2:
            raise ValueError(f"'{y.name}' has a single class, so there is nothing to classify.")
    elif task == "regression" and not ptypes.is_numeric_dtype(y):
        raise ValueError(
            f"'{y.name}' is not numeric, so it cannot be a regression target. Choose Classification."
        )
