"""The supervised learning pipeline of lecture 1, driven by the user's choices.

Steps 1-4 of the lecture map onto: pick a hypothesis class (MODELS), split the data,
fit one model per hyperparameter value, and score each on the held-out test set.
"""

from pandas.api import types as ptypes
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MODELS = {
    "logistic": {
        "label": "Logistic regression",
        "task": "classification",
        "hyperparameter": "C",
        "grid": [0.01, 0.1, 1, 10, 100],
        "integer": False,
        "note": "One weight per feature, so the model can be read directly.",
        "factory": lambda value: LogisticRegression(C=value, max_iter=2000),
    },
    "tree": {
        "label": "Decision tree",
        "task": "classification",
        "hyperparameter": "max_depth",
        "grid": [1, 2, 3, 4, 5, 6, 8, 10],
        "integer": True,
        "note": "Interpretable rules; deeper trees fit more and generalise less.",
        "factory": lambda value: DecisionTreeClassifier(max_depth=value, random_state=0),
    },
    "knn": {
        "label": "k-nearest neighbours",
        "task": "classification",
        "hyperparameter": "n_neighbors",
        "grid": [1, 3, 5, 7, 9, 15, 25],
        "integer": True,
        "note": "No training phase; the whole dataset is the model.",
        "factory": lambda value: KNeighborsClassifier(n_neighbors=value),
    },
    "forest": {
        "label": "Random forest",
        "task": "classification",
        "hyperparameter": "n_estimators",
        "grid": [5, 10, 25, 50, 100, 200],
        "integer": True,
        "note": "Usually accurate, but no longer readable by a human.",
        "factory": lambda value: RandomForestClassifier(n_estimators=value, random_state=0),
    },
    "ridge": {
        "label": "Ridge regression",
        "task": "regression",
        "hyperparameter": "alpha",
        "grid": [0.001, 0.01, 0.1, 1, 10, 100],
        "integer": False,
        "note": "The lecture's example: linear model with a quadratic penalisation.",
        "factory": lambda value: Ridge(alpha=value),
    },
    "tree_reg": {
        "label": "Regression tree",
        "task": "regression",
        "hyperparameter": "max_depth",
        "grid": [1, 2, 3, 4, 5, 6, 8, 10],
        "integer": True,
        "note": "Piecewise-constant predictions along readable splits.",
        "factory": lambda value: DecisionTreeRegressor(max_depth=value, random_state=0),
    },
    "knn_reg": {
        "label": "k-nearest neighbours",
        "task": "regression",
        "hyperparameter": "n_neighbors",
        "grid": [1, 3, 5, 7, 9, 15, 25],
        "integer": True,
        "note": "Averages the targets of the closest points.",
        "factory": lambda value: KNeighborsRegressor(n_neighbors=value),
    },
    "forest_reg": {
        "label": "Random forest",
        "task": "regression",
        "hyperparameter": "n_estimators",
        "grid": [5, 10, 25, 50, 100, 200],
        "integer": True,
        "note": "Strong baseline, at the cost of interpretability.",
        "factory": lambda value: RandomForestRegressor(n_estimators=value, random_state=0),
    },
}

SCORES = {
    "accuracy": {"label": "Accuracy", "task": "classification", "fn": accuracy_score, "maximise": True},
    "f1_macro": {
        "label": "Macro F1",
        "task": "classification",
        "fn": lambda y, p: f1_score(y, p, average="macro"),
        "maximise": True,
    },
    "balanced_accuracy": {
        "label": "Balanced accuracy",
        "task": "classification",
        "fn": balanced_accuracy_score,
        "maximise": True,
    },
    "r2": {"label": "R2", "task": "regression", "fn": r2_score, "maximise": True},
    "mse": {"label": "Mean squared error", "task": "regression", "fn": mean_squared_error, "maximise": False},
    "mae": {"label": "Mean absolute error", "task": "regression", "fn": mean_absolute_error, "maximise": False},
}

DEFAULT_MODEL = {"classification": "tree", "regression": "ridge"}
DEFAULT_SCORE = {"classification": "accuracy", "regression": "r2"}


def models_for(task):
    return {key: spec for key, spec in MODELS.items() if spec["task"] == task}


def scores_for(task):
    return {key: spec for key, spec in SCORES.items() if spec["task"] == task}


def _preprocessor(X):
    """Scale numeric columns, one-hot encode the rest. Automated on purpose."""
    numeric = [c for c in X.columns if ptypes.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    steps = [("numeric", StandardScaler(), numeric)]
    if categorical:
        steps.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(steps)


def run_sweep(frame, target, task, model_key, values, test_size, random_state, scoring):
    """Split, fit one model per hyperparameter value, score on the held-out test set."""
    spec = MODELS[model_key]
    score = SCORES[scoring]

    frame = frame.dropna()
    X, y = frame.drop(columns=[target]), frame[target]

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    results, best_index, best_prediction = [], 0, None
    for index, value in enumerate(values):
        pipeline = Pipeline([("prepare", _preprocessor(X)), ("model", spec["factory"](value))])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        results.append(float(score["fn"](y_test, prediction)))

        improved = results[index] > results[best_index] if score["maximise"] else results[index] < results[best_index]
        if index == 0 or improved:
            best_index, best_prediction = index, prediction

    summary = {
        "values": list(values),
        "scores": results,
        "best_value": values[best_index],
        "best_score": results[best_index],
        "hyperparameter": spec["hyperparameter"],
        "model_label": spec["label"],
        "score_label": score["label"],
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    if task == "classification":
        labels = sorted(y.astype(str).unique())
        summary["labels"] = labels
        matrix = confusion_matrix(
            y_test.astype(str), best_prediction.astype(str), labels=labels
        ).tolist()
        summary["confusion"] = list(zip(labels, matrix))
    return summary


def automatic_choice(task):
    """Every decision taken without the user: the AutoML baseline of lecture 1."""
    model_key = DEFAULT_MODEL[task]
    return {
        "model_key": model_key,
        "values": MODELS[model_key]["grid"],
        "test_size": 0.3,
        "random_state": 0,
        "scoring": DEFAULT_SCORE[task],
    }
