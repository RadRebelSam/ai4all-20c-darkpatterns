"""Model training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.data import LABEL_COLUMN, TEXT_COLUMN

RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelResult:
    """Evaluation result for a trained model."""

    name: str
    pipeline: Pipeline
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "model": self.name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def make_classifier(model_name: str) -> BaseEstimator:
    """Create a classifier by display name."""
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if model_name == "Naive Bayes":
        return MultinomialNB()
    if model_name == "Linear SVM":
        return LinearSVC(random_state=RANDOM_STATE)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=25, random_state=RANDOM_STATE)
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=35,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def make_pipeline(model_name: str) -> Pipeline:
    """Create a TF-IDF text classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=8000,
                    stop_words="english",
                ),
            ),
            ("classifier", make_classifier(model_name)),
        ]
    )


def model_names() -> list[str]:
    """Return the classifiers used in the project comparison."""
    return [
        "Logistic Regression",
        "Naive Bayes",
        "Linear SVM",
        "Decision Tree",
        "Random Forest",
    ]


def split_dataset(df: pd.DataFrame):
    """Split a binary dataset into stratified train and test partitions."""
    return train_test_split(
        df[TEXT_COLUMN],
        df[LABEL_COLUMN],
        test_size=0.2,
        stratify=df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )


def evaluate_pipeline(name: str, pipeline: Pipeline, x_test, y_test) -> ModelResult:
    """Evaluate a fitted pipeline with binary classification metrics."""
    predictions = pipeline.predict(x_test)
    return ModelResult(
        name=name,
        pipeline=pipeline,
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, zero_division=0),
        recall=recall_score(y_test, predictions, zero_division=0),
        f1=f1_score(y_test, predictions, zero_division=0),
    )


def train_and_compare(df: pd.DataFrame) -> list[ModelResult]:
    """Train all configured models and return results sorted by F1 score."""
    x_train, x_test, y_train, y_test = split_dataset(df)
    results = []
    for name in model_names():
        pipeline = make_pipeline(name)
        pipeline.fit(x_train, y_train)
        results.append(evaluate_pipeline(name, pipeline, x_test, y_test))
    return sorted(results, key=lambda result: result.f1, reverse=True)


@dataclass(frozen=True)
class CrossValResult:
    """Cross-validation summary for a single model."""

    name: str
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_f1_mean: float
    cv_f1_std: float

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "cv_accuracy_mean": round(self.cv_accuracy_mean, 4),
            "cv_accuracy_std": round(self.cv_accuracy_std, 4),
            "cv_f1_mean": round(self.cv_f1_mean, 4),
            "cv_f1_std": round(self.cv_f1_std, 4),
        }


def cross_validate_model(name: str, df: pd.DataFrame, n_splits: int = 5) -> CrossValResult:
    """Run stratified k-fold cross-validation for a single model."""
    pipeline = make_pipeline(name)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        df[TEXT_COLUMN],
        df[LABEL_COLUMN],
        cv=cv,
        scoring=["accuracy", "f1"],
        n_jobs=-1,
    )
    return CrossValResult(
        name=name,
        cv_accuracy_mean=float(np.mean(scores["test_accuracy"])),
        cv_accuracy_std=float(np.std(scores["test_accuracy"])),
        cv_f1_mean=float(np.mean(scores["test_f1"])),
        cv_f1_std=float(np.std(scores["test_f1"])),
    )


def cross_validate_all(df: pd.DataFrame, n_splits: int = 5) -> list[CrossValResult]:
    """Run cross-validation for all configured models, sorted by mean F1."""
    results = [cross_validate_model(name, df, n_splits) for name in model_names()]
    return sorted(results, key=lambda r: r.cv_f1_mean, reverse=True)


def results_to_frame(results: list[ModelResult]) -> pd.DataFrame:
    """Convert model results into a metrics table."""
    return pd.DataFrame([result.as_dict() for result in results])


def save_model(pipeline: Pipeline, path: Path) -> None:
    """Persist a trained pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    """Load a persisted pipeline."""
    return joblib.load(path)
