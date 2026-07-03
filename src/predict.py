"""Prediction helpers used by the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline

from src.data import load_dark_pattern_category_dataset, load_primary_binary_dataset
from src.filters import (
    is_low_context_product_snippet,
    is_simple_price_or_discount_snippet,
)
from src.modeling import load_model, make_pipeline, save_model

DEFAULT_MODEL_PATH = Path("artifacts/best_binary_model.joblib")
DEFAULT_CATEGORY_MODEL_PATH = Path("artifacts/best_category_model.joblib")


@dataclass(frozen=True)
class Prediction:
    """Prediction output for one text sample."""

    text: str
    label: int
    label_name: str
    confidence: float | None
    suppressed_by_filter: bool = False
    filter_reason: str | None = None


def train_default_model() -> Pipeline:
    """Train the default Logistic Regression model on the full primary dataset."""
    df = load_primary_binary_dataset()
    pipeline = make_pipeline("Logistic Regression")
    pipeline.fit(df["text"], df["label"])
    return pipeline


def get_or_train_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load the saved model or train and save a default model when missing."""
    if model_path.exists():
        return load_model(model_path)

    pipeline = train_default_model()
    save_model(pipeline, model_path)
    return pipeline


def train_default_category_model() -> Pipeline:
    """Train the default dark-pattern category model on dark-pattern rows."""
    df = load_dark_pattern_category_dataset()
    pipeline = make_pipeline("Logistic Regression")
    pipeline.fit(df["text"], df["category"])
    return pipeline


def get_or_train_category_model(
    model_path: Path = DEFAULT_CATEGORY_MODEL_PATH,
) -> Pipeline:
    """Load the saved category model or train and save one when missing."""
    if model_path.exists():
        return load_model(model_path)

    pipeline = train_default_category_model()
    save_model(pipeline, model_path)
    return pipeline


def predict_dark_pattern_category(text: str, pipeline: Pipeline) -> tuple[str, float | None]:
    """Predict the likely category for text already considered suspicious."""
    normalized = " ".join(text.strip().split())
    if not normalized:
        raise ValueError("Prediction text cannot be empty")

    category = str(pipeline.predict([normalized])[0])
    confidence = None
    if hasattr(pipeline[-1], "predict_proba"):
        classes = list(pipeline[-1].classes_)
        class_index = classes.index(category)
        confidence = float(pipeline.predict_proba([normalized])[0][class_index])
    return category, confidence


def predict_text(text: str, pipeline: Pipeline) -> Prediction:
    """Predict whether a text sample contains a dark pattern."""
    normalized = " ".join(text.strip().split())
    if not normalized:
        raise ValueError("Prediction text cannot be empty")

    label = int(pipeline.predict([normalized])[0])
    confidence = None
    if hasattr(pipeline[-1], "predict_proba"):
        confidence = float(pipeline.predict_proba([normalized])[0][label])

    return Prediction(
        text=normalized,
        label=label,
        label_name="Dark Pattern" if label == 1 else "Not Dark Pattern",
        confidence=confidence,
    )


def predict_text_for_demo(
    text: str,
    pipeline: Pipeline,
    *,
    suppress_simple_discounts: bool = True,
    suppress_product_titles: bool = True,
) -> Prediction:
    """Predict text and apply demo safeguards used by user-facing interfaces."""
    prediction = predict_text(text, pipeline)
    if (
        suppress_simple_discounts
        and prediction.label == 1
        and is_simple_price_or_discount_snippet(prediction.text)
    ):
        return Prediction(
            text=prediction.text,
            label=0,
            label_name="Not Dark Pattern",
            confidence=prediction.confidence,
            suppressed_by_filter=True,
            filter_reason=(
                "Plain price/discount text was suppressed because it did not include "
                "urgency or scarcity pressure language."
            ),
        )
    if (
        suppress_product_titles
        and prediction.label == 1
        and is_low_context_product_snippet(prediction.text)
    ):
        return Prediction(
            text=prediction.text,
            label=0,
            label_name="Not Dark Pattern",
            confidence=prediction.confidence,
            suppressed_by_filter=True,
            filter_reason=(
                "Catalog-style product title/spec text was suppressed because it did not "
                "include urgency or scarcity pressure language."
            ),
        )
    return prediction
