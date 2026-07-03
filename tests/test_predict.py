import pytest

from src.data import load_primary_binary_dataset
from src.modeling import make_pipeline
from src.predict import predict_text, predict_text_for_demo


@pytest.fixture()
def small_pipeline():
    df = load_primary_binary_dataset()
    sample = df.groupby("label").head(80).reset_index(drop=True)
    pipeline = make_pipeline("Logistic Regression")
    pipeline.fit(sample["text"], sample["label"])
    return pipeline


def test_predict_text_returns_label_name_and_confidence(small_pipeline):
    prediction = predict_text(
        "Only 2 left in stock. Buy now before the sale ends.",
        small_pipeline,
    )

    assert prediction.label in {0, 1}
    assert prediction.label_name in {"Dark Pattern", "Not Dark Pattern"}
    assert prediction.confidence is None or 0 <= prediction.confidence <= 1


def test_predict_text_rejects_empty_input(small_pipeline):
    with pytest.raises(ValueError, match="cannot be empty"):
        predict_text("   ", small_pipeline)


def test_predict_text_for_demo_can_suppress_simple_discount(small_pipeline):
    raw = predict_text("Previous price: $99.00 47% off", small_pipeline)
    demo = predict_text_for_demo("Previous price: $99.00 47% off", small_pipeline)

    assert raw.label in {0, 1}
    if raw.label == 1:
        assert demo.label == 0
        assert demo.suppressed_by_filter
        assert demo.filter_reason is not None


def test_predict_text_for_demo_keeps_pressure_discount(small_pipeline):
    demo = predict_text_for_demo(
        "Hurry, 47% off ends soon. Only 2 left at this price.",
        small_pipeline,
    )

    assert not demo.suppressed_by_filter


def test_predict_text_for_demo_can_suppress_product_title_noise(small_pipeline):
    raw = predict_text(
        "2025 Tablet 10 inch Android 14 Tablet 8+64GB 1280x800 IPS Touchscreen 5000mAh US",
        small_pipeline,
    )
    demo = predict_text_for_demo(
        "2025 Tablet 10 inch Android 14 Tablet 8+64GB 1280x800 IPS Touchscreen 5000mAh US",
        small_pipeline,
    )

    assert raw.label in {0, 1}
    if raw.label == 1:
        assert demo.label == 0
        assert demo.suppressed_by_filter
        assert demo.filter_reason is not None
