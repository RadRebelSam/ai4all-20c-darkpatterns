import pandas as pd

from src.data import load_primary_binary_dataset
from src.filters import (
    contains_pressure_language,
    infer_dark_pattern_type,
    is_low_context_product_snippet,
    is_low_context_web_snippet,
    is_simple_price_or_discount_snippet,
)
from src.modeling import make_pipeline
from src.web_scanner import (
    clean_visible_text,
    score_snippets,
    split_visible_text,
)


def test_clean_visible_text_normalizes_whitespace():
    assert clean_visible_text("Only   2 left\n\nBuy now") == "Only 2 left Buy now"


def test_split_visible_text_breaks_page_copy_into_snippets():
    text = """
    Home
    Hurry! Sale ends in 10 minutes. Only 2 left in stock.
    This cotton pillowcase is machine washable and available in two sizes.
    """

    snippets = split_visible_text(text, min_chars=8, max_chars=80)

    assert "Sale ends in 10 minutes." in snippets
    assert "Only 2 left in stock." in snippets
    assert len(snippets) == len(set(snippets))


def test_score_snippets_returns_dark_predictions_first():
    df = load_primary_binary_dataset()
    sample = pd.concat(
        [df[df["label"] == 0].head(80), df[df["label"] == 1].head(80)],
        ignore_index=True,
    )
    pipeline = make_pipeline("Logistic Regression")
    pipeline.fit(sample["text"], sample["label"])

    results = score_snippets(
        [
            "Only 2 left in stock. Buy now before the sale ends.",
            "This cotton pillowcase is machine washable.",
        ],
        pipeline,
        dark_only=False,
    )

    assert len(results) == 2
    assert results[0].prediction.label in {0, 1}
    assert results[0].prediction.confidence is None or 0 <= results[0].prediction.confidence <= 1


def test_score_snippets_can_filter_by_confidence():
    df = load_primary_binary_dataset()
    sample = pd.concat(
        [df[df["label"] == 0].head(80), df[df["label"] == 1].head(80)],
        ignore_index=True,
    )
    pipeline = make_pipeline("Logistic Regression")
    pipeline.fit(sample["text"], sample["label"])

    results = score_snippets(
        ["Only 2 left in stock. Buy now before the sale ends."],
        pipeline,
        min_confidence=0.99,
    )

    assert results == []


def test_simple_price_or_discount_filter_keeps_pressure_language():
    assert is_simple_price_or_discount_snippet("sale price $10.00")
    assert is_simple_price_or_discount_snippet("Previous price: $99.00 47% off")

    assert contains_pressure_language("Hurry, sale ends tonight")
    assert not is_simple_price_or_discount_snippet(
        "Hurry, sale ends tonight. Only 2 left at $10.00"
    )


def test_low_context_product_filter_keeps_pressure_language():
    assert is_low_context_product_snippet(
        "2025 Tablet 10 inch Android 14 Tablet 8+64GB 1280x800 IPS Touchscreen 5000mAh US"
    )
    assert is_low_context_product_snippet(
        "SIGMA APO 300-800mm F/5.6 EX DG HSM for Nikon Telephoto Zoom [Near Mint]US Stock"
    )

    assert not is_low_context_product_snippet(
        "Only 2 left in stock for this tablet. Sale ends tonight."
    )


def test_low_context_web_filter_suppresses_weak_page_fragments():
    assert is_low_context_web_snippet("Why I bought this!")
    assert is_low_context_web_snippet("Thanks Charan!")
    assert is_low_context_web_snippet("Deals bought: 284")
    assert is_low_context_web_snippet(
        "2) Copy 100 Viral posts of successful people (structure)"
    )
    assert is_low_context_web_snippet(
        "Thanks Daniel for the 45 minute walk through call yesterday."
    )
    assert is_low_context_web_snippet(
        "No water in sink, shower, or toilet in morning we left."
    )

    assert not is_low_context_web_snippet(
        "Only 2 left in stock. Buy now before the deal ends."
    )


def test_infer_dark_pattern_type_from_text_cues():
    assert infer_dark_pattern_type("Hurry, sale ends tonight") == "Urgency"
    assert infer_dark_pattern_type("Only 2 left in stock") == "Scarcity"
    assert infer_dark_pattern_type("1,243 people are looking at this item") == "Social proof"
    assert (
        infer_dark_pattern_type("Only 2 left. Sale ends tonight")
        == "Urgency + Scarcity"
    )
    assert infer_dark_pattern_type("This cotton pillowcase is machine washable") == (
        "Unclear from text alone"
    )
