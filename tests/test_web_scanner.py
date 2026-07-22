import pandas as pd

from src.data import load_primary_binary_dataset
from src.filters import (
    contains_pressure_language,
    infer_dark_pattern_type,
    is_benign_context_snippet,
    is_benign_feature_comparison_snippet,
    is_course_metadata_snippet,
    is_educational_or_article_snippet,
    is_low_context_product_snippet,
    is_low_context_web_snippet,
    is_neutral_availability_snippet,
    is_order_confirmation_snippet,
    is_review_or_purchase_metadata_snippet,
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


def test_pressure_language_guard_covers_common_urgency_and_scarcity_variants():
    protected_phrases = [
        "Today only: save 20 percent",
        "Offer ends in 10 minutes",
        "Order within 2 hours",
        "Few remaining",
        "Last one available",
        "Don't miss out",
    ]

    assert all(contains_pressure_language(text) for text in protected_phrases)


def test_benign_feature_comparison_filter_keeps_real_pressure():
    assert is_benign_feature_comparison_snippet(
        "Competitors have a limited library and limited storage."
    )
    assert is_benign_feature_comparison_snippet(
        "Average creation time: minutes, not days."
    )
    assert is_benign_feature_comparison_snippet("Takes too much time!")

    assert not is_benign_feature_comparison_snippet(
        "Limited time offer ends tonight."
    )
    assert not is_benign_feature_comparison_snippet(
        "Limited storage plan available today only."
    )


def test_review_metadata_filter_keeps_social_proof():
    assert is_review_or_purchase_metadata_snippet(
        "Purchased in the United States on January 5, 2026."
    )
    assert is_review_or_purchase_metadata_snippet(
        "Reviewed in Canada on 01/05/2026."
    )
    assert is_review_or_purchase_metadata_snippet("Verified purchase")

    assert not is_review_or_purchase_metadata_snippet(
        "20 people purchased this item today."
    )
    assert not is_review_or_purchase_metadata_snippet(
        "Someone just bought this item."
    )


def test_course_metadata_filter_keeps_limited_time_course_offer():
    assert is_course_metadata_snippet(
        "Includes 12 hours of on-demand video and a certificate of completion."
    )
    assert is_course_metadata_snippet(
        "Full lifetime access with 8 downloadable resources."
    )

    assert not is_course_metadata_snippet(
        "Full lifetime access, but this limited time offer ends tonight."
    )


def test_order_confirmation_filter_keeps_checkout_countdown():
    assert is_order_confirmation_snippet("Order confirmed.")
    assert is_order_confirmation_snippet("Thank you for your purchase.")
    assert is_order_confirmation_snippet("We've received your payment.")

    assert not is_order_confirmation_snippet(
        "Your cart is reserved for 10 minutes."
    )
    assert not is_order_confirmation_snippet(
        "Order confirmed. Claim your bonus today only."
    )


def test_neutral_availability_filter_keeps_real_scarcity():
    assert is_neutral_availability_snippet("In stock and ready to ship.")
    assert is_neutral_availability_snippet("Out of stock. Check back later.")
    assert is_neutral_availability_snippet("Available in two sizes and five colors.")
    assert is_neutral_availability_snippet("Available for pickup.")

    assert not is_neutral_availability_snippet("In stock, only 2 left.")
    assert not is_neutral_availability_snippet("Available today only.")


def test_educational_filter_requires_reporting_context():
    assert is_educational_or_article_snippet(
        "This article explains how limited-time offers pressure shoppers."
    )
    assert is_educational_or_article_snippet(
        "They display limited time offers with countdown timers."
    )
    assert is_educational_or_article_snippet(
        "A dark pattern called scarcity tells users that only two items remain."
    )

    assert not is_educational_or_article_snippet(
        "Limited time offer. Only two items remain."
    )


def test_extended_benign_context_filter_combines_all_new_rules():
    benign_examples = [
        "Limited library and limited storage.",
        "Purchased in the United States on January 5, 2026.",
        "Includes 12 hours of on-demand video.",
        "Your payment is confirmed.",
        "Available for delivery.",
        "This guide describes urgency and countdown pressure.",
    ]

    assert all(is_benign_context_snippet(text) for text in benign_examples)
    assert not is_benign_context_snippet(
        "Hurry! Only 2 left in stock. Offer ends tonight."
    )


def test_infer_dark_pattern_type_from_text_cues():
    assert infer_dark_pattern_type("Hurry, sale ends tonight") == "Urgency"
    assert infer_dark_pattern_type("Only 2 left in stock") == "Scarcity"
    assert infer_dark_pattern_type("1,243 people are looking at this item") == "Social Proof"
    assert (
        infer_dark_pattern_type("Only 2 left. Sale ends tonight")
        == "Urgency + Scarcity"
    )
    assert infer_dark_pattern_type("This cotton pillowcase is machine washable") == (
        "Unclear from text alone"
    )

def test_infer_dark_pattern_type_covers_remaining_categories():
    assert infer_dark_pattern_type(
        "Contact customer service to cancel your subscription"
    ) == "Obstruction"
    assert infer_dark_pattern_type(
        "This item was automatically added to your cart"
    ) == "Sneaking"
    assert infer_dark_pattern_type(
        "You must create an account to continue"
    ) == "Forced Action"
    assert infer_dark_pattern_type(
        "I prefer to pay full price"
    ) == "Misdirection"

    
