"""Scan visible webpage text for possible dark-pattern phrases."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.filters import infer_dark_pattern_type
from src.predict import (
    get_or_train_category_model,
    get_or_train_model,
    predict_dark_pattern_category,
)
from src.web_scanner import (
    extract_visible_text_with_playwright,
    score_snippets,
    split_visible_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan visible webpage text for possible dark patterns."
    )
    parser.add_argument("url", help="Webpage URL to scan")
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Show non-dark-pattern snippets too",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of snippets to print",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser window during scanning",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=3000,
        help="Milliseconds to wait after page load so popups can appear",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Minimum model confidence for displayed dark-pattern snippets",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each scanning step for demo narration",
    )
    parser.add_argument(
        "--show-snippets",
        type=int,
        default=0,
        help="Print the first N extracted snippets before model filtering",
    )
    parser.add_argument(
        "--no-price-filter",
        action="store_true",
        help="Do not suppress simple price/discount snippets without pressure language",
    )
    parser.add_argument(
        "--no-product-filter",
        action="store_true",
        help="Do not suppress catalog-style product title/spec snippets without pressure language",
    )
    parser.add_argument(
        "--no-context-filter",
        action="store_true",
        help="Do not suppress testimonial fragments, bare counters, and vague short snippets",
    )
    return parser.parse_args()


def log_step(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[step] {message}")


def possible_type_for_text(text: str, category_model):
    """Return likely category from the category model, with rule fallback."""
    try:
        category, confidence = predict_dark_pattern_category(text, category_model)
        if confidence is None:
            return category
        return f"{category} ({confidence:.1%})"
    except Exception:
        return infer_dark_pattern_type(text)


def main() -> None:
    args = parse_args()
    log_step(args.verbose, "Loading trained model from artifacts/ or training a default model if missing.")
    model = get_or_train_model()
    log_step(
        args.verbose,
        f"Opening webpage with Playwright: {args.url} "
        f"({'visible browser' if args.headed else 'headless browser'}).",
    )
    log_step(args.verbose, f"Waiting {args.wait_ms} ms after DOM load for dynamic content or popups.")
    page_text = extract_visible_text_with_playwright(
        args.url,
        wait_ms=args.wait_ms,
        headless=not args.headed,
    )
    log_step(args.verbose, f"Extracted {len(page_text):,} characters of visible body text.")
    log_step(args.verbose, "Splitting visible text into short snippets suitable for the model.")
    snippets = split_visible_text(page_text)
    log_step(args.verbose, f"Created {len(snippets):,} unique snippets after cleaning and deduplication.")

    if args.show_snippets:
        print()
        print(f"First {min(args.show_snippets, len(snippets))} extracted snippets before prediction:")
        for index, snippet in enumerate(snippets[: args.show_snippets], start=1):
            print(f"{index}. {snippet}")

    log_step(
        args.verbose,
        "Running each snippet through TF-IDF + Logistic Regression to predict dark pattern vs not dark pattern.",
    )
    results = score_snippets(
        snippets,
        model,
        dark_only=not args.include_neutral,
        min_confidence=None if args.include_neutral else args.threshold,
        suppress_simple_discounts=not args.no_price_filter,
        suppress_product_titles=not args.no_product_filter,
        suppress_context_light=not args.no_context_filter,
    )
    if args.include_neutral:
        log_step(args.verbose, "Showing both dark-pattern and neutral predictions.")
    else:
        log_step(
            args.verbose,
            f"Filtering results to dark-pattern predictions with confidence >= {args.threshold:.0%}.",
        )
        if not args.no_price_filter:
            log_step(
                args.verbose,
                "Suppressing plain price/discount snippets unless they include pressure language.",
            )
        if not args.no_product_filter:
            log_step(
                args.verbose,
                "Suppressing catalog-style product titles/specs unless they include pressure language.",
            )
        if not args.no_context_filter:
            log_step(
                args.verbose,
                "Suppressing testimonial fragments, bare counters, and vague short snippets.",
            )

    print()
    print(f"Scanned URL: {args.url}")
    print(f"Visible text snippets reviewed: {len(snippets)}")
    if not args.include_neutral:
        print(f"Displayed dark-pattern threshold: {args.threshold:.0%}")
    print()

    if not results:
        print("No likely dark-pattern text snippets were flagged above the threshold.")
        return

    category_model = get_or_train_category_model()
    for index, result in enumerate(results[: args.limit], start=1):
        confidence = result.prediction.confidence
        confidence_text = "n/a" if confidence is None else f"{confidence:.1%}"
        print(f"{index}. {result.prediction.label_name} ({confidence_text})")
        print(f"   Possible type: {possible_type_for_text(result.snippet, category_model)}")
        print(f"   {result.snippet}")


if __name__ == "__main__":
    main()
