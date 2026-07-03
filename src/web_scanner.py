"""Utilities for scanning webpage text with the dark-pattern model."""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass
from typing import Iterable

from sklearn.pipeline import Pipeline

from src.filters import (
    contains_pressure_language,
    is_low_context_product_snippet,
    is_low_context_web_snippet,
    is_simple_price_or_discount_snippet,
)
from src.predict import Prediction, predict_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class SnippetPrediction:
    """A model prediction for one webpage text snippet."""

    snippet: str
    prediction: Prediction


def clean_visible_text(text: str) -> str:
    """Normalize browser-extracted text into a compact plain-text string."""
    return " ".join(text.split())


def split_visible_text(
    text: str,
    *,
    min_chars: int = 12,
    max_chars: int = 220,
) -> list[str]:
    """Split page text into snippets suitable for model prediction.

    The model was trained on short website phrases, not whole pages. This keeps
    snippets near the phrase/sentence level while preserving short marketing copy.
    """
    raw_chunks = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(text)]
    snippets: list[str] = []

    for chunk in raw_chunks:
        cleaned = clean_visible_text(chunk)
        if len(cleaned) < min_chars:
            continue
        if len(cleaned) <= max_chars:
            snippets.append(cleaned)
            continue

        words = cleaned.split()
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if len(candidate) <= max_chars:
                current.append(word)
            else:
                if len(" ".join(current)) >= min_chars:
                    snippets.append(" ".join(current))
                current = [word]
        if len(" ".join(current)) >= min_chars:
            snippets.append(" ".join(current))

    return dedupe_preserve_order(snippets)


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    """Remove duplicates without changing first-seen order."""
    seen = set()
    deduped = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def score_snippets(
    snippets: Iterable[str],
    pipeline: Pipeline,
    *,
    dark_only: bool = True,
    min_confidence: float | None = None,
    suppress_simple_discounts: bool = True,
    suppress_product_titles: bool = True,
    suppress_context_light: bool = True,
) -> list[SnippetPrediction]:
    """Run the trained model over snippets and sort likely dark patterns first."""
    results = []
    for snippet in snippets:
        prediction = predict_text(snippet, pipeline)
        if dark_only and prediction.label != 1:
            continue
        if (
            min_confidence is not None
            and prediction.confidence is not None
            and prediction.confidence < min_confidence
        ):
            continue
        if (
            suppress_simple_discounts
            and prediction.label == 1
            and is_simple_price_or_discount_snippet(snippet)
        ):
            continue
        if (
            suppress_product_titles
            and prediction.label == 1
            and is_low_context_product_snippet(snippet)
        ):
            continue
        if (
            suppress_context_light
            and prediction.label == 1
            and is_low_context_web_snippet(snippet)
        ):
            continue
        results.append(SnippetPrediction(snippet=snippet, prediction=prediction))

    return sorted(
        results,
        key=lambda result: (
            result.prediction.label,
            result.prediction.confidence if result.prediction.confidence is not None else 0,
        ),
        reverse=True,
    )


def extract_visible_text_with_playwright(
    url: str,
    *,
    wait_ms: int = 3000,
    headless: bool = True,
) -> str:
    """Open a webpage with Playwright and return visible body text.

    Raises:
        RuntimeError: if Playwright is not installed or browser binaries are missing.
    """
    if sys.platform.startswith("win") and hasattr(
        asyncio, "WindowsProactorEventLoopPolicy"
    ):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install -r requirements.txt` "
            "and then `python -m playwright install chromium`."
        ) from exc

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(wait_ms)
            text = page.locator("body").inner_text(timeout=10000)
            browser.close()
            return text
    except (PlaywrightError, NotImplementedError) as exc:
        raise RuntimeError(
            "Unable to scan the page with Playwright. If this is the first run, "
            "install the browser with `python -m playwright install chromium`."
        ) from exc
