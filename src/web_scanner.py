"""Utilities for scanning webpage text with the dark-pattern model."""

from __future__ import annotations

import asyncio
import html.parser
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

from sklearn.pipeline import Pipeline

from src.filters import (
    contains_pressure_language,
    is_benign_context_snippet,
    is_low_context_product_snippet,
    is_low_context_web_snippet,
    is_simple_price_or_discount_snippet,
)
from src.predict import Prediction, predict_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
SKIPPED_HTML_TAGS = {"script", "style", "noscript", "svg", "canvas"}


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


class _VisibleTextParser(html.parser.HTMLParser):
    """Small HTML text extractor used when Playwright is unavailable or times out."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIPPED_HTML_TAGS:
            self._skip_depth += 1
        if tag in {"br", "div", "li", "p", "section", "tr"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_HTML_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"div", "li", "p", "section", "tr"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        return clean_visible_text(" ".join(self._chunks))


def extract_visible_text_with_http(url: str, *, timeout: int = 10) -> str:
    """Fetch page HTML directly and return best-effort visible text."""
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read(2_000_000).decode(charset, errors="replace")
    except (OSError, URLError) as exc:
        raise RuntimeError(f"Unable to fetch page HTML: {exc}") from exc

    parser = _VisibleTextParser()
    parser.feed(html)
    text = parser.text()
    if not text:
        raise RuntimeError("The page loaded, but no readable text was found.")
    return text


def score_snippets(
    snippets: Iterable[str],
    pipeline: Pipeline,
    *,
    dark_only: bool = True,
    min_confidence: float | None = None,
    suppress_simple_discounts: bool = True,
    suppress_product_titles: bool = True,
    suppress_context_light: bool = True,
    suppress_benign_context: bool = True,
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
            suppress_benign_context
            and prediction.label == 1
            and is_benign_context_snippet(snippet)
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


def _extract_visible_text_with_playwright_direct(
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
            page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type
                in {"image", "media", "font", "stylesheet"}
                else route.continue_(),
            )
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=8000)
                page.wait_for_timeout(wait_ms)
                text = page.locator("body").inner_text(timeout=4000)
            finally:
                browser.close()
            return text
    except (PlaywrightError, NotImplementedError) as exc:
        raise RuntimeError(
            "Unable to scan the page with Playwright. If this is the first run, "
            "install the browser with `python -m playwright install chromium`."
        ) from exc


def _extract_visible_text_with_playwright_subprocess(
    url: str,
    *,
    wait_ms: int,
    headless: bool,
    timeout_ms: int,
) -> str:
    payload = json.dumps({"url": url, "wait_ms": wait_ms, "headless": headless})
    env = {**os.environ, "WEB_SCANNER_PLAYWRIGHT_CHILD": "1"}
    completed = subprocess.run(
        [sys.executable, "-m", "src.web_scanner", "--extract-visible-text"],
        input=payload,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout_ms / 1000,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(error or "Playwright extraction failed.")
    return completed.stdout


def extract_visible_text_with_playwright(
    url: str,
    *,
    wait_ms: int = 3000,
    headless: bool = True,
    timeout_ms: int | None = None,
) -> str:
    """Open a webpage and return visible body text with a bounded Playwright run.

    If Chromium hangs or the page blocks browser automation, this falls back to a
    plain HTML fetch so the Streamlit app can still return useful text quickly.
    """
    timeout_ms = timeout_ms or max(12000, wait_ms + 14000)

    if os.environ.get("WEB_SCANNER_PLAYWRIGHT_CHILD") == "1":
        return _extract_visible_text_with_playwright_direct(
            url,
            wait_ms=wait_ms,
            headless=headless,
        )

    try:
        return _extract_visible_text_with_playwright_subprocess(
            url,
            wait_ms=wait_ms,
            headless=headless,
            timeout_ms=timeout_ms,
        )
    except (RuntimeError, subprocess.TimeoutExpired) as playwright_exc:
        try:
            return extract_visible_text_with_http(url)
        except RuntimeError as http_exc:
            raise RuntimeError(
                "Unable to scan the page. Playwright did not finish in time, "
                f"and the HTML fallback also failed: {http_exc}"
            ) from playwright_exc


def _main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--extract-visible-text":
        payload = json.loads(sys.stdin.read())
        text = _extract_visible_text_with_playwright_direct(
            payload["url"],
            wait_ms=payload["wait_ms"],
            headless=payload["headless"],
        )
        sys.stdout.write(text)
        return
    raise SystemExit("Usage: python -m src.web_scanner --extract-visible-text")


if __name__ == "__main__":
    _main()
