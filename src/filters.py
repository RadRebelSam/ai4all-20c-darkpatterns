"""Rule-based filters used by demo interfaces."""

from __future__ import annotations

import re

PRICE_OR_DISCOUNT_RE = re.compile(
    r"(\$\s?\d|£\s?\d|€\s?\d|\d+\s?%+\s*off|sale price|previous price|was\s+\$|now\s+\$)",
    re.IGNORECASE,
)
PRESSURE_LANGUAGE_RE = re.compile(
    r"("
    r"hurry|last chance|limited time|limited offer|ends soon|ends tonight|"
    r"sale ends|deal ends|expires|almost gone|selling fast|"
    r"only\s+\d+\s+(left|remaining)|\d+\s+(left|remaining)\s+in stock|"
    r"low stock|while supplies last|act now"
    r")",
    re.IGNORECASE,
)
PRODUCT_TITLE_NOISE_RE = re.compile(
    r"("
    r"\b(us|usa|uk|eu)\s+stock\b|"
    r"\bnear\s+mint\b|"
    r"\bopen\s+box\b|"
    r"\brefurbished\b|"
    r"\b\d+(\.\d+)?\s?(mm|gb|tb|mah|hz|inch|inches|mp|w)\b|"
    r"\b\d+\s?x\s?\d+\b|"
    r"\bf/\d+(\.\d+)?\b|"
    r"\b(android|ios)\s+\d+\b"
    r")",
    re.IGNORECASE,
)
PRODUCT_CATALOG_WORD_RE = re.compile(
    r"("
    r"tablet|phone|camera|lens|telephoto|zoom|touchscreen|processor|storage|"
    r"battery|screen|display|laptop|monitor|headphones|speaker|watch"
    r")",
    re.IGNORECASE,
)


def contains_pressure_language(snippet: str) -> bool:
    """Return whether text includes urgency/scarcity pressure language."""
    return bool(PRESSURE_LANGUAGE_RE.search(snippet))


def is_simple_price_or_discount_snippet(snippet: str) -> bool:
    """Detect plain price/discount labels that often create false positives.

    This intentionally does not suppress price text that also includes pressure
    language such as "hurry", "limited time", or "only 2 left".
    """
    return bool(PRICE_OR_DISCOUNT_RE.search(snippet)) and not contains_pressure_language(
        snippet
    )


def is_low_context_product_snippet(snippet: str) -> bool:
    """Detect product-title/spec snippets that often create demo false positives.

    This filter is intentionally conservative. It suppresses catalog-like product
    titles, model numbers, specs, and stock-origin labels only when there is no
    urgency or scarcity pressure language.
    """
    if contains_pressure_language(snippet):
        return False

    has_catalog_word = bool(PRODUCT_CATALOG_WORD_RE.search(snippet))
    noise_matches = PRODUCT_TITLE_NOISE_RE.findall(snippet)
    has_many_numbers = sum(char.isdigit() for char in snippet) >= 6

    return bool(noise_matches) and (has_catalog_word or has_many_numbers)
