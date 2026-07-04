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
URGENCY_RE = re.compile(
    r"(hurry|last chance|limited time|limited offer|ends soon|ends tonight|"
    r"sale ends|deal ends|expires|act now)",
    re.IGNORECASE,
)
SCARCITY_RE = re.compile(
    r"(almost gone|selling fast|only\s+\d+\s+(left|remaining)|"
    r"\d+\s+(left|remaining)\s+in stock|low stock|while supplies last)",
    re.IGNORECASE,
)
SOCIAL_PROOF_RE = re.compile(
    r"(\d+[\d,]*\s+(people|customers|users|shoppers|visitors|members)\b|"
    r"people are (looking|viewing|watching)|someone just bought|"
    r"popular|trending|best[- ]?seller|reviews?|ratings?|deals bought)",
    re.IGNORECASE,
)
CONFIRMSHAMING_RE = re.compile(
    r"(no thanks|i don't want|i do not want|not interested).*(save|deal|offer|discount|money)",
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
BARE_SOCIAL_COUNTER_RE = re.compile(
    r"^(deals bought|customers|users|reviews|ratings|downloads|students|members)\s*:\s*[\d,]+$",
    re.IGNORECASE,
)
TESTIMONIAL_FRAGMENT_RE = re.compile(
    r"^("
    r"thanks\s+[a-z][a-z .'-]*!?|"
    r"thank you\s+[a-z][a-z .'-]*!?|"
    r"why\s+i\s+bought\s+this!?|"
    r"showed\s+me\s+.+|"
    r".+\s+walk\s+through\s+call\s+yesterday\.?"
    r")$",
    re.IGNORECASE,
)
CONTENT_TITLE_RE = re.compile(
    r"^(\d+\)?\s*)?(copy|write|create|generate|build|structure|template|lesson|module)\b",
    re.IGNORECASE,
)
PAST_TENSE_LEFT_RE = re.compile(r"\bleft\b", re.IGNORECASE)


def contains_pressure_language(snippet: str) -> bool:
    """Return whether text includes urgency/scarcity pressure language."""
    return bool(PRESSURE_LANGUAGE_RE.search(snippet))


def infer_dark_pattern_type(snippet: str) -> str:
    """Infer a likely dark-pattern category from surface text cues.

    This is a rule-based explanation helper, not a trained multi-class model.
    It should be shown as "possible type" rather than a definitive category.
    """
    matches = []
    if URGENCY_RE.search(snippet):
        matches.append("Urgency")
    if SCARCITY_RE.search(snippet):
        matches.append("Scarcity")
    if SOCIAL_PROOF_RE.search(snippet):
        matches.append("Social proof")
    if CONFIRMSHAMING_RE.search(snippet):
        matches.append("Confirmshaming")

    if not matches:
        return "Unclear from text alone"
    return " + ".join(matches)


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


def is_low_context_web_snippet(snippet: str) -> bool:
    """Detect webpage fragments that are too thin to judge as dark patterns.

    These snippets often come from testimonials, counters, menus, or course/module
    titles. They can be useful context on a page, but by themselves they are not
    enough evidence for a dark-pattern flag unless they include pressure language.
    """
    if contains_pressure_language(snippet):
        return False

    cleaned = " ".join(snippet.split())
    words = cleaned.split()
    if len(words) <= 3:
        return True
    if BARE_SOCIAL_COUNTER_RE.match(cleaned):
        return True
    if TESTIMONIAL_FRAGMENT_RE.match(cleaned):
        return True
    if CONTENT_TITLE_RE.match(cleaned) and len(words) <= 12:
        return True
    if PAST_TENSE_LEFT_RE.search(cleaned):
        return True
    return False
