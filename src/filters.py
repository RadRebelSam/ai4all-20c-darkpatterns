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
    r"sale ends|deal ends|offer ends|expires|today only|one day only|"
    r"order within|don't miss out|do not miss out|almost gone|selling fast|"
    r"only\s+\d+\s+(left|remaining)|\d+\s+(left|remaining)\s+in stock|"
    r"few remaining|last one available|stock running low|low stock|"
    r"while supplies last|act now"
    r")",
    re.IGNORECASE,
)
URGENCY_RE = re.compile(
    r"(hurry|last chance|limited time|limited offer|ends soon|ends tonight|"
    r"sale ends|deal ends|offer ends|expires|today only|one day only|"
    r"order within|don't miss out|do not miss out|act now)",
    re.IGNORECASE,
)
SCARCITY_RE = re.compile(
    r"(almost gone|selling fast|only\s+\d+\s+(left|remaining)|"
    r"\d+\s+(left|remaining)\s+in stock|few remaining|last one available|"
    r"stock running low|low stock|while supplies last)",
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
MISDIRECTION_RE = re.compile(
    r"(no,?\s+i\s+(don't|do not)\s+want|"
    r"skip\s+(this\s+)?(deal|offer|discount)|"
    r"i\s+prefer\s+to\s+pay\s+full\s+price)",
    re.IGNORECASE,
)
OBSTRUCTION_RE = re.compile(
    r"(call\s+to\s+cancel|"
    r"contact\s+customer\s+service\s+to\s+(cancel|unsubscribe)|"
    r"cannot\s+be\s+cancelled\s+online|"
    r"multiple\s+steps\s+to\s+(cancel|unsubscribe|delete))",
    re.IGNORECASE,
)
SNEAKING_RE = re.compile(
    r"(automatically\s+(added|renews|enrolled)|"
    r"pre-?selected|"
    r"opt-?out\s+by\s+default|"
    r"processing\s+fee\s+added\s+at\s+checkout)",
    re.IGNORECASE,
)
FORCED_ACTION_RE = re.compile(
    r"(must\s+(create|sign up for)\s+an?\s+account|"
    r"required\s+to\s+(share|allow)|"
    r"you\s+must\s+agree\s+to\s+continue)",
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
BENIGN_FEATURE_COMPARISON_RE = re.compile(
    r"("
    r"\blimited\s+(library|storage|features?|functionality|selection|options|"
    r"catalog|capacity|support|integrations?|customization|resolution)\b|"
    r"\btime[- ]consuming\b|"
    r"\btakes?\s+too\s+much\s+time\b|"
    r"\baverage\s+.+\s+time\s*:\s*minutes?\s*,?\s*not\s+days?\b|"
    r"\breplacement\s+for\s+stock\s+footage\b|"
    r"\boverused\s+stock\s+footage\b|"
    r"\bcosts?\s+a\s+lot\b|"
    r"\bhard\s+to\s+use\b"
    r")",
    re.IGNORECASE,
)
REVIEW_PURCHASE_METADATA_RE = re.compile(
    r"("
    r"\bverified\s+purchase\b|"
    r"\b(purchased|reviewed)\s+in\s+.{2,80}\s+on\s+"
    r"(january|february|march|april|may|june|july|august|september|"
    r"october|november|december|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b|"
    r"\bdate\s+of\s+stay\s*:\s*|"
    r"\btrip\s+type\s*:\s*"
    r")",
    re.IGNORECASE,
)
COURSE_METADATA_RE = re.compile(
    r"("
    r"\b\d+(\.\d+)?\s+hours?\s+of\s+on[- ]demand\s+video\b|"
    r"\bcertificate\s+of\s+completion\b|"
    r"\bfull\s+lifetime\s+access\b|"
    r"\bdownloadable\s+resources?\b|"
    r"\bcoding\s+exercises?\b|"
    r"\bpractice\s+tests?\b"
    r")",
    re.IGNORECASE,
)
ORDER_CONFIRMATION_RE = re.compile(
    r"("
    r"^(your\s+)?(order|payment|purchase)\s+"
    r"(is\s+)?(confirmed|complete|completed|received|successful)\b|"
    r"^(thank\s+you|thanks)\s+for\s+your\s+(order|purchase)\b|"
    r"^we('ve|\s+have)\s+received\s+your\s+(order|payment)\b"
    r")",
    re.IGNORECASE,
)
NEUTRAL_AVAILABILITY_RE = re.compile(
    r"("
    r"^(currently\s+)?(in\s+stock|out\s+of\s+stock|unavailable)\b|"
    r"\bavailable\s+in\s+"
    r"((\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
    r"(sizes?|colou?rs?|styles?|formats?)\b|"
    r"\bavailable\s+for\s+(pickup|delivery|download)\b|"
    r"\bready\s+to\s+ship\b|"
    r"\bcheck\s+back\s+later\b"
    r")",
    re.IGNORECASE,
)
EDUCATIONAL_CONTEXT_RE = re.compile(
    r"("
    r"\b(this|the)\s+(article|guide|lesson|section|study|research)\b|"
    r"\bdark\s+patterns?\b|"
    r"\bexamples?\s+of\b|"
    r"\bknown\s+as\b|"
    r"\brefers?\s+to\b|"
    r"\b(research|studies)\s+(shows?|found)\b|"
    r"\b(websites?|companies|brands|apps?|games?|they)\s+"
    r"(use|uses|display|displays|show|shows|create|creates)\b"
    r")",
    re.IGNORECASE,
)
EDUCATIONAL_PATTERN_CUE_RE = re.compile(
    r"("
    r"\bdark\s+patterns?\b|"
    r"\b(urgency|scarcity|social\s+proof|confirmshaming)\b|"
    r"\b(countdown|pressure|manipulat|deceptive)\w*\b"
    r")",
    re.IGNORECASE,
)


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
        matches.append("Social Proof")
    if CONFIRMSHAMING_RE.search(snippet):
        matches.append("Confirmshaming")
    if MISDIRECTION_RE.search(snippet):
        matches.append("Misdirection")
    if OBSTRUCTION_RE.search(snippet):
        matches.append("Obstruction")
    if SNEAKING_RE.search(snippet):
        matches.append("Sneaking")
    if FORCED_ACTION_RE.search(snippet):
        matches.append("Forced Action")

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


def is_benign_feature_comparison_snippet(snippet: str) -> bool:
    """Detect ordinary feature limitations or product-comparison copy.

    Words such as "limited" and "time" are strong model features, but phrases
    such as "limited storage" or "time consuming" describe a product comparison
    rather than pressure on the shopper.
    """
    return bool(BENIGN_FEATURE_COMPARISON_RE.search(snippet)) and not (
        contains_pressure_language(snippet)
    )


def is_review_or_purchase_metadata_snippet(snippet: str) -> bool:
    """Detect review-platform metadata rather than social-proof pressure."""
    return bool(REVIEW_PURCHASE_METADATA_RE.search(snippet)) and not (
        contains_pressure_language(snippet)
    )


def is_course_metadata_snippet(snippet: str) -> bool:
    """Detect ordinary course contents and access information."""
    return bool(COURSE_METADATA_RE.search(snippet)) and not contains_pressure_language(
        snippet
    )


def is_order_confirmation_snippet(snippet: str) -> bool:
    """Detect completed order/payment messages without a countdown or threat."""
    return bool(ORDER_CONFIRMATION_RE.search(snippet.strip())) and not (
        contains_pressure_language(snippet)
    )


def is_neutral_availability_snippet(snippet: str) -> bool:
    """Detect ordinary availability or fulfillment labels without scarcity."""
    return bool(NEUTRAL_AVAILABILITY_RE.search(snippet.strip())) and not (
        contains_pressure_language(snippet)
    )


def is_educational_or_article_snippet(snippet: str) -> bool:
    """Detect text that discusses dark patterns instead of directing a shopper.

    This requires both reporting/educational language and a dark-pattern cue or
    recognized pressure phrase so ordinary marketing copy is not broadly hidden.
    """
    has_context = bool(EDUCATIONAL_CONTEXT_RE.search(snippet))
    has_pattern_cue = bool(EDUCATIONAL_PATTERN_CUE_RE.search(snippet))
    return has_context and (has_pattern_cue or contains_pressure_language(snippet))


def is_benign_context_snippet(snippet: str) -> bool:
    """Return whether any extended demo filter identifies benign context."""
    return any(
        (
            is_benign_feature_comparison_snippet(snippet),
            is_review_or_purchase_metadata_snippet(snippet),
            is_course_metadata_snippet(snippet),
            is_order_confirmation_snippet(snippet),
            is_neutral_availability_snippet(snippet),
            is_educational_or_article_snippet(snippet),
        )
    )
