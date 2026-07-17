"""Streamlit demo for dark-pattern recognition."""

# Ensure project root is on sys.path so `from src import ...` works when
# Streamlit launches the app from a different working directory.
import html
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.data import load_primary_binary_dataset
from src import filters as demo_filters
from src.modeling import make_pipeline, model_names
from src.predict import (
    get_or_train_category_model,
    get_or_train_model,
    predict_dark_pattern_category,
    predict_text,
    predict_text_for_demo,
)
from src.web_scanner import (
    extract_visible_text_with_playwright,
    score_snippets,
    split_visible_text,
)

st.set_page_config(
    page_title="Dark Pattern Recognition",
    page_icon="!",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #667085;
        --line: #d9e2ec;
        --paper: #ffffff;
        --mist: #f5f7fb;
        --teal: #11756f;
        --teal-soft: #dff4ef;
        --amber: #b7791f;
        --amber-soft: #fff2d6;
        --rose: #b42318;
        --rose-soft: #ffe4e2;
        --blue-soft: #e7f0ff;
    }

    .main .block-container {
        max-width: 1080px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
    }

    .hero {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(135deg, #ffffff 0%, #edf7f5 55%, #fff6e8 100%);
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: var(--ink);
        font-size: 2.1rem;
        line-height: 1.1;
        margin: 0 0 0.35rem;
    }

    .hero p {
        color: var(--muted);
        font-size: 1.02rem;
        margin: 0;
        max-width: 760px;
    }

    .result-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--paper);
        padding: 1rem;
        margin: 0.75rem 0;
    }

    .result-card.danger {
        border-color: #f3b5af;
        background: var(--rose-soft);
    }

    .result-card.clear {
        border-color: #a8ddd2;
        background: var(--teal-soft);
    }

    .result-card.warning {
        border-color: #f4c771;
        background: var(--amber-soft);
    }

    .eyebrow {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .result-title {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 750;
        line-height: 1.15;
        margin: 0 0 0.4rem;
    }

    .result-copy {
        color: var(--muted);
        margin: 0;
    }

    .model-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 0.75rem;
        margin-top: 0.75rem;
    }

    .model-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--paper);
        padding: 0.9rem;
    }

    .model-name {
        color: var(--ink);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .model-role {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 0.45rem;
    }

    .pill {
        display: inline-block;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.18rem 0.55rem;
        margin-bottom: 0.55rem;
    }

    .pill.dark {
        color: var(--rose);
        background: var(--rose-soft);
    }

    .pill.clear {
        color: var(--teal);
        background: var(--teal-soft);
    }

    .small-note {
        color: var(--muted);
        font-size: 0.85rem;
        margin: 0.15rem 0 0;
    }

    .vote-row {
        margin: 0.7rem 0;
    }

    .vote-label {
        display: flex;
        justify-content: space-between;
        color: var(--ink);
        font-weight: 700;
        margin-bottom: 0.28rem;
    }

    .bar-track {
        height: 0.9rem;
        border-radius: 999px;
        background: #e8edf3;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
    }

    .bar-fill.dark {
        background: #d92d20;
    }

    .bar-fill.clear {
        background: #0f766e;
    }

    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        background: var(--paper);
    }

    .comparison-table th {
        background: var(--mist);
        color: var(--ink);
        font-size: 0.85rem;
        text-align: left;
        padding: 0.7rem 0.8rem;
    }

    .comparison-table td {
        border-top: 1px solid var(--line);
        color: var(--muted);
        padding: 0.7rem 0.8rem;
    }

    .model-row {
        margin: 0.85rem 0;
    }

    .model-row-label {
        align-items: baseline;
        color: var(--ink);
        display: flex;
        font-weight: 700;
        justify-content: space-between;
        margin-bottom: 0.3rem;
    }

    .model-result {
        color: var(--muted);
        font-size: 0.86rem;
        font-weight: 600;
    }

    .model-bar-track {
        background: #e8edf3;
        border-radius: 999px;
        height: 0.8rem;
        overflow: hidden;
    }

    .model-bar-fill {
        border-radius: 999px;
        height: 100%;
        min-width: 4%;
    }

    .model-bar-fill.suspicious {
        background: #d92d20;
    }

    .model-bar-fill.okay {
        background: #0f766e;
    }

    .model-bar-fill.none {
        background: #98a2b3;
    }

    .scan-summary {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--paper);
        padding: 1rem;
        margin: 0.9rem 0;
    }

    .scan-summary-title {
        color: var(--ink);
        font-size: 1.1rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .scan-flag {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--paper);
        padding: 0.95rem;
        margin: 0.8rem 0;
    }

    .scan-snippet {
        color: var(--ink);
        font-size: 0.96rem;
        line-height: 1.45;
        margin: 0.55rem 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <h1>Dark Pattern Recognition</h1>
        <p>
            Test e-commerce copy against multiple text classifiers and compare
            whether they flag the snippet as potentially manipulative.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training comparison models...")
def get_comparison_models():
    """Train all demo classifiers once per Streamlit session."""
    df = load_primary_binary_dataset()
    trained_models = {}
    for name in model_names():
        pipeline = make_pipeline(name)
        pipeline.fit(df["text"], df["label"])
        trained_models[name] = pipeline
    return trained_models


def format_confidence(confidence: float | None) -> str:
    """Format optional model confidence for display."""
    if confidence is None:
        return "No score"
    return f"{confidence:.1%}"


def friendly_prediction(label_name: str) -> str:
    """Convert model labels into plain-language output."""
    if label_name == "Dark Pattern":
        return "Looks suspicious"
    return "Looks okay"


def confidence_explanation(confidence: float | None) -> str:
    """Explain confidence in plain language."""
    if confidence is None:
        return "No probability score"
    if confidence >= 0.85:
        return "Strong signal"
    if confidence >= 0.65:
        return "Moderate signal"
    return "Weak signal"


def confidence_detail(confidence: float | None) -> str:
    """Explain exactly what the signal value means."""
    if confidence is None:
        return "This model does not report a percent probability."
    return f"{confidence:.1%} probability for this answer"


def infer_dark_pattern_type(text: str) -> str:
    """Infer a likely category, even during Streamlit stale-module reloads."""
    helper = getattr(demo_filters, "infer_dark_pattern_type", None)
    if helper is not None:
        return helper(text)

    matches = []
    if re.search(
        r"hurry|last chance|limited time|limited offer|ends soon|ends tonight|"
        r"sale ends|deal ends|expires|act now",
        text,
        re.IGNORECASE,
    ):
        matches.append("Urgency")
    if re.search(
        r"almost gone|selling fast|only\s+\d+\s+(left|remaining)|"
        r"\d+\s+(left|remaining)\s+in stock|low stock|while supplies last",
        text,
        re.IGNORECASE,
    ):
        matches.append("Scarcity")
    if re.search(
        r"\d+[\d,]*\s+(people|customers|users|shoppers|visitors|members)\b|"
        r"people are (looking|viewing|watching)|someone just bought|"
        r"popular|trending|best[- ]?seller|reviews?|ratings?|deals bought",
        text,
        re.IGNORECASE,
    ):
        matches.append("Social proof")
    if re.search(
        r"(no thanks|i don't want|i do not want|not interested).*"
        r"(save|deal|offer|discount|money)",
        text,
        re.IGNORECASE,
    ):
        matches.append("Confirmshaming")

    if not matches:
        return "Unclear from text alone"
    return " + ".join(matches)


@st.cache_resource(show_spinner="Loading category model...")
def get_category_model():
    """Load or train the second-stage dark-pattern category model."""
    return get_or_train_category_model()


def possible_type_for_text(text: str) -> str:
    """Return category-model prediction with rule-based fallback."""
    try:
        category, confidence = predict_dark_pattern_category(text, get_category_model())
    except Exception:
        return infer_dark_pattern_type(text)

    if confidence is None:
        return category
    return f"{category} ({confidence:.1%})"


def model_result_rows(text: str, *, apply_filters: bool) -> list[dict[str, object]]:
    """Run the entered text through every configured model."""
    rows = []
    for name, pipeline in get_comparison_models().items():
        if apply_filters:
            prediction = predict_text_for_demo(text, pipeline)
            filter_note = (
                "Adjusted by a demo false-positive filter"
                if prediction.suppressed_by_filter
                else "No adjustment"
            )
        else:
            prediction = predict_text(text, pipeline)
            filter_note = "Filters off"
        rows.append(
            {
                "model": name,
                "prediction": prediction.label_name,
                "friendly_prediction": friendly_prediction(prediction.label_name),
                "confidence": prediction.confidence,
                "confidence_label": format_confidence(prediction.confidence),
                "confidence_text": confidence_explanation(prediction.confidence),
                "confidence_detail": confidence_detail(prediction.confidence),
                "pattern_type": possible_type_for_text(text)
                if prediction.label == 1
                else "Not flagged",
                "filter": filter_note,
            }
        )
    return rows


def render_main_result(prediction) -> None:
    """Render the main prediction in plain language."""
    if prediction.suppressed_by_filter:
        title = "Probably okay"
        body = (
            "The first pass looked suspicious, but a demo filter recognized "
            "ordinary webpage text without pressure directed at the shopper."
        )
        st.warning(f"**Main answer: {title}**\n\n{body}")
    elif prediction.label == 1:
        title = "Looks suspicious"
        body = (
            "This text may be using pressure, urgency, scarcity, or social proof "
            "to influence a shopper."
        )
        st.error(f"**Main answer: {title}**\n\n{body}")
    else:
        title = "Looks okay"
        body = (
            "This text was not flagged by the main checker. A full website review "
            "could still reveal issues outside the text."
        )
        st.success(f"**Main answer: {title}**\n\n{body}")

    confidence = format_confidence(prediction.confidence)
    confidence_text = confidence_explanation(prediction.confidence)
    confidence_detail_text = confidence_detail(prediction.confidence)
    if prediction.confidence is None:
        st.caption(f"Signal strength: {confidence_text}. {confidence_detail_text}")
    else:
        st.caption(
            f"Signal strength: {confidence_text}. "
            f"{confidence_detail_text} ({confidence})."
        )
    if prediction.label == 1:
        st.caption(f"Possible type: {possible_type_for_text(prediction.text)}")


def render_checker_details(rows: list[dict[str, object]]) -> None:
    """Render model results as an easy-to-scan comparison chart."""
    st.caption("Each bar shows how confident that model was in its own answer.")
    for row in rows:
        confidence = row["confidence"]
        width = 8 if confidence is None else max(4, round(float(confidence) * 100))
        bar_class = "none"
        if row["prediction"] == "Dark Pattern":
            bar_class = "suspicious"
        elif row["prediction"] == "Not Dark Pattern":
            bar_class = "okay"
        st.markdown(
            f"""
            <div class="model-row">
                <div class="model-row-label">
                    <span>{row["model"]}</span>
                    <span class="model-result">{row["friendly_prediction"]}</span>
                </div>
                <div class="model-bar-track">
                    <div class="model-bar-fill {bar_class}" style="width: {width}%"></div>
                </div>
                <div class="small-note">{row["confidence_text"]} - {row["confidence_detail"]}</div>
                <div class="small-note">Possible type: {row["pattern_type"]}</div>
                <div class="small-note">{row["filter"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def scan_url(
    url: str,
    *,
    threshold: float,
    limit: int,
    wait_ms: int,
    apply_filters: bool,
    hide_context_light: bool,
) -> tuple[list[str], list]:
    """Scan a webpage and return all reviewed snippets plus displayed results."""
    model = get_or_train_model()
    page_text = extract_visible_text_with_playwright(url, wait_ms=wait_ms)
    snippets = split_visible_text(page_text)
    results = score_snippets(
        snippets,
        model,
        dark_only=True,
        min_confidence=threshold,
        suppress_simple_discounts=apply_filters,
        suppress_product_titles=apply_filters,
        suppress_context_light=apply_filters and hide_context_light,
        suppress_benign_context=apply_filters,
    )
    return snippets, results[:limit]


def scan_url_with_progress(
    url: str,
    *,
    threshold: float,
    limit: int,
    wait_ms: int,
    apply_filters: bool,
    hide_context_light: bool,
    progress,
    status,
) -> tuple[list[str], list]:
    """Scan a webpage while updating Streamlit progress UI."""
    status.write("Loading the trained model...")
    progress.progress(10)
    model = get_or_train_model()

    status.write("Opening the page with Playwright...")
    progress.progress(30)
    page_text = extract_visible_text_with_playwright(url, wait_ms=wait_ms)

    status.write("Splitting visible text into short snippets...")
    progress.progress(60)
    snippets = split_visible_text(page_text)

    status.write("Scoring snippets with the model and applying filters...")
    progress.progress(80)
    results = score_snippets(
        snippets,
        model,
        dark_only=True,
        min_confidence=threshold,
        suppress_simple_discounts=apply_filters,
        suppress_product_titles=apply_filters,
        suppress_context_light=apply_filters and hide_context_light,
        suppress_benign_context=apply_filters,
    )

    status.write("Done.")
    progress.progress(100)
    return snippets, results[:limit]


def render_scan_results(url: str, snippets: list[str], results: list) -> None:
    """Render webpage scan results in the app."""
    safe_url = html.escape(url)
    st.markdown(
        f"""
        <section class="scan-summary">
            <div class="eyebrow">Webpage scan</div>
            <div class="scan-summary-title">{safe_url}</div>
            <p class="result-copy">
                The scanner reviewed visible page text and flagged snippets that
                looked suspicious above your probability threshold.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Snippets reviewed", len(snippets))
    metric_cols[1].metric("Snippets shown", len(results))
    metric_cols[2].metric("Review needed", "Yes" if results else "No")

    if not results:
        st.success(
            "No high-confidence suspicious snippets were found with these settings."
        )
        return

    st.warning(
        "These are text-only flags. They are leads for review, not final proof of a dark pattern."
    )
    st.subheader("Flagged snippets")
    for index, result in enumerate(results, start=1):
        confidence = result.prediction.confidence
        confidence_text = "No probability score" if confidence is None else f"{confidence:.1%}"
        confidence_label = confidence_explanation(confidence)
        confidence_detail_text = confidence_detail(confidence)
        pattern_type = html.escape(possible_type_for_text(result.snippet))
        width = 8 if confidence is None else max(4, round(float(confidence) * 100))
        safe_snippet = html.escape(result.snippet)
        st.markdown(
            f"""
            <section class="scan-flag">
                <div class="model-row-label">
                    <span>Flag {index}: Looks suspicious</span>
                    <span class="model-result">{confidence_text}</span>
                </div>
                <div class="model-bar-track">
                    <div class="model-bar-fill suspicious" style="width: {width}%"></div>
                </div>
                <div class="small-note">{confidence_label} - {confidence_detail_text}</div>
                <div class="small-note">Possible type: {pattern_type}</div>
                <p class="scan-snippet">{safe_snippet}</p>
            </section>
            """,
            unsafe_allow_html=True,
        )


examples = {
    "Urgency example": "Hurry! Sale ends in 10 minutes. Buy now before prices go up.",
    "Scarcity example": "Only 2 left in stock. Add to cart before this item sells out.",
    "Social proof example": "1,243 people are looking at this item right now.",
    "Filter example": "Previous price: $99.00 47% off",
    "Catalog title example": (
        "2025 Tablet 10 inch Android 14 Tablet 8+64GB 1280x800 IPS "
        "Touchscreen 5000mAh US"
    ),
    "Neutral example": "This cotton pillowcase is machine washable and available in two sizes.",
}
webpage_examples = [
    "https://www.ulta.com/promotion/sale",
    "https://toonbee.ai/",
    "https://www.cnn.com/",
    "https://www.udemy.com/course/c-in-1-hour/",
    "https://www.booking.com/reviews/mx/hotel/amp-hostal-nojoch-che-coba.en-gb.html",
    "https://usercentrics.com/knowledge-hub/dark-patterns-and-how-they-affect-consent/",
]

text_tab, scanner_tab = st.tabs(["Analyze text", "Scan webpage"])

with text_tab:
    input_col, option_col = st.columns([2.2, 1], gap="large")

    with input_col:
        if "website_text" not in st.session_state:
            st.session_state["website_text"] = ""

        if "selected_example_text" in st.session_state:
            st.session_state["website_text"] = st.session_state.pop("selected_example_text")

        st.write("**Text**")
        text = st.text_area(
            "Text",
            key="website_text",
            height=170,
            label_visibility="collapsed",
            placeholder=(
                "Paste e-commerce text here, such as a product banner, "
                "checkout message, or sale notice."
            ),
        )

        st.caption("Examples to try")
        example_columns = st.columns(3)
        for index, (label, example_text) in enumerate(examples.items()):
            with example_columns[index % 3]:
                if st.button(label, use_container_width=True):
                    st.session_state["selected_example_text"] = example_text
                    st.rerun()

    with option_col:
        st.subheader("Run check")
        st.write("The app will show the main answer and each model's result.")
        apply_text_filters = st.checkbox(
            "Apply demo filters",
            value=True,
            key="apply_text_filters",
        )
        st.caption(
            "Filters hide obvious false positives like plain discounts, product specs, "
            "metadata, status messages, and educational examples. "
            "Turn off to see raw model output."
        )
        analyze = st.button("Analyze text", type="primary", use_container_width=True)

    if analyze:
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            model = get_or_train_model()
            st.session_state["last_text"] = text
            st.session_state["last_prediction"] = (
                predict_text_for_demo(text, model)
                if apply_text_filters
                else predict_text(text, model)
            )
            st.session_state["last_rows"] = model_result_rows(
                text,
                apply_filters=apply_text_filters,
            )

    if "last_prediction" in st.session_state and "last_rows" in st.session_state:
        prediction = st.session_state["last_prediction"]
        rows = st.session_state["last_rows"]

        render_main_result(prediction)
        st.subheader("Model comparison")
        render_checker_details(rows)

with scanner_tab:
    scan_col, settings_col = st.columns([2.2, 1], gap="large")

    with scan_col:
        if "webpage_url" not in st.session_state:
            st.session_state["webpage_url"] = ""

        if "selected_example_url" in st.session_state:
            st.session_state["webpage_url"] = st.session_state.pop("selected_example_url")

        st.write("**Webpage URL**")
        url = st.text_input(
            "Webpage URL",
            key="webpage_url",
            label_visibility="collapsed",
        )
        st.caption(
            "This scans visible page text only. It cannot judge layout, button styling, "
            "checkout friction, images, or the full user flow."
        )
        st.caption("Examples to try")
        url_example_columns = st.columns(3)
        for index, example_url in enumerate(webpage_examples):
            with url_example_columns[index % 3]:
                if st.button(example_url, use_container_width=True):
                    st.session_state["selected_example_url"] = example_url
                    st.rerun()

    with settings_col:
        st.subheader("Scan settings")
        threshold = st.slider("Minimum probability", 0.50, 0.95, 0.80, 0.05)
        st.caption(
            "Only show snippets when the model is at least this confident. "
            "Higher means fewer, stronger flags."
        )
        limit = st.slider("Maximum flags shown", 3, 20, 8, 1)
        st.caption(
            "Caps how many suspicious snippets appear in the results so the page stays readable."
        )
        wait_ms = st.slider("Page wait time", 1000, 8000, 3000, 500)
        st.caption(
            "How long Playwright waits after opening the page. Longer can catch popups "
            "or delayed content, but scanning takes more time."
        )
        apply_scan_filters = st.checkbox(
            "Apply demo filters",
            value=True,
            key="apply_scan_filters",
        )
        st.caption(
            "Filters hide plain discounts, product/spec text, benign comparisons, "
            "metadata, status messages, educational examples, and vague fragments. "
            "Turn off to inspect raw scanner output."
        )
        hide_context_light = True
        if apply_scan_filters:
            hide_context_light = st.checkbox(
                "Hide vague short snippets",
                value=True,
                key="hide_context_light",
            )
            st.caption(
                "Hides weak fragments like testimonials, bare counters, or tiny headings "
                "unless they include real urgency or scarcity language."
            )
        scan = st.button("Scan webpage", type="primary", use_container_width=True)

    if scan:
        if not url.strip():
            st.warning("Enter a webpage URL first.")
        else:
            try:
                st.info(
                    "Scanning can take a few seconds while the app opens the page. "
                    "If browser scanning stalls, it falls back to plain page text."
                )
                progress = st.progress(0)
                status = st.empty()
                snippets, results = scan_url_with_progress(
                    url.strip(),
                    threshold=threshold,
                    limit=limit,
                    wait_ms=wait_ms,
                    apply_filters=apply_scan_filters,
                    hide_context_light=hide_context_light,
                    progress=progress,
                    status=status,
                )
                st.session_state["last_scan_url"] = url.strip()
                st.session_state["last_scan_snippets"] = snippets
                st.session_state["last_scan_results"] = results
            except RuntimeError as exc:
                st.error(str(exc))

    if (
        "last_scan_url" in st.session_state
        and "last_scan_snippets" in st.session_state
        and "last_scan_results" in st.session_state
    ):
        render_scan_results(
            st.session_state["last_scan_url"],
            st.session_state["last_scan_snippets"],
            st.session_state["last_scan_results"],
        )

st.divider()
st.caption(
    "Model: TF-IDF + Logistic Regression trained on a balanced dark-pattern text dataset. "
    "Source code: https://github.com/RadRebelSam/ai4all-20c-darkpatterns"
)
