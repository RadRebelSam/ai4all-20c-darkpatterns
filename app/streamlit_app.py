"""Streamlit demo for dark-pattern recognition."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data import load_primary_binary_dataset
from src.modeling import make_pipeline, split_dataset
from src.predict import get_or_train_model, predict_text_for_demo

st.set_page_config(
    page_title="Dark Pattern Recognition",
    page_icon="!",
    layout="wide",
)

st.title("Dark Pattern Recognition")
st.write(
    "Enter e-commerce website text to check whether the model flags it as a dark pattern."
)

# ── Prediction panel ──────────────────────────────────────────────────────────

examples = {
    "Urgency example": "Hurry! Sale ends in 10 minutes. Buy now before prices go up.",
    "Scarcity example": "Only 2 left in stock. Add to cart before this item sells out.",
    "Social proof example": "1,243 people are looking at this item right now.",
    "Plain discount example": "Previous price: $99.00 47% off",
    "Catalog title example": "2025 Tablet 10 inch Android 14 Tablet 8+64GB 1280x800 IPS Touchscreen 5000mAh US",
    "Neutral example": "This cotton pillowcase is machine washable and available in two sizes.",
}

selected_example = st.selectbox("Example text", ["Custom"] + list(examples))
default_text = "" if selected_example == "Custom" else examples[selected_example]
text = st.text_area("Website text", value=default_text, height=150)

if st.button("Analyze text", type="primary"):
    if not text.strip():
        st.warning("Enter some website text first.")
    else:
        model = get_or_train_model()
        prediction = predict_text_for_demo(text, model)
        st.metric("Prediction", prediction.label_name)
        if prediction.confidence is not None:
            st.progress(prediction.confidence)
            st.caption(f"Model confidence: {prediction.confidence:.1%}")

        if prediction.suppressed_by_filter:
            st.warning(
                "The raw model considered this snippet suspicious, but the demo filter "
                "suppressed it because it looks like low-context sale or product-catalog "
                "text without extra urgency or scarcity pressure language."
            )
            st.caption(prediction.filter_reason)
        elif prediction.label == 1:
            st.info(
                "The model flagged this text as potentially manipulative. "
                "A human review is still needed before making a final judgment."
            )
        else:
            st.success(
                "The model did not flag this text as a dark pattern. "
                "This does not guarantee the full website is free of deceptive design."
            )

st.divider()

# ── Inline charts ─────────────────────────────────────────────────────────────

st.subheader("Model Performance")

tab1, tab2, tab3 = st.tabs(
    ["Model Comparison", "Category Distribution", "Top Features"]
)

with tab1:
    metrics_path = "reports/model_metrics.csv"
    try:
        metrics = pd.read_csv(metrics_path)
        metric_cols = ["accuracy", "precision", "recall", "f1"]
        fig, ax = plt.subplots(figsize=(9, 5))
        bar_height = 0.18
        y = np.arange(len(metrics))
        colors = plt.cm.Set2(np.linspace(0, 0.6, 4))
        for i, (col, color) in enumerate(zip(metric_cols, colors)):
            ax.barh(y + i * bar_height, metrics[col], height=bar_height,
                    label=col.capitalize(), color=color)
        ax.set_yticks(y + bar_height * 1.5)
        ax.set_yticklabels(metrics["model"])
        ax.set_xlim(0.80, 1.02)
        ax.set_xlabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except FileNotFoundError:
        st.info("Run `python scripts/train_models.py` to generate metrics.")

with tab2:
    try:
        df = load_primary_binary_dataset()
        counts = df["category"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(counts.index, counts.values, color=plt.cm.Set2(0.1))
        for i, v in enumerate(counts.values):
            ax.text(v + 3, i, str(v), va="center", fontsize=9)
        ax.set_xlabel("Number of Examples")
        ax.set_title("Dark Pattern Category Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.info(f"Could not load dataset: {e}")

with tab3:
    try:
        df = load_primary_binary_dataset()
        pipeline = make_pipeline("Logistic Regression")
        pipeline.fit(df["text"], df["label"])
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["classifier"]
        feature_names = np.array(tfidf.get_feature_names_out())
        coefs = clf.coef_[0]
        top_pos = np.argsort(coefs)[-12:]
        top_neg = np.argsort(coefs)[:12]
        idx = np.concatenate([top_neg, top_pos])
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = [plt.cm.Set2(0.5) if c < 0 else plt.cm.Set2(0.15) for c in coefs[idx]]
        ax.barh(range(len(idx)), coefs[idx], color=colors)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(feature_names[idx], fontsize=9)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Logistic Regression Coefficient")
        ax.set_title("Top TF-IDF Features for Dark Pattern Detection")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.info(f"Could not generate feature chart: {e}")

st.divider()
st.caption(
    "Model: TF-IDF + Logistic Regression trained on a balanced dark-pattern text dataset. "
    "Source code: https://github.com/RadRebelSam/ai4all-20c-darkpatterns"
)
