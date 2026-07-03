"""Generate publication-quality visualizations for the dark-pattern project."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.data import (
    CATEGORY_COLUMN,
    LABEL_COLUMN,
    NOT_DARK_PATTERN,
    TEXT_COLUMN,
    load_dark_pattern_category_dataset,
    load_primary_binary_dataset,
)
from src.modeling import make_pipeline, split_dataset

# ---------------------------------------------------------------------------
# Style & output directory
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
FIGURE_DIRS = [Path("reports") / "figures", Path("docs") / "figures"]
for figure_dir in FIGURE_DIRS:
    figure_dir.mkdir(parents=True, exist_ok=True)

SAVE_KW = dict(dpi=150, bbox_inches="tight")

# Colour palette
PALETTE = sns.color_palette("Set2", 8)


def save_figure(fig, filename: str) -> None:
    """Save one figure to report and GitHub Pages figure directories."""
    for figure_dir in FIGURE_DIRS:
        fig.savefig(figure_dir / filename, **SAVE_KW)


# ---------------------------------------------------------------------------
# 1.  Model Performance Comparison  (horizontal bar chart)
# ---------------------------------------------------------------------------


def plot_model_comparison() -> None:
    metrics = pd.read_csv("reports/model_metrics.csv")

    metric_cols = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    colours = [PALETTE[i] for i in range(len(metric_cols))]

    models = metrics["model"].tolist()
    n_models = len(models)
    bar_height = 0.18
    y_positions = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (col, label, colour) in enumerate(zip(metric_cols, labels, colours)):
        offsets = y_positions + i * bar_height
        values = metrics[col].tolist()
        bars = ax.barh(offsets, values, height=bar_height, label=label, color=colour)
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=8,
            )

    ax.set_yticks(y_positions + bar_height * 1.5)
    ax.set_yticklabels(models)
    ax.set_xlim(0.80, 1.02)
    ax.set_xlabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    save_figure(fig, "model_comparison.png")
    plt.close(fig)
    print("  Saved model_comparison.png")


def plot_category_model_comparison() -> None:
    """Plot the second-stage category model comparison."""
    metrics_path = Path("reports") / "category_model_metrics.csv"
    if not metrics_path.exists():
        print("  Skipped category_model_comparison.png (missing category metrics)")
        return

    metrics = pd.read_csv(metrics_path)
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    colours = [PALETTE[i] for i in range(len(metric_cols))]

    models = metrics["model"].tolist()
    bar_height = 0.18
    y_positions = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (col, label, colour) in enumerate(zip(metric_cols, labels, colours)):
        offsets = y_positions + i * bar_height
        values = metrics[col].tolist()
        bars = ax.barh(offsets, values, height=bar_height, label=label, color=colour)
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=8,
            )

    ax.set_yticks(y_positions + bar_height * 1.5)
    ax.set_yticklabels(models)
    ax.set_xlim(0.55, 1.02)
    ax.set_xlabel("Score")
    ax.set_title(
        "Second-Stage Category Model Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    save_figure(fig, "category_model_comparison.png")
    plt.close(fig)
    print("  Saved category_model_comparison.png")


def plot_two_stage_pipeline() -> None:
    """Show the current binary-then-category model flow."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    boxes = [
        (0.04, 0.56, 0.20, 0.24, "Website text", "Snippet from app\\nor webpage scan", PALETTE[2]),
        (0.31, 0.56, 0.22, 0.24, "Model 1", "Binary detector\\nDark vs Not Dark", PALETTE[0]),
        (0.61, 0.70, 0.27, 0.20, "Not Dark Pattern", "Stop: no type needed", PALETTE[7]),
        (0.61, 0.37, 0.27, 0.22, "Model 2", "Category classifier\\nfor suspicious text", PALETTE[1]),
        (0.38, 0.08, 0.50, 0.18, "Likely type", "Urgency, Scarcity, Social Proof, Misdirection, Obstruction, Sneaking, Forced Action", PALETTE[4]),
    ]

    for x, y, w, h, title, subtitle, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.5,
            edgecolor="#2f3437",
            facecolor=color,
            alpha=0.9,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.35, subtitle, ha="center", va="center", fontsize=10)

    arrow_kw = dict(arrowstyle="->", color="#2f3437", lw=2, shrinkA=4, shrinkB=4)
    ax.annotate("", xy=(0.31, 0.68), xytext=(0.24, 0.68), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.61, 0.79), xytext=(0.53, 0.72), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.61, 0.49), xytext=(0.53, 0.64), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.63, 0.26), xytext=(0.74, 0.37), arrowprops=arrow_kw)

    ax.text(0.56, 0.82, "if not suspicious", fontsize=9, color="#475569")
    ax.text(0.54, 0.53, "if suspicious", fontsize=9, color="#475569")
    ax.set_title("Two-Stage Dark Pattern Recognition Pipeline", fontsize=16, fontweight="bold", pad=12)

    save_figure(fig, "two_stage_pipeline.png")
    plt.close(fig)
    print("  Saved two_stage_pipeline.png")


# ---------------------------------------------------------------------------
# 2.  Confusion Matrix  (Logistic Regression, 80/20 split)
# ---------------------------------------------------------------------------


def plot_confusion_matrix() -> None:
    df = load_primary_binary_dataset()
    x_train, x_test, y_train, y_test = split_dataset(df)

    pipeline = make_pipeline("Logistic Regression")
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    class_labels = ["Not Dark Pattern", "Dark Pattern"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix - Logistic Regression", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, "confusion_matrix.png")
    plt.close(fig)
    print("  Saved confusion_matrix.png")


def plot_category_confusion_matrix() -> None:
    """Plot the second-stage category model confusion matrix."""
    df = load_dark_pattern_category_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN],
        df[CATEGORY_COLUMN],
        test_size=0.2,
        stratify=df[CATEGORY_COLUMN],
        random_state=42,
    )

    pipeline = make_pipeline("Linear SVM")
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Category Confusion Matrix - Linear SVM", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    plt.tight_layout()
    save_figure(fig, "category_confusion_matrix.png")
    plt.close(fig)
    print("  Saved category_confusion_matrix.png")


def plot_category_per_class_f1() -> None:
    """Plot per-category F1 scores for the best second-stage model."""
    df = load_dark_pattern_category_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN],
        df[CATEGORY_COLUMN],
        test_size=0.2,
        stratify=df[CATEGORY_COLUMN],
        random_state=42,
    )

    pipeline = make_pipeline("Linear SVM")
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    labels = sorted(y_test.unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    scores = pd.DataFrame(
        {
            "category": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).sort_values("f1")

    fig, ax = plt.subplots(figsize=(12, 7.5))
    bars = ax.barh(scores["category"], scores["f1"], color=PALETTE[1], height=0.62)
    for bar, val, count in zip(bars, scores["f1"], scores["support"]):
        x_pos = min(val + 0.018, 1.01)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"F1 {val:.2f}  |  test n={count}",
            va="center",
            fontsize=10,
            color="#334155",
        )
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Second-Stage Model F1 by Dark Pattern Type", fontsize=16, fontweight="bold")
    ax.text(
        0,
        -0.9,
        "Lower scores often reflect fewer training examples or overlap between similar categories.",
        fontsize=10,
        color="#475569",
    )

    plt.tight_layout()
    save_figure(fig, "category_per_class_f1.png")
    plt.close(fig)
    print("  Saved category_per_class_f1.png")


def plot_model_summary_card() -> None:
    """Create a compact visual comparing the two trained models."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    cards = [
        (
            0.06,
            "Model 1",
            "Binary detector",
            "TF-IDF + Logistic Regression",
            "F1 = 0.936",
            "Answers: Dark Pattern or Not Dark Pattern",
            PALETTE[0],
        ),
        (
            0.54,
            "Model 2",
            "Type classifier",
            "TF-IDF + Linear SVM",
            "Macro F1 = 0.894",
            "Answers: likely dark-pattern category",
            PALETTE[1],
        ),
    ]

    for x, label, title, model, score, note, color in cards:
        patch = FancyBboxPatch(
            (x, 0.18),
            0.40,
            0.62,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.2,
            edgecolor="#cbd5e1",
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + 0.04, 0.68, label, fontsize=12, color="#64748b", fontweight="bold")
        ax.text(x + 0.04, 0.58, title, fontsize=20, color="#0f172a", fontweight="bold")
        ax.text(x + 0.04, 0.47, model, fontsize=12, color="#334155")
        ax.text(x + 0.04, 0.35, score, fontsize=24, color=color, fontweight="bold")
        ax.text(x + 0.04, 0.26, note, fontsize=11, color="#475569")

    ax.annotate("", xy=(0.53, 0.49), xytext=(0.46, 0.49), arrowprops=dict(arrowstyle="->", lw=2, color="#334155"))
    ax.text(0.50, 0.55, "if suspicious", ha="center", fontsize=9, color="#64748b")
    ax.set_title("Two Trained Models, Two Different Jobs", fontsize=17, fontweight="bold", pad=10)

    save_figure(fig, "two_model_summary_card.png")
    plt.close(fig)
    print("  Saved two_model_summary_card.png")


def plot_example_result_flow() -> None:
    """Visualize one example moving through both models."""
    example = "Only 2 left in stock. Order now before it sells out."
    binary_model = make_pipeline("Logistic Regression")
    binary_df = load_primary_binary_dataset()
    binary_model.fit(binary_df[TEXT_COLUMN], binary_df[LABEL_COLUMN])
    label = int(binary_model.predict([example])[0])
    prob = binary_model.predict_proba([example])[0][label]

    category_df = load_dark_pattern_category_dataset()
    category_model = make_pipeline("Linear SVM")
    category_model.fit(category_df[TEXT_COLUMN], category_df[CATEGORY_COLUMN])
    category = category_model.predict([example])[0]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    steps = [
        ("Input text", f'"{example}"'),
        ("Model 1 result", f"Dark Pattern\\nconfidence {prob:.1%}"),
        ("Model 2 result", f"Likely type\\n{category}"),
    ]
    xs = [0.06, 0.38, 0.70]
    colors = [PALETTE[2], PALETTE[0], PALETTE[1]]
    for x, (title, body), color in zip(xs, steps, colors):
        patch = FancyBboxPatch(
            (x, 0.30),
            0.24,
            0.38,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.3,
            edgecolor="#334155",
            facecolor=color,
            alpha=0.9,
        )
        ax.add_patch(patch)
        ax.text(x + 0.12, 0.57, title, ha="center", fontsize=13, fontweight="bold")
        ax.text(x + 0.12, 0.43, body, ha="center", va="center", fontsize=10)

    arrow_kw = dict(arrowstyle="->", lw=2, color="#334155", shrinkA=5, shrinkB=5)
    ax.annotate("", xy=(0.38, 0.49), xytext=(0.30, 0.49), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.70, 0.49), xytext=(0.62, 0.49), arrowprops=arrow_kw)
    ax.set_title("Example Result Flow Through Both Models", fontsize=16, fontweight="bold", pad=12)

    save_figure(fig, "example_result_flow.png")
    plt.close(fig)
    print("  Saved example_result_flow.png")


# ---------------------------------------------------------------------------
# 3.  Category Distribution  (horizontal bar chart)
# ---------------------------------------------------------------------------


def plot_category_distribution() -> None:
    df = load_primary_binary_dataset()
    counts = df[CATEGORY_COLUMN].value_counts().sort_values()

    colours = []
    for cat in counts.index:
        if cat == NOT_DARK_PATTERN:
            colours.append(PALETTE[7])  # distinct colour for non-dark-pattern
        else:
            colours.append(PALETTE[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(counts.index, counts.values, color=colours)

    for bar, val in zip(bars, counts.values):
        ax.text(
            val + 5,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Number of Examples")
    ax.set_title("Dark Pattern Category Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, "category_distribution.png")
    plt.close(fig)
    print("  Saved category_distribution.png")


# ---------------------------------------------------------------------------
# 4.  Top TF-IDF Features  (diverging horizontal bar chart)
# ---------------------------------------------------------------------------


def plot_top_features() -> None:
    df = load_primary_binary_dataset()

    pipeline = make_pipeline("Logistic Regression")
    pipeline.fit(df[TEXT_COLUMN], df[LABEL_COLUMN])

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]

    feature_names = np.array(tfidf.get_feature_names_out())
    coefficients = clf.coef_[0]

    top_pos_idx = np.argsort(coefficients)[-15:]
    top_neg_idx = np.argsort(coefficients)[:15]

    top_idx = np.concatenate([top_neg_idx, top_pos_idx])
    top_features = feature_names[top_idx]
    top_coefs = coefficients[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [PALETTE[3] if c < 0 else PALETTE[1] for c in top_coefs]
    ax.barh(range(len(top_features)), top_coefs, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title(
        "Top 15 TF-IDF Features for Dark Pattern Detection",
        fontsize=14,
        fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PALETTE[1], label="Dark Pattern"),
        Patch(facecolor=PALETTE[3], label="Not Dark Pattern"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    save_figure(fig, "top_features.png")
    plt.close(fig)
    print("  Saved top_features.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Generating visualizations...")
    plot_two_stage_pipeline()
    plot_model_summary_card()
    plot_example_result_flow()
    plot_model_comparison()
    plot_category_model_comparison()
    plot_confusion_matrix()
    plot_category_confusion_matrix()
    plot_category_per_class_f1()
    plot_category_distribution()
    plot_top_features()
    print("Done - all figures saved to reports/figures/ and docs/figures/")


if __name__ == "__main__":
    main()
