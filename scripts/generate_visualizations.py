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
from sklearn.metrics import confusion_matrix

from src.data import (
    CATEGORY_COLUMN,
    LABEL_COLUMN,
    NOT_DARK_PATTERN,
    TEXT_COLUMN,
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
    plot_model_comparison()
    plot_category_model_comparison()
    plot_confusion_matrix()
    plot_category_distribution()
    plot_top_features()
    print("Done - all figures saved to reports/figures/ and docs/figures/")


if __name__ == "__main__":
    main()
