"""Generate EDA-only visualizations that do not require model training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import (
    CATEGORY_COLUMN,
    LABEL_COLUMN,
    NOT_DARK_PATTERN,
    SOURCE_COLUMN,
    TEXT_COLUMN,
    load_akash_binary_dataset,
    load_devitachi_binary_dataset,
    load_expanded_binary_dataset,
    load_primary_binary_dataset,
)

REPORT_DIR = Path("reports") / "figures" / "eda"
DOCS_DIR = Path("docs") / "figures" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

SAVE_KW = dict(dpi=160, bbox_inches="tight")
PALETTE = sns.color_palette("Set2", 8)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


def save_to_both(fig: plt.Figure, filename: str) -> None:
    report_path = REPORT_DIR / filename
    docs_path = DOCS_DIR / filename
    fig.savefig(report_path, **SAVE_KW)
    shutil.copy2(report_path, docs_path)
    print(f"  Saved {report_path} and {docs_path}")


def plot_class_balance() -> None:
    df = load_primary_binary_dataset()
    counts = (
        df[LABEL_COLUMN]
        .map({0: "Not Dark Pattern", 1: "Dark Pattern"})
        .value_counts()
        .reindex(["Not Dark Pattern", "Dark Pattern"])
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[7], PALETTE[1]])
    ax.set_title("Primary Dataset Class Balance", fontsize=15, fontweight="bold")
    ax.set_ylabel("Number of Text Snippets")
    ax.set_ylim(0, max(counts.values) * 1.18)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 20,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout()
    save_to_both(fig, "eda_class_balance.png")
    plt.close(fig)


def plot_dark_category_distribution() -> None:
    df = load_primary_binary_dataset()
    dark = df[df[CATEGORY_COLUMN] != NOT_DARK_PATTERN]
    counts = dark[CATEGORY_COLUMN].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(counts.index, counts.values, color=PALETTE[0])
    ax.set_title("Dark Pattern Category Distribution", fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of Dark-Pattern Examples")
    for bar, value in zip(bars, counts.values):
        ax.text(value + 5, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    fig.tight_layout()
    save_to_both(fig, "eda_dark_category_distribution.png")
    plt.close(fig)


def plot_text_length_distribution() -> None:
    df = load_primary_binary_dataset().copy()
    df["word_count"] = df[TEXT_COLUMN].str.split().str.len()
    df["label_name"] = df[LABEL_COLUMN].map({0: "Not Dark Pattern", 1: "Dark Pattern"})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(
        data=df,
        x="word_count",
        hue="label_name",
        bins=30,
        multiple="layer",
        alpha=0.6,
        palette=[PALETTE[7], PALETTE[1]],
        ax=ax,
    )
    ax.set_xlim(0, min(60, df["word_count"].quantile(0.98)))
    ax.set_title("Text Snippet Length Distribution", fontsize=15, fontweight="bold")
    ax.set_xlabel("Words per Text Snippet")
    ax.set_ylabel("Number of Snippets")
    fig.tight_layout()
    save_to_both(fig, "eda_text_length_distribution.png")
    plt.close(fig)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def plot_common_dark_words() -> None:
    df = load_primary_binary_dataset()
    dark_text = " ".join(df[df[LABEL_COLUMN] == 1][TEXT_COLUMN])
    counts = Counter(tokenize(dark_text)).most_common(20)
    words = [word for word, _ in counts][::-1]
    values = [value for _, value in counts][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(words, values, color=PALETTE[2])
    ax.set_title("Most Common Words in Dark-Pattern Text", fontsize=15, fontweight="bold")
    ax.set_xlabel("Word Count")
    for bar, value in zip(bars, values):
        ax.text(value + 3, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    fig.tight_layout()
    save_to_both(fig, "eda_common_dark_words.png")
    plt.close(fig)


def plot_source_contribution() -> None:
    primary = load_primary_binary_dataset()
    devitachi = load_devitachi_binary_dataset()
    akash = load_akash_binary_dataset()
    expanded = load_expanded_binary_dataset()

    counts = pd.Series(
        {
            "Krish Uppal\nprimary rows": len(primary),
            "Devitachi\nusable rows": len(devitachi),
            "Akash Nath\nusable rows": len(akash),
            "Expanded\nunique rows": len(expanded),
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[1], PALETTE[0], PALETTE[2], PALETTE[4]])
    ax.set_title("Dataset Source Contribution", fontsize=15, fontweight="bold")
    ax.set_ylabel("Rows")
    ax.set_ylim(0, max(counts.values) * 1.18)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 60,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    save_to_both(fig, "eda_source_contribution.png")
    plt.close(fig)


def plot_missing_data_overview() -> None:
    files = {
        "Krish Uppal": pd.read_csv("datasets/krishuppal - dark-patterns.csv"),
        "Devitachi": pd.read_csv("datasets/devitachi - dark-pattern.csv"),
        "Akash Nath classification": pd.read_csv("datasets/- pattern_classifications.csv"),
    }
    rows = []
    for source, df in files.items():
        rows.append(
            {
                "source": source,
                "total_cells": int(df.shape[0] * df.shape[1]),
                "missing_cells": int(df.isna().sum().sum()),
                "missing_percent": float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            }
        )
    overview = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(overview["source"], overview["missing_percent"], color=PALETTE[3])
    ax.set_title("Missing Data Overview by Source", fontsize=15, fontweight="bold")
    ax.set_ylabel("Missing Cells (%)")
    ax.set_ylim(0, max(1, overview["missing_percent"].max() * 1.25))
    for bar, row in zip(bars, overview.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row.missing_percent + 0.05,
            f"{row.missing_percent:.1f}%\n({row.missing_cells:,} cells)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    save_to_both(fig, "eda_missing_data_overview.png")
    plt.close(fig)


def main() -> None:
    print("Generating EDA-only visualizations...")
    plot_class_balance()
    plot_dark_category_distribution()
    plot_text_length_distribution()
    plot_common_dark_words()
    plot_source_contribution()
    plot_missing_data_overview()
    print("Done.")


if __name__ == "__main__":
    main()
