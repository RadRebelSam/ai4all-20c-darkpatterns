"""Dataset loading utilities for dark-pattern text classification."""

from pathlib import Path

import pandas as pd

DATASET_DIR = Path("datasets")

PRIMARY_BINARY_FILE = "krishuppal - dark-patterns.csv"
ADARSH_CATEGORY_FILE = "adarshm09 - dark-pattern-dataset.csv"
DEVITACHI_FILE = "devitachi - dark-pattern.csv"
MOHIT_FILE = "mohitsharma527 - dark-patterns-on-ecommerce-platforms.csv"
AKASH_CLASSIFICATION_FILE = "- pattern_classifications.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
CATEGORY_COLUMN = "category"
SOURCE_COLUMN = "source"

NOT_DARK_PATTERN = "Not Dark Pattern"


def dataset_path(filename: str) -> Path:
    """Return a dataset path and fail clearly when it is missing."""
    path = DATASET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def normalize_text(value: object) -> str:
    """Normalize text values from CSV/XLSX files into clean strings."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _drop_empty_text(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[TEXT_COLUMN].str.len() > 0].reset_index(drop=True)


def load_primary_binary_dataset() -> pd.DataFrame:
    """Load the balanced binary dark-pattern dataset.

    Returns:
        DataFrame with `text`, `label`, `category`, and `source`.
        `label` is 1 for dark pattern and 0 for not dark pattern.
    """
    raw = pd.read_csv(dataset_path(PRIMARY_BINARY_FILE))
    df = pd.DataFrame(
        {
            TEXT_COLUMN: raw["text"].map(normalize_text),
            LABEL_COLUMN: raw["label"].astype(int),
            CATEGORY_COLUMN: raw["Pattern Category"].map(normalize_text),
            SOURCE_COLUMN: "krishuppal",
        }
    )
    return _drop_empty_text(df)


def load_adarsh_category_dataset() -> pd.DataFrame:
    """Load the category-labeled dark-pattern dataset."""
    raw = pd.read_csv(dataset_path(ADARSH_CATEGORY_FILE))
    category = raw["Category"].map(normalize_text)
    df = pd.DataFrame(
        {
            TEXT_COLUMN: raw["Title"].map(normalize_text),
            LABEL_COLUMN: (category != NOT_DARK_PATTERN).astype(int),
            CATEGORY_COLUMN: category,
            SOURCE_COLUMN: "adarshm09",
        }
    )
    return _drop_empty_text(df)


def load_ecommerce_patterns_dataset(filename: str, source_name: str) -> pd.DataFrame:
    """Load an e-commerce dark-pattern source with pattern metadata."""
    raw = pd.read_csv(dataset_path(filename))
    category = raw["Pattern Category"].map(normalize_text)
    df = pd.DataFrame(
        {
            TEXT_COLUMN: raw["Pattern String"].map(normalize_text),
            LABEL_COLUMN: (category != NOT_DARK_PATTERN).astype(int),
            CATEGORY_COLUMN: category,
            "pattern_type": raw["Pattern Type"].map(normalize_text),
            "page_location": raw["Where in website?"].map(normalize_text),
            "deceptive_flag": raw["Deceptive?"].map(normalize_text),
            SOURCE_COLUMN: source_name,
        }
    )
    return _drop_empty_text(df)


def load_devitachi_binary_dataset() -> pd.DataFrame:
    """Load Devitachi examples as positive dark-pattern text rows.

    The Devitachi file contains dark-pattern instances with categories and pattern
    metadata. Its `Deceptive?` column is not equivalent to the project's binary
    dark-pattern label, so these rows are used as additional positive examples.
    """
    df = load_ecommerce_patterns_dataset(DEVITACHI_FILE, "devitachi")
    df[LABEL_COLUMN] = 1
    return df


def load_akash_binary_dataset() -> pd.DataFrame:
    """Load Akash Nath's binary pattern classification rows."""
    raw = pd.read_csv(dataset_path(AKASH_CLASSIFICATION_FILE))
    raw = raw.dropna(subset=["Pattern String", "classification"]).copy()
    raw["classification"] = raw["classification"].astype(int)
    category = raw["classification"].map({0: NOT_DARK_PATTERN, 1: "Dark Pattern"})
    df = pd.DataFrame(
        {
            TEXT_COLUMN: raw["Pattern String"].map(normalize_text),
            LABEL_COLUMN: raw["classification"],
            CATEGORY_COLUMN: category,
            SOURCE_COLUMN: "akashnath29",
        }
    )
    return _drop_empty_text(df)


def load_expanded_binary_dataset() -> pd.DataFrame:
    """Load primary data plus Devitachi and Akash Nath supplemental rows."""
    frames = [
        load_primary_binary_dataset(),
        load_devitachi_binary_dataset()[
            [TEXT_COLUMN, LABEL_COLUMN, CATEGORY_COLUMN, SOURCE_COLUMN]
        ],
        load_akash_binary_dataset(),
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return deduplicate_text_with_source_history(combined)


def load_secondary_category_datasets() -> pd.DataFrame:
    """Load compatible category datasets and remove exact duplicate text rows."""
    frames = [
        load_adarsh_category_dataset(),
        load_devitachi_binary_dataset(),
        load_ecommerce_patterns_dataset(MOHIT_FILE, "mohitsharma527"),
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return deduplicate_text(combined)


def deduplicate_text(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate text rows while keeping the first source."""
    return df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)


def deduplicate_text_with_source_history(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate text rows while preserving all dataset source names."""
    rows = []
    for _, group in df.groupby(TEXT_COLUMN, sort=False):
        row = group.iloc[0].copy()
        row[SOURCE_COLUMN] = "; ".join(sorted(set(group[SOURCE_COLUMN].astype(str))))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def validate_binary_dataset(df: pd.DataFrame) -> None:
    """Validate the minimum expectations for binary text classification."""
    required = {TEXT_COLUMN, LABEL_COLUMN, CATEGORY_COLUMN, SOURCE_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if df[TEXT_COLUMN].isna().any() or (df[TEXT_COLUMN].str.len() == 0).any():
        raise ValueError("Dataset contains empty text values")
    labels = set(df[LABEL_COLUMN].unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected binary labels {{0, 1}}, found: {labels}")


def summarize_labels(df: pd.DataFrame, column: str = LABEL_COLUMN) -> dict[str, int]:
    """Return sorted value counts as a plain dictionary."""
    counts = df[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}
