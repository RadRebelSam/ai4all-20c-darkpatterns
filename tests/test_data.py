import pandas as pd
import pytest

from src.data import (
    CATEGORY_COLUMN,
    LABEL_COLUMN,
    NOT_DARK_PATTERN,
    SOURCE_COLUMN,
    TEXT_COLUMN,
    deduplicate_text,
    deduplicate_text_with_source_history,
    load_akash_binary_dataset,
    load_adarsh_category_dataset,
    load_devitachi_binary_dataset,
    load_expanded_binary_dataset,
    load_primary_binary_dataset,
    load_secondary_category_datasets,
    summarize_labels,
    validate_binary_dataset,
)


def test_primary_binary_dataset_is_balanced_and_valid():
    df = load_primary_binary_dataset()

    validate_binary_dataset(df)

    assert df.shape[0] == 2356
    assert summarize_labels(df) == {"0": 1178, "1": 1178}
    assert set(df.columns) == {
        TEXT_COLUMN,
        LABEL_COLUMN,
        CATEGORY_COLUMN,
        SOURCE_COLUMN,
    }


def test_adarsh_dataset_normalizes_category_labels():
    df = load_adarsh_category_dataset()

    validate_binary_dataset(df)

    assert df.shape[0] == 2988
    assert NOT_DARK_PATTERN in set(df[CATEGORY_COLUMN])
    assert df[df[CATEGORY_COLUMN] == NOT_DARK_PATTERN][LABEL_COLUMN].eq(0).all()
    assert df[df[CATEGORY_COLUMN] != NOT_DARK_PATTERN][LABEL_COLUMN].eq(1).all()


def test_secondary_datasets_are_deduplicated():
    df = load_secondary_category_datasets()

    validate_binary_dataset(df[[TEXT_COLUMN, LABEL_COLUMN, CATEGORY_COLUMN, SOURCE_COLUMN]])

    assert df[TEXT_COLUMN].is_unique
    assert df.shape[0] < 2988 + 1818 + 1818
    assert {"adarshm09", "devitachi"}.issubset(set(df[SOURCE_COLUMN]))


def test_devitachi_dataset_loads_as_positive_supplemental_examples():
    df = load_devitachi_binary_dataset()

    validate_binary_dataset(
        pd.concat(
            [
                df[[TEXT_COLUMN, LABEL_COLUMN, CATEGORY_COLUMN, SOURCE_COLUMN]],
                load_primary_binary_dataset().query("label == 0").head(1),
            ],
            ignore_index=True,
        )
    )

    assert df.shape[0] == 1512
    assert df[LABEL_COLUMN].eq(1).all()
    assert {"pattern_type", "page_location", "deceptive_flag"}.issubset(df.columns)


def test_akash_binary_dataset_is_valid_after_dropping_missing_rows():
    df = load_akash_binary_dataset()

    validate_binary_dataset(df)

    assert df.shape[0] == 2423
    assert summarize_labels(df) == {"0": 1891, "1": 532}
    assert set(df[SOURCE_COLUMN]) == {"akashnath29"}


def test_expanded_binary_dataset_uses_devitachi_and_akash_sources():
    df = load_expanded_binary_dataset()

    validate_binary_dataset(df)

    assert df[TEXT_COLUMN].is_unique
    source_text = "; ".join(df[SOURCE_COLUMN].unique())
    assert "krishuppal" in source_text
    assert "devitachi" in source_text
    assert "akashnath29" in source_text
    assert df.shape[0] > load_primary_binary_dataset().shape[0]


def test_deduplicate_text_with_source_history_keeps_duplicate_provenance():
    df = pd.DataFrame(
        {
            TEXT_COLUMN: ["Only 2 left", "Only 2 left", "Regular product info"],
            LABEL_COLUMN: [1, 1, 0],
            CATEGORY_COLUMN: ["Scarcity", "Scarcity", NOT_DARK_PATTERN],
            SOURCE_COLUMN: ["krishuppal", "devitachi", "akashnath29"],
        }
    )

    result = deduplicate_text_with_source_history(df)

    assert result.shape[0] == 2
    assert result.loc[0, SOURCE_COLUMN] == "devitachi; krishuppal"


def test_deduplicate_text_keeps_first_source():
    df = pd.DataFrame(
        {
            TEXT_COLUMN: ["Only 2 left", "Only 2 left", "Regular product info"],
            LABEL_COLUMN: [1, 1, 0],
            CATEGORY_COLUMN: ["Scarcity", "Scarcity", NOT_DARK_PATTERN],
            SOURCE_COLUMN: ["first", "second", "third"],
        }
    )

    result = deduplicate_text(df)

    assert result.shape[0] == 2
    assert result.loc[0, SOURCE_COLUMN] == "first"


def test_validate_binary_dataset_rejects_bad_labels():
    df = pd.DataFrame(
        {
            TEXT_COLUMN: ["hello"],
            LABEL_COLUMN: [2],
            CATEGORY_COLUMN: ["Other"],
            SOURCE_COLUMN: ["test"],
        }
    )

    with pytest.raises(ValueError, match="Expected binary labels"):
        validate_binary_dataset(df)
