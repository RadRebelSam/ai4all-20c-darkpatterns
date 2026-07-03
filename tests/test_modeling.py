from pathlib import Path

import pandas as pd

from src.data import (
    CATEGORY_COLUMN,
    load_dark_pattern_category_dataset,
    load_primary_binary_dataset,
    validate_category_dataset,
)
from src.modeling import (
    load_model,
    make_pipeline,
    model_names,
    save_model,
    train_and_compare_categories,
    train_and_compare,
)


def test_model_names_include_expected_classifiers():
    assert model_names() == [
        "Logistic Regression",
        "Naive Bayes",
        "Linear SVM",
        "Decision Tree",
        "Random Forest",
    ]


def test_pipeline_trains_and_predicts_on_small_sample():
    df = load_primary_binary_dataset()
    sample = pd.concat(
        [
            df[df["label"] == 0].head(40),
            df[df["label"] == 1].head(40),
        ],
        ignore_index=True,
    )
    pipeline = make_pipeline("Logistic Regression")

    pipeline.fit(sample["text"], sample["label"])
    predictions = pipeline.predict(
        [
            "Only 2 left in stock. Buy now before the sale ends.",
            "This cotton pillowcase is machine washable.",
        ]
    )

    assert len(predictions) == 2
    assert set(predictions).issubset({0, 1})


def test_train_and_compare_returns_sorted_results():
    df = load_primary_binary_dataset()

    results = train_and_compare(df)

    assert len(results) == 5
    assert results[0].f1 >= results[-1].f1
    assert all(0 <= result.f1 <= 1 for result in results)


def test_model_can_be_saved_and_loaded(tmp_path: Path):
    df = load_primary_binary_dataset()
    sample = pd.concat(
        [
            df[df["label"] == 0].head(30),
            df[df["label"] == 1].head(30),
        ],
        ignore_index=True,
    )
    pipeline = make_pipeline("Naive Bayes")
    pipeline.fit(sample["text"], sample["label"])

    model_path = tmp_path / "model.joblib"
    save_model(pipeline, model_path)
    loaded = load_model(model_path)

    assert model_path.exists()
    assert loaded.predict(["Limited time only"])[0] in {0, 1}


def test_dark_pattern_category_dataset_is_valid():
    df = load_dark_pattern_category_dataset()

    validate_category_dataset(df)

    assert df[CATEGORY_COLUMN].nunique() >= 4
    assert "Scarcity" in set(df[CATEGORY_COLUMN])


def test_train_and_compare_categories_returns_sorted_results():
    df = load_dark_pattern_category_dataset()
    sample = (
        df.groupby(CATEGORY_COLUMN)
        .head(20)
        .reset_index(drop=True)
    )

    results = train_and_compare_categories(sample)

    assert len(results) == 5
    assert results[0].f1_macro >= results[-1].f1_macro
    assert all(0 <= result.f1_macro <= 1 for result in results)
