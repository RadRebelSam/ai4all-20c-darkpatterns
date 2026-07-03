from scripts.train_models import load_training_dataset
from src.data import SOURCE_COLUMN, validate_binary_dataset


def test_load_training_dataset_primary():
    df = load_training_dataset("primary")

    validate_binary_dataset(df)

    assert set(df[SOURCE_COLUMN]) == {"krishuppal"}


def test_load_training_dataset_expanded():
    df = load_training_dataset("expanded")

    validate_binary_dataset(df)

    source_text = "; ".join(df[SOURCE_COLUMN].unique())
    assert "krishuppal" in source_text
    assert "devitachi" in source_text
    assert "akashnath29" in source_text
