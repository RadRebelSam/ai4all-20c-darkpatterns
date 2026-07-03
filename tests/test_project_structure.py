from pathlib import Path


def test_required_project_files_exist():
    required = [
        "README.md",
        "PROJECT_PLAN.md",
        "requirements.txt",
        "pyproject.toml",
        "src/__init__.py",
        "src/data.py",
    ]

    for filename in required:
        assert Path(filename).exists(), f"Missing required file: {filename}"


def test_dataset_folder_contains_expected_files():
    dataset_dir = Path("datasets")
    assert dataset_dir.exists()
    assert (dataset_dir / "krishuppal - dark-patterns.csv").exists()
    assert (dataset_dir / "adarshm09 - dark-pattern-dataset.csv").exists()

