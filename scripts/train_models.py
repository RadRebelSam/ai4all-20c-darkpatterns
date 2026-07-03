"""Train dark-pattern classifiers and save metrics/artifacts."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    CATEGORY_COLUMN,
    SOURCE_COLUMN,
    load_dark_pattern_category_dataset,
    load_expanded_binary_dataset,
    load_primary_binary_dataset,
    summarize_labels,
    validate_binary_dataset,
    validate_category_dataset,
)
from src.modeling import (
    category_results_to_frame,
    results_to_frame,
    save_model,
    train_and_compare,
    train_and_compare_categories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dark-pattern classifiers.")
    parser.add_argument(
        "--dataset",
        choices=["primary", "expanded"],
        default="primary",
        help="Use the balanced primary dataset or primary + Devitachi + Akash supplemental rows.",
    )
    parser.add_argument(
        "--skip-category-model",
        action="store_true",
        help="Only train the binary model and skip the second-stage category model.",
    )
    return parser.parse_args()


def load_training_dataset(name: str):
    if name == "primary":
        return load_primary_binary_dataset()
    if name == "expanded":
        return load_expanded_binary_dataset()
    raise ValueError(f"Unknown dataset option: {name}")


def main() -> None:
    args = parse_args()
    df = load_training_dataset(args.dataset)
    validate_binary_dataset(df)

    results = train_and_compare(df)
    metrics = results_to_frame(results)

    report_dir = Path("reports")
    artifact_dir = Path("artifacts")
    report_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)

    metrics_path = report_dir / (
        "model_metrics.csv"
        if args.dataset == "primary"
        else f"model_metrics_{args.dataset}.csv"
    )
    artifact_path = artifact_dir / (
        "best_binary_model.joblib"
        if args.dataset == "primary"
        else f"best_binary_model_{args.dataset}.joblib"
    )

    metrics.to_csv(metrics_path, index=False)
    save_model(results[0].pipeline, artifact_path)

    print(f"Dataset option: {args.dataset}")
    print(f"Rows: {len(df)}")
    print(f"Label counts: {summarize_labels(df)}")
    print(f"Source counts: {df[SOURCE_COLUMN].value_counts().to_dict()}")
    print()
    print(metrics.to_string(index=False))
    print(f"Best model: {results[0].name}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model saved to: {artifact_path}")

    if args.skip_category_model:
        return

    print()
    print("Training second-stage dark-pattern category model...")
    category_df = load_dark_pattern_category_dataset()
    validate_category_dataset(category_df)
    category_results = train_and_compare_categories(category_df)
    category_metrics = category_results_to_frame(category_results)
    category_metrics_path = report_dir / "category_model_metrics.csv"
    category_artifact_path = artifact_dir / "best_category_model.joblib"

    category_metrics.to_csv(category_metrics_path, index=False)
    save_model(category_results[0].pipeline, category_artifact_path)

    print(f"Category rows: {len(category_df)}")
    print(f"Category counts: {summarize_labels(category_df, CATEGORY_COLUMN)}")
    print()
    print(category_metrics.to_string(index=False))
    print(f"Best category model: {category_results[0].name}")
    print(f"Category metrics saved to: {category_metrics_path}")
    print(f"Category model saved to: {category_artifact_path}")


if __name__ == "__main__":
    main()
