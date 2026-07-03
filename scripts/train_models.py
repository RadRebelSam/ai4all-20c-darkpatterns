"""Train dark-pattern classifiers and save metrics/artifacts."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    SOURCE_COLUMN,
    load_expanded_binary_dataset,
    load_primary_binary_dataset,
    summarize_labels,
    validate_binary_dataset,
)
from src.modeling import results_to_frame, save_model, train_and_compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dark-pattern classifiers.")
    parser.add_argument(
        "--dataset",
        choices=["primary", "expanded"],
        default="primary",
        help="Use the balanced primary dataset or primary + Devitachi + Akash supplemental rows.",
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


if __name__ == "__main__":
    main()
