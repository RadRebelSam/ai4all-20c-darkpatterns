"""Run 5-fold cross-validation across all models and print a summary table."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data import load_primary_binary_dataset
from src.modeling import cross_validate_all


def main() -> None:
    print("Loading dataset...")
    df = load_primary_binary_dataset()

    print("Running 5-fold stratified cross-validation (this may take a minute)...")
    results = cross_validate_all(df, n_splits=5)

    rows = [r.as_dict() for r in results]
    frame = pd.DataFrame(rows)
    frame.columns = ["Model", "Acc Mean", "Acc Std", "F1 Mean", "F1 Std"]

    print("\n5-Fold Cross-Validation Results")
    print("=" * 60)
    print(frame.to_string(index=False))

    out_path = Path("reports") / "cv_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame([r.as_dict() for r in results]).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
