from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parents[0]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_data import load_canonical_splits
from deBerta_fineTuned import train_and_evaluate_deberta


OUTPUT_DIR = ROOT_DIR / "models" / "deberta_canonical"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"


def main():
    train_df, val_df, test_df, _ = load_canonical_splits()
    print(f"Canonical split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    train_and_evaluate_deberta(
        train_df,
        val_df,
        test_df,
        output_dir=OUTPUT_DIR,
        best_model_dir=BEST_MODEL_DIR,
        metrics_path=METRICS_PATH,
        confusion_matrix_path=CONFUSION_MATRIX_PATH,
        test_metrics_title="Canonical Test Metrics:",
        confusion_matrix_title="DeBERTa Canonical Test Confusion Matrix",
    )


if __name__ == "__main__":
    main()
