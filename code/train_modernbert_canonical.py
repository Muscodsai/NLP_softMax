from pathlib import Path
import sys
from datasets import Dataset

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from benchmark_data import load_canonical_splits
from train_modernbert import train_modernbert


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "models" / "modern_BERT_canonical"
MODEL_ID = "answerdotai/ModernBERT-base"


def to_hf_dataset(df):
    return Dataset.from_pandas(
        df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )


def main():
    train_df, val_df, test_df, label_encoder = load_canonical_splits()
    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)

    print(f"Canonical split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    train_modernbert(
        train_ds,
        val_ds,
        label_names=list(label_encoder.classes_),
        output_dir=OUTPUT_DIR,
        model_id=MODEL_ID,
        run_label="canonical ModernBERT",
    )


if __name__ == "__main__":
    main()
