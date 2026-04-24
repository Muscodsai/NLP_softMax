import argparse

from benchmark_all_models import evaluate_baselines, evaluate_encoder_model, evaluate_gemma
from benchmark_data import ROOT_DIR, load_canonical_splits
import pandas as pd


MODERNBERT_DIR = ROOT_DIR / "models" / "modern_BERT_canonical"
DEBERTA_DIR = ROOT_DIR / "models" / "deberta_canonical" / "best_model"
OUTPUT_PATH = ROOT_DIR / "metrics" / "model_comparison_canonical.csv"
PER_CLASS_OUTPUT_PATH = ROOT_DIR / "metrics" / "model_comparison_canonical_per_class.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Compare canonical-split models on one shared test set."
    )
    parser.add_argument(
        "--include-gemma",
        action="store_true",
        help="Run the gated Gemma benchmark as part of the comparison.",
    )
    args = parser.parse_args()

    train_df, val_df, test_df, label_encoder = load_canonical_splits()
    labels = list(label_encoder.classes_)

    rows = []
    per_class_rows = []

    baseline_rows, baseline_per_class_rows = evaluate_baselines(train_df, test_df, labels)
    rows.extend(baseline_rows)
    per_class_rows.extend(baseline_per_class_rows)

    modernbert_row, modernbert_per_class_rows = evaluate_encoder_model(
        "ModernBERT Canonical",
        MODERNBERT_DIR,
        test_df,
        max_length=1024,
        label_names=labels,
    )
    rows.append(modernbert_row)
    per_class_rows.extend(modernbert_per_class_rows)

    deberta_row, deberta_per_class_rows = evaluate_encoder_model(
        "DeBERTa Canonical",
        DEBERTA_DIR,
        test_df,
        max_length=256,
        label_names=labels,
    )
    rows.append(deberta_row)
    per_class_rows.extend(deberta_per_class_rows)

    gemma_row, gemma_per_class_rows = evaluate_gemma(test_df, labels, include_gemma=args.include_gemma)
    rows.append(gemma_row)
    per_class_rows.extend(gemma_per_class_rows)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["status", "macro_f1", "accuracy"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    per_class_results = pd.DataFrame(per_class_rows).sort_values(
        by=["status", "model", "label"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    per_class_results.to_csv(PER_CLASS_OUTPUT_PATH, index=False)

    print("Canonical split:")
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print()
    print(results.to_string(index=False))
    print()
    print("Per-class performance:")
    print(per_class_results.to_string(index=False))
    print()
    print(f"Saved comparison table to {OUTPUT_PATH}")
    print(f"Saved per-class comparison table to {PER_CLASS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
