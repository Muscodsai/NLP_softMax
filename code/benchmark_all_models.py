import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from baseline import fit_tf_idf_vectoriser, get_baseline_models, transform_X_tf_idf
from benchmark_data import ROOT_DIR, load_canonical_splits


def patch_torch_compile_for_python_312():
    if sys.version_info < (3, 12) or not hasattr(torch, "compile"):
        return

    try:
        torch.compile(lambda x: x)
    except RuntimeError as exc:
        if "Python 3.12+" not in str(exc):
            raise

        def _noop_compile(model=None, *args, **kwargs):
            if model is None:
                return lambda fn: fn
            return model

        torch.compile = _noop_compile


patch_torch_compile_for_python_312()

LABEL_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "evaluated_samples"]
DEFAULT_MODERNBERT_DIR = ROOT_DIR / "models" / "modern_BERT_canonical"
DEFAULT_DEBERTA_DIR = ROOT_DIR / "models" / "deberta_canonical" / "best_model"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "metrics" / "model_comparison.csv"
DEFAULT_PER_CLASS_OUTPUT_PATH = ROOT_DIR / "metrics" / "model_comparison_per_class.csv"
GEMMA_MODEL_ID = "google/gemma-7b-it"


def build_modernbert_text(row: pd.Series) -> str:
    # match the text template used during ModernBERT training
    return f"Subject: {row['subject']}\n\nBody: {row['body']}"


def build_deberta_text(row: pd.Series) -> str:
    # match the text template used during DeBERTa training
    return f"[SUBJECT] {row['subject']} [BODY] {row['body']}"


def metric_row(model_name: str, y_true, y_pred, label_names: list[str]) -> tuple[dict[str, object], list[dict[str, object]]]:
    # write both one overall row and one row per label so the CSVs stay aligned
    overall_row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "evaluated_samples": len(y_true),
        "status": "ok",
        "notes": "",
    }
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        zero_division=0,
    )
    per_class_rows = []
    for idx, label_name in enumerate(label_names):
        per_class_rows.append(
            {
                "model": model_name,
                "label": label_name,
                "precision": precisions[idx],
                "recall": recalls[idx],
                "f1": f1s[idx],
                "support": int(supports[idx]),
                "status": "ok",
                "notes": "",
            }
        )
    return overall_row, per_class_rows

# skip evaluation and write empty rows if a model is missing or fails to load
def skipped_row(
    model_name: str,
    reason: str,
    label_names: list[str] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    row = {
        "model": model_name,
        "status": "skipped",
        "notes": reason,
    }
    for column in LABEL_COLUMNS:
        row[column] = None
    per_class_rows = []
    for label_name in label_names or []:
        per_class_rows.append(
            {
                "model": model_name,
                "label": label_name,
                "precision": None,
                "recall": None,
                "f1": None,
                "support": None,
                "status": "skipped",
                "notes": reason,
            }
        )
    return row, per_class_rows


def evaluate_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_names: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_x = train_df[["subject", "body"]]
    train_y = train_df["label"]
    test_x = test_df[["subject", "body"]]
    test_y = test_df["label"]

    subject_vectoriser = fit_tf_idf_vectoriser(train_x["subject"])
    body_vectoriser = fit_tf_idf_vectoriser(train_x["body"])
    train_x_transformed = transform_X_tf_idf(train_x, subject_vectoriser, body_vectoriser)
    test_x_transformed = transform_X_tf_idf(test_x, subject_vectoriser, body_vectoriser)

    rows = []
    per_class_rows = []
    for model_name, model in get_baseline_models().items():
        model.fit(train_x_transformed, train_y)
        predictions = model.predict(test_x_transformed)
        row = {
            "model": model_name,
            "accuracy": accuracy_score(test_y, predictions),
            "macro_precision": precision_score(test_y, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(test_y, predictions, average="macro", zero_division=0),
            "macro_f1": f1_score(test_y, predictions, average="macro", zero_division=0),
            "evaluated_samples": len(test_y),
            "status": "ok",
            "notes": "",
        }
        rows.append(row)

        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            test_y,
            predictions,
            labels=label_names,
            zero_division=0,
        )
        for idx, label_name in enumerate(label_names):
            per_class_rows.append(
                {
                    "model": model_name,
                    "label": label_name,
                    "precision": precisions[idx],
                    "recall": recalls[idx],
                    "f1": f1s[idx],
                    "support": int(supports[idx]),
                    "status": "ok",
                    "notes": "",
                }
            )

    return rows, per_class_rows


def evaluate_encoder_model(
    model_name: str,
    model_dir: Path,
    test_df: pd.DataFrame,
    *,
    max_length: int,
    label_names: list[str],
    text_builder,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not model_dir.is_dir():
        return skipped_row(model_name, f"Missing local model directory: {model_dir}", label_names)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model.eval()

    predictions = []
    # rebuild each sample in the same format the model saw during training
    for _, row in test_df.iterrows():
        text = text_builder(row)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        predictions.append(int(outputs.logits.argmax(dim=-1).item()))

    return metric_row(model_name, test_df["label_id"].tolist(), predictions, label_names)


def build_gemma_prompt(text: str, labels: list[str]) -> str:
    subject = text.split("Subject: ")[1].split("\n\nBody: ")[0]
    body = text.split("\n\nBody: ")[1]
    return (
        "Classify the following email into exactly one label from:\n"
        + ", ".join(labels)
        + f"\n\nSubject: {subject}\n\nBody:\n{body}\n\nAnswer:"
    )


def evaluate_gemma(
    test_df: pd.DataFrame,
    labels: list[str],
    include_gemma: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not include_gemma:
        return skipped_row("Gemma 7B", "Pass --include-gemma to run the gated 7B benchmark.", labels)

    try:
        # Gemma is optional because it needs remote access, authentication, and more memory
        tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL_ID,
            device_map="auto" if torch.cuda.is_available() else None,
            dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        model.eval()
    except Exception as exc:
        return skipped_row("Gemma 7B", f"Unable to load model: {exc}", labels)

    label_to_id = {label: idx for idx, label in enumerate(labels)}
    predictions = []
    valid_indices = []

    # only keep samples where the generated text clearly names one benchmark label
    for idx, text in enumerate(test_df["text"].tolist()):
        prompt = build_gemma_prompt(text, labels)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_part = generated.split("Answer:")[-1].strip()
        predicted_label = next((label for label in labels if label in answer_part), None)
        if predicted_label is None:
            continue

        valid_indices.append(idx)
        predictions.append(label_to_id[predicted_label])

    if not predictions:
        return skipped_row("Gemma 7B", "No valid label strings were generated on the benchmark split.", labels)

    # Gemma may score on fewer than the full test samples, so note that in the output rows.
    y_true = test_df.iloc[valid_indices]["label_id"].tolist()
    row, per_class_rows = metric_row("Gemma 7B", y_true, predictions, labels)
    row["notes"] = f"Evaluated {len(predictions)} of {len(test_df)} samples with parsable outputs."
    for per_class_row in per_class_rows:
        per_class_row["notes"] = row["notes"]
    return row, per_class_rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare repository models on one canonical test split."
    )
    parser.add_argument(
        "--modernbert-dir",
        type=Path,
        default=DEFAULT_MODERNBERT_DIR,
        help="Directory containing the ModernBERT checkpoint to benchmark.",
    )
    parser.add_argument(
        "--deberta-dir",
        type=Path,
        default=DEFAULT_DEBERTA_DIR,
        help="Directory containing the DeBERTa checkpoint to benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path for the overall comparison table.",
    )
    parser.add_argument(
        "--per-class-output",
        type=Path,
        default=DEFAULT_PER_CLASS_OUTPUT_PATH,
        help="CSV path for the per-class comparison table.",
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

    # baselines are retrained directly on the canonical split before scoring.
    baseline_rows, baseline_per_class_rows = evaluate_baselines(train_df, test_df, labels)
    rows.extend(baseline_rows)
    per_class_rows.extend(baseline_per_class_rows)

    # local encoder checkpoints are scored on the same canonical test split.
    modernbert_row, modernbert_per_class_rows = evaluate_encoder_model(
        "ModernBERT",
        args.modernbert_dir,
        test_df,
        max_length=1024,
        label_names=labels,
        text_builder=build_modernbert_text,
    )
    rows.append(modernbert_row)
    per_class_rows.extend(modernbert_per_class_rows)

    deberta_row, deberta_per_class_rows = evaluate_encoder_model(
        "DeBERTa v3",
        args.deberta_dir,
        test_df,
        max_length=256,
        label_names=labels,
        text_builder=build_deberta_text,
    )
    rows.append(deberta_row)
    per_class_rows.extend(deberta_per_class_rows)

    # Gemma remains optional because it is not a fully local checkpoint.
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    per_class_results.to_csv(args.per_class_output, index=False)

    print("Canonical split:")
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print()
    print(results.to_string(index=False))
    print()
    print("Per-class performance:")
    print(per_class_results.to_string(index=False))
    print()
    print(f"Saved comparison table to {args.output}")
    print(f"Saved per-class comparison table to {args.per_class_output}")
    print("Canonical benchmark note: use checkpoints trained on this shared split for final ranking.")


if __name__ == "__main__":
    main()
