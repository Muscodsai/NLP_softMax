import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def patch_torch_compile_for_python_312():
    # patch torch.compile to avoid incompatibility between 3.12 and mordernbert's dependencies
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

# define paths relative to the repository root for portability
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "misc" / "school_email_labeled.csv"
MODEL_DIR = ROOT_DIR / "models" / "modern_BERT"
CONFUSION_MATRIX_PATH = ROOT_DIR / "models" / "bert_confusion_matrix_600.png"


def load_test_data():
    # create the same held-out split used during training
    all_df = pd.read_csv(DATA_PATH).fillna("")
    all_df["text"] = "Subject: " + all_df["subject"] + "\n\nBody: " + all_df["body"]

    le = LabelEncoder()
    all_df["label_id"] = le.fit_transform(all_df["label"])

    _, test_df = train_test_split(
        all_df,
        test_size=0.2,
        stratify=all_df["label_id"],
        random_state=42,
    )

    test_ds = Dataset.from_pandas(
        test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )

    return test_ds, test_df, le


def tokenize_fn(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=1024,
    )


def evaluate_model(model_path, label_encoder, test_ds, test_df, title, output_path):
    # prepare model and tokenizer
    labels = list(label_encoder.classes_)
    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model.eval()

    # tokenize once before the per-example inference loop
    test_ds_tokenized = test_ds.map(lambda batch: tokenize_fn(batch, tokenizer), batched=True)

    predictions = []
    # run inference
    for example in test_ds_tokenized:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = np.argmax(outputs.logits.numpy(), axis=-1)
        predictions.extend(preds.tolist())

    # compute metrics
    true_labels = test_df["label_id"].tolist()
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    metrics = {
        "accuracy": accuracy.compute(predictions=predictions, references=true_labels)["accuracy"],
        "macro_f1": f1.compute(predictions=predictions, references=true_labels, average="macro")["f1"],
    }

    print(f"\n{title} metrics:")
    print(metrics)

    # plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

    return metrics


if __name__ == "__main__":
    test_ds, test_df, le = load_test_data()
    print(f"Test set size: {len(test_df)}")

    model_dir = str(MODEL_DIR)
    assert os.path.isdir(model_dir), f"Model directory not found: {model_dir}"

    evaluate_model(
        model_dir,
        le,
        test_ds,
        test_df,
        "ModernBERT Fine-tuned Test Confusion Matrix",
        str(CONFUSION_MATRIX_PATH),
    )
