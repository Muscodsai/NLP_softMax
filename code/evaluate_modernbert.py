import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def load_test_data():
    all_df = pd.read_csv("../misc/school_email_labeled.csv").fillna("")
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
    labels = list(label_encoder.classes_)
    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.eval()

    test_ds_tokenized = test_ds.map(lambda batch: tokenize_fn(batch, tokenizer), batched=True)

    predictions = []
    for example in test_ds_tokenized:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = np.argmax(outputs.logits.numpy(), axis=-1)
        predictions.extend(preds.tolist())

    true_labels = test_df["label_id"].tolist()
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    metrics = {
        "accuracy": accuracy.compute(predictions=predictions, references=true_labels)["accuracy"],
        "macro_f1": f1.compute(predictions=predictions, references=true_labels, average="macro")["f1"],
    }

    print(f"\n{title} metrics:")
    print(metrics)

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

    model_dir = "../models/modern_BERT"
    assert os.path.isdir(model_dir), f"Model directory not found: {model_dir}"

    evaluate_model(
        model_dir,
        le,
        test_ds,
        test_df,
        "ModernBERT Fine-tuned (600 samples) Test Confusion Matrix",
        "../models/bert_confusion_matrix_600.png",
    )
