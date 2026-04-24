import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from pathlib import Path

MODEL_NAME = "microsoft/deberta-v3-base"
ROOT_DIR = Path(__file__).resolve().parents[1]
BEST_MODEL_DIR = ROOT_DIR / "best_model"
METRICS_PATH = ROOT_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = ROOT_DIR / "confusion_matrix.png"

def load_dataset_with_validation_set() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # The two-stage split keeps label proportions roughly stable across train/val/test.
    data = pd.read_csv(ROOT_DIR / "misc" / "school_email_labeled.csv").fillna("")
    train_val, test_data = train_test_split(
        data,
        test_size=0.15,
        stratify=data["label"],
        random_state=42
    )

    train_data, val_data = train_test_split(
        train_val,
        test_size=0.1765,
        stratify=train_val["label"],
        random_state=42
    )
    return train_data, val_data, test_data

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "subject" not in df.columns:
        df["subject"] = ""
    if "body" not in df.columns:
        df["body"] = ""
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    # DeBERTa was trained with explicit subject/body markers instead of the ModernBERT template.
    df["text"] = (
        "[SUBJECT] " +
        df["subject"] +
        " [BODY] " +
        df["body"]
    )
    return df

def tokenize_dataframe(df, tokenizer, max_length=256):
    return tokenizer(
        df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

class FormatDataset(Dataset):
    def __init__(self, df, tokenizer=None, label_to_id=None, max_length=256):
        # Pre-tokenize once so the training loop can work with simple tensor batches.
        df = preprocess_input(df)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        if label_to_id is None:
            label_to_id = {
                label: idx for idx, label in enumerate(sorted(df["label"].unique()))
            }

        encodings = tokenize_dataframe(df, tokenizer, max_length=max_length)
        labels = [label_to_id[label] for label in df["label"].tolist()]

        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach()
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np
import torch
import json

def evaluate_model(model, data_loader, device):
    # Shared evaluation helper used for validation during training and final test scoring.
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def train_and_evaluate_deberta(
    train_df,
    val_df,
    test_df,
    *,
    output_dir,
    best_model_dir,
    metrics_path,
    confusion_matrix_path,
    test_metrics_title="Final Test Metrics:",
    confusion_matrix_title="Confusion Matrix — Test Set",
):
    # Keep label ordering fixed so saved checkpoints and metric reports use the same ids.
    labels = sorted(train_df["label"].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    # All three splits use the same tokenizer and label mapping.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    train_dataset = FormatDataset(train_df, tokenizer=tokenizer, label_to_id=label_to_id)
    val_dataset = FormatDataset(val_df, tokenizer=tokenizer, label_to_id=label_to_id)
    test_dataset = FormatDataset(test_df, tokenizer=tokenizer, label_to_id=label_to_id)

    num_labels = len(labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(labels)},
        label2id=label_to_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train/val/test loaders are built separately because only train should shuffle.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # Train for a few epochs and keep the checkpoint with the best validation accuracy.
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise ValueError(
                    "Encountered non-finite DeBERTa training loss. "
                    "This usually indicates a bad attention/backend configuration or unstable numerics."
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}\ntrain loss: {avg_loss:.4f}\nval loss: {val_loss:.4f}\nval acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(best_model_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_model_dir)
            print("checkpoint saved")

    # Final reported metrics always come from reloading the best saved checkpoint.
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir, use_safetensors=True)
    model.to(device)
    _, _, all_labels, all_preds = evaluate_model(model, test_loader, device)

    avg = "binary" if num_labels == 2 else "macro"
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=avg, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=avg, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=avg, zero_division=0)

    print("\n")
    print(test_metrics_title)
    print("\n")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}  ({avg})")
    print(f"Recall : {recall:.4f}  ({avg})")
    print(f"F1 : {f1:.4f}  ({avg})")

    # Save both a numeric summary and a confusion matrix artifact for later comparison.
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, num_labels), max(5, num_labels - 1)))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(confusion_matrix_title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close(fig)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "averaging": avg
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    train_df, val_df, test_df = load_dataset_with_validation_set()
    train_and_evaluate_deberta(
        train_df,
        val_df,
        test_df,
        output_dir=ROOT_DIR,
        best_model_dir=BEST_MODEL_DIR,
        metrics_path=METRICS_PATH,
        confusion_matrix_path=CONFUSION_MATRIX_PATH,
    )


if __name__ == "__main__":
    main()
