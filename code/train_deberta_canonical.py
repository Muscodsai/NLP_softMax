from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from benchmark_data import load_canonical_splits


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "models" / "deberta_canonical"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
MODEL_NAME = "microsoft/deberta-v3-base"


class CanonicalDataset(Dataset):
    def __init__(self, df, tokenizer, label_to_id, max_length=256):
        texts = ("[SUBJECT] " + df["subject"] + " [BODY] " + df["body"]).tolist()
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = [label_to_id[label] for label in df["label"].tolist()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx].clone().detach() for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def evaluate_model(model, data_loader, device):
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


def main():
    train_df, val_df, test_df, label_encoder = load_canonical_splits()
    labels = list(label_encoder.classes_)
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    print(f"Canonical split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    train_dataset = CanonicalDataset(train_df, tokenizer, label_to_id)
    val_dataset = CanonicalDataset(val_df, tokenizer, label_to_id)
    test_dataset = CanonicalDataset(test_df, tokenizer, label_to_id)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label={i: label for i, label in enumerate(labels)},
        label2id=label_to_id,
        use_safetensors=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}\n"
            f"train loss: {avg_train_loss:.4f}\n"
            f"val loss: {val_loss:.4f}\n"
            f"val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(BEST_MODEL_DIR, safe_serialization=True)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            print("checkpoint saved")

    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR, use_safetensors=True)
    model.to(device)
    _, _, y_true, y_pred = evaluate_model(model, test_loader, device)

    avg = "binary" if len(labels) == 2 else "macro"
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

    print("\nCanonical Test Metrics:\n")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}  ({avg})")
    print(f"Recall : {recall:.4f}  ({avg})")
    print(f"F1 : {f1:.4f}  ({avg})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("DeBERTa Canonical Test Confusion Matrix", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "averaging": avg,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
