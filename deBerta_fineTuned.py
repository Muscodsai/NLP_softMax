import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

MODEL_NAME = "microsoft/deberta-v3-base"

def load_dataset_with_validation_set() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv("misc/school_email_labeled.csv")
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
    df["subject"] = df["subject"]
    df["body"] = df["body"]
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
    def __init__(self, df):
        df = preprocess_input(df)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        encodings = tokenize_dataframe(df, tokenizer)   
        labels = LabelEncoder().fit_transform(df["label"])

        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
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

train_df, val_df, test_df = load_dataset_with_validation_set()
train_dataset = FormatDataset(train_df)
val_dataset   = FormatDataset(val_df)
test_dataset  = FormatDataset(test_df)
num_labels = len(train_df["label"].unique())
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16)
test_loader  = DataLoader(test_dataset,  batch_size=16)

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3
total_steps = len(train_loader) * epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc  = correct / total
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}\ntrain loss: {avg_loss:.4f}\nval loss: {avg_val_loss:.4f}\nval acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("best_model")
        print("checkpoint saved")


model = AutoModelForSequenceClassification.from_pretrained("best_model")
model.to(device)
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

avg = "binary" if num_labels == 2 else "macro"

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average=avg, zero_division=0)
recall = recall_score(all_labels, all_preds, average=avg, zero_division=0)
f1 = f1_score(all_labels, all_preds, average=avg, zero_division=0)


print("\n")
print("Final Test Metrics:")
print("\n")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}  ({avg})")
print(f"Recall : {recall:.4f}  ({avg})")
print(f"F1 : {f1:.4f}  ({avg})")

cm = confusion_matrix(all_labels, all_preds)
classes = [str(i) for i in range(num_labels)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(max(6, num_labels), max(5, num_labels - 1)))
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Confusion Matrix — Test Set", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "averaging": avg
}

with open("test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
