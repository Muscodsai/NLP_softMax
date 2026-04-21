import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import evaluate

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    ModernBertForSequenceClassification,
)

# Load data
df = pd.read_csv("school_email_labeled.csv").fillna("")
df["text"] = "Subject: " + df["subject"] + "\n\nBody: " + df["body"]

# Encode labels
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

id2label = {i: name for i, name in enumerate(le.classes_)}
label2id = {name: i for i, name in id2label.items()}

# Train/val split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label_id"],
    random_state=42,
)

train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False)
val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False)

# Tokenizer/model
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ModernBertForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(le.classes_),
    id2label=id2label,
    label2id=label2id,
)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=1024,   # ModernBERT can handle up to 4096 tokens, but we use 1024 for efficiency
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# Train
args = TrainingArguments(
    output_dir="./modernbert_email_cls",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)

trainer.save_model("./modernbert_email_cls")
tokenizer.save_pretrained("./modernbert_email_cls")