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


def load_data():
    all_df = pd.read_csv("../misc/school_email_labeled.csv").fillna("")
    all_df["text"] = "Subject: " + all_df["subject"] + "\n\nBody: " + all_df["body"]

    le = LabelEncoder()
    all_df["label_id"] = le.fit_transform(all_df["label"])

    id2label = {i: name for i, name in enumerate(le.classes_)}
    label2id = {name: i for i, name in id2label.items()}

    train_val_df, test_df = train_test_split(
        all_df,
        test_size=0.2,
        stratify=all_df["label_id"],
        random_state=42,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        stratify=train_val_df["label_id"],
        random_state=42,
    )

    train_ds = Dataset.from_pandas(
        train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )
    val_ds = Dataset.from_pandas(
        val_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )
    test_ds = Dataset.from_pandas(
        test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )

    return train_ds, val_ds, test_ds, le, id2label, label2id


if __name__ == "__main__":
    train_ds, val_ds, test_ds, le, id2label, label2id = load_data()

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

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
            max_length=1024,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    output_dir = "../models/modern_BERT"

    args = TrainingArguments(
        output_dir=output_dir,
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

    print("Validation metrics:")
    print(trainer.evaluate(eval_dataset=val_ds))

    print("Saving fine-tuned model to", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
