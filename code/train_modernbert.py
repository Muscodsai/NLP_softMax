import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import evaluate


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

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    ModernBertForSequenceClassification,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "misc" / "school_email_labeled.csv"
MODEL_OUTPUT_DIR = ROOT_DIR / "models" / "modern_BERT"


def get_precision_args():
    # bf16=True only if the GPU explicitly supports BF16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": False}


def load_data():
    # keep the same random state and stratification to ensure the same split between training and evaluation
    all_df = pd.read_csv(DATA_PATH).fillna("")
    all_df["text"] = "Subject: " + all_df["subject"] + "\n\nBody: " + all_df["body"]

    # encode labels and create mappings
    le = LabelEncoder()
    all_df["label_id"] = le.fit_transform(all_df["label"])

    id2label = {i: name for i, name in enumerate(le.classes_)}
    label2id = {name: i for i, name in id2label.items()}

    # create splits with same random state
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
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )

    def tokenize_fn(batch):
        # tokenize the batch of examples use the same tokenizer and settings to ensure consistency
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # use the same collator to ensure consistent padding
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

    # save checkpoints under the repository models directory
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    precision_args = get_precision_args()

    args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
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
        **precision_args,
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

    print("Saving fine-tuned model to", MODEL_OUTPUT_DIR)
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
