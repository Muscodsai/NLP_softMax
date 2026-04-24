import numpy as np
from pathlib import Path
import sys
import torch
from datasets import Dataset
import evaluate

from benchmark_data import load_canonical_splits


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

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    ModernBertForSequenceClassification,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "models" / "modern_BERT_canonical"
MODEL_ID = "answerdotai/ModernBERT-base"


def get_precision_args():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": False}


def to_hf_dataset(df):
    return Dataset.from_pandas(
        df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )


def main():
    train_df, val_df, test_df, label_encoder = load_canonical_splits()
    labels = list(label_encoder.classes_)
    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in id2label.items()}

    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)

    print(f"Canonical split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = ModernBertForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024)

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
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
        report_to="none",
        **get_precision_args(),
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

    print("Saving canonical ModernBERT model to", OUTPUT_DIR)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
