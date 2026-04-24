# NLP_softMax
NLP project on email message categorization

## Setup
Prerequisites:
- Python 3.12
- Internet access for first-time model downloads
- A Hugging Face account and access token for gated models such as `google/gemma-7b-it`
- A writable local environment for model caches and output artifacts

Verified environment versions:
- Python `3.12`
- `torch==2.3.1`
- `transformers==4.56.2`
- `accelerate==1.13.0`
- `datasets==2.14.6`
- `numpy==1.26.4`
- `pandas==3.0.2`
- `pyarrow==14.0.2`
- `protobuf==5.29.5`
- `scikit-learn==1.8.0`
- `gradio==6.13.0`
- `dspy-ai` from `misc/requirements.txt`

1. Create a Python venv
```bash
python3 -m venv venv
```
2. Activate it
```bash
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r misc/requirements.txt
```
4. Optional but recommended: log in to Hugging Face before running transformer-based scripts
```bash
huggingface-cli login
```

All commands below are intended to be run from the repository root.

## Unified Comparison
The easiest way to compare models fairly is to score them on one shared test split instead of letting each script create its own split.

For the benchmark in this repo, the intended workflow is:
- train ModernBERT on the canonical split
- train DeBERTa on the same canonical split
- compare both checkpoints, plus the classical baselines, on the shared test set

It uses one canonical stratified split of `misc/school_email_labeled.csv`:
- 60% train
- 20% validation
- 20% test

1. Train ModernBERT on the canonical split
```bash
python code/train_modernbert_canonical.py
```

This saves the model to `models/modern_BERT_canonical/`.

2. Train DeBERTa on the canonical split
```bash
python code/train_deberta_canonical.py
```

This saves the best checkpoint to `models/deberta_canonical/best_model/`.

3. Run the benchmark
```bash
python code/benchmark_all_models.py
```

This benchmark compares:
- Naive Bayes baseline
- SVM baseline
- `models/modern_BERT_canonical/`
- `models/deberta_canonical/best_model/`

Outputs:
- overall comparison table: `metrics/model_comparison.csv`
- per-class comparison table: `metrics/model_comparison_per_class.csv`

To also include Gemma in the same benchmark:

```bash
python code/benchmark_all_models.py --include-gemma
```

Notes:
- The full benchmark expects both canonical checkpoints above to exist before you run it.
- If a checkpoint is missing, that model is reported as skipped in the benchmark output instead of being included in the final comparison.
- Gemma requires Hugging Face access approval and local authentication.
- The per-class output includes `precision`, `recall`, `f1`, and `support` for every label and model.
- If you need to benchmark different checkpoint directories, `code/benchmark_all_models.py` also accepts `--modernbert-dir` and `--deberta-dir`.

## External Access
- `answerdotai/ModernBERT-base`: public on Hugging Face, no token required for normal download.
- `microsoft/deberta-v3-base`: public on Hugging Face, no token required for normal download.
- `google/gemma-7b-it`: gated on Hugging Face. You must have access approved for the model and be logged in with a token locally.
- `huggingface/allenai/Olmo-3-7B-Instruct` in the DSPy script is accessed through the configured DSPy backend, not through local fine-tuning code in this repo.

If you need Gemma, request access on Hugging Face first, then authenticate locally:

```bash
huggingface-cli login
```

## Demo
```bash
python code/demo.py
```

This launches a Gradio app with these tabs:
- Baseline Model Demo
- ModernBERT Demo
- Pretrained OLMo Model Demo
- Gemma Demo

Notes:
- The demo needs an available local port for Gradio. In this environment, the default range `7860-7959` was already occupied.
- The OLMo demo path also depends on `DECODER_API_KEY`.
- The ModernBERT demo needs a trained local model under `models/modern_BERT/`.
- The Gemma demo needs Hugging Face access approval and local authentication for `google/gemma-7b-it`.

To launch the demo locally:

```bash
source venv/bin/activate
python code/demo.py
```

If Gradio reports that the default ports are busy, launch it on a specific port:

```bash
GRADIO_SERVER_PORT=7861 python code/demo.py
```

Then open the local URL printed by Gradio in your browser.

## Classical baselines
Train and evaluate the TF-IDF + Naive Bayes / SVM baselines:

```bash
python code/baseline.py
```

This script trains both models, prints a classification report for each one, and displays confusion matrices.

## ModernBERT fine-tuning
Train the main ModernBERT classifier:

```bash
python code/train_modernbert.py
```

This script fine-tunes `answerdotai/ModernBERT-base` on `misc/school_email_labeled.csv` and saves the model to `models/modern_BERT/`.

Prerequisites:
- Hugging Face download access
- Enough disk space for the model and checkpoints
- GPU strongly recommended for practical training speed

Evaluate the saved ModernBERT model:

```bash
python code/evaluate_modernbert.py
```

This uses the same held-out 20% split and writes the confusion matrix to `models/bert_confusion_matrix_600.png`.

Prerequisites:
- `models/modern_BERT/` must already exist from a completed training run


## DeBERTa fine-tuning and evaluation
Train and evaluate the DeBERTa model:

```bash
python code/deBerta_fineTuned.py
```

This script trains `microsoft/deberta-v3-base`, saves the best checkpoint to `best_model/`, writes a confusion matrix to `confusion_matrix.png`, and saves summary metrics to `test_metrics.json`.

Prerequisites:
- Hugging Face download access
- GPU recommended for practical training speed

## Pretrained OLMo prompting evaluation
Evaluate the DSPy + OLMo prompting pipelines:

```bash
python code/pretrained_decoder.py
```

Prerequisites:
- Set `DECODER_API_KEY` in your environment or `.env`.
- The script uses the remote model `huggingface/allenai/Olmo-3-7B-Instruct`.
- Internet access is required for remote inference.

Outputs are written under `models/` for saved prompt programs and under `metrics/` for confusion matrices and metric reports.

## Raw Gemma classification
Evaluate the raw `google/gemma-7b-it` causal language model on the labelled dataset:

```bash
python code/eval_gemma_baseline.py
```

The script uses `misc/school_email_labeled.csv` and, if present, also includes `misc/school_email_balanced_300_labeled_part2.csv`. It prints prediction progress, accuracy, macro precision, macro recall, macro F1, and the confusion matrix. GPU is strongly recommended for loading the 7B model.

Prerequisites:
- Hugging Face access approval for `google/gemma-7b-it`
- Local Hugging Face authentication via `huggingface-cli login`
- Significant RAM or GPU memory; GPU is strongly recommended

## Notes
- The transformer-based scripts download model weights from Hugging Face on first run, so internet access and sufficient disk space are required unless the models are already cached.
- `code/pretrained_decoder.py` additionally requires valid API access through `DECODER_API_KEY`.
- In the current environment, `code/train_modernbert.py` and `code/deBerta_fineTuned.py` were verified to start correctly under Python 3.12 after dependency and compatibility fixes.
- `code/evaluate_modernbert.py` still depends on a completed ModernBERT training run because it expects `models/modern_BERT/` to exist.
