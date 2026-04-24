# NLP_softMax
NLP project on email message categorization


## Setup
1. Create Python venv
```
python3 -m venv venv
```
2. Activate Python venv
```
source venv/bin/activate
```
3. Install dependencies
```
pip install -r misc/requirements.txt
```

## Demo
```
python code/demo.py
```

## Training and evaluation for fine tuned ModernBERT
Use:

```
python code/train_modernbert.py
```

This script trains ModernBERT on the split 600-sample dataset and saves the fine-tuned model to `models/modern_BERT`.

To evaluate the fine-tuned model and save a confusion matrix:

```
python code/evaluate_modernbert.py
```

The evaluation script uses the held-out 20% test split from the 600 samples and writes the confusion matrix image under `models/`.

## Raw Gemma classification
Use the raw `google/gemma-7b-it` causal language model to classify held-out test emails with prompts:

```
python code/classify_raw_gemma.py --num_examples 10
```

This script uses the existing `misc/school_email_labeled.csv` test split and prints raw Gemma predictions, accuracy, macro F1, and a confusion matrix. GPU is recommended for loading the 7B model.