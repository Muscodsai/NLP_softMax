import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]

# Load combined data
train_path = ROOT_DIR / "misc" / "school_email_labeled.csv"
if not train_path.exists():
    train_path = Path('/home/qingb/school_email_labeled.csv')
part2_path = ROOT_DIR / "misc" / "school_email_balanced_300_labeled_part2.csv"
if not part2_path.exists():
    part2_path = Path('/home/qingb/school_email_balanced_300_labeled_part2.csv')

train_df = pd.read_csv(train_path).fillna('')
part2_df = pd.read_csv(part2_path).fillna('')
all_df = pd.concat([train_df, part2_df], ignore_index=True)
all_df['text'] = 'Subject: ' + all_df['subject'] + '\n\nBody: ' + all_df['body']

le = LabelEncoder()
all_df['label_id'] = le.fit_transform(all_df['label'])
labels = list(le.classes_)
label2id = {name: i for i, name in enumerate(labels)}

train_val_df, test_df = train_test_split(all_df, test_size=0.2, stratify=all_df['label_id'], random_state=42)

print('Test sample count:', len(test_df))
print('Labels:', labels)


def build_prompt(text):
    subject = text.split('Subject: ')[1].split('\n\nBody: ')[0]
    body = text.split('\n\nBody: ')[1]
    return f"""Classify the following email into exactly one label from:\n{', '.join(labels)}\n\nSubject: {subject}\n\nBody:\n{body}\n\nAnswer:"""


def predict_with_model(model, tokenizer, text, device):
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_part = generated.split('Answer:')[-1].strip()
    for label in labels:
        if label in answer_part:
            return label2id[label]
    return -1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

base_tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b-it')
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-7b-it',
    device_map='auto' if torch.cuda.is_available() else None,
    dtype=torch.bfloat16 if torch.cuda.is_available() else None,
)
base_model.eval()
if device.type == 'cuda':
    base_model = base_model.to(device)

predictions = []
for i, row in enumerate(test_df.itertuples(index=False), 1):
    pred = predict_with_model(base_model, base_tokenizer, row.text, device)
    predictions.append(pred)
    if i % 10 == 0 or i == len(test_df):
        print(f'Evaluated {i}/{len(test_df)} samples', end='\r')

true_labels = test_df['label_id'].tolist()
valid_indices = [i for i, p in enumerate(predictions) if p != -1]
valid_preds = [predictions[i] for i in valid_indices]
valid_true = [true_labels[i] for i in valid_indices]

print('\nValid predictions:', len(valid_preds), 'out of', len(predictions))
if not valid_preds:
    raise ValueError('No valid predictions were generated.')

accuracy = accuracy_score(valid_true, valid_preds)
macro_precision = precision_score(valid_true, valid_preds, average='macro', zero_division=0)
macro_recall = recall_score(valid_true, valid_preds, average='macro', zero_division=0)
macro_f1 = f1_score(valid_true, valid_preds, average='macro', zero_division=0)
cm = confusion_matrix(valid_true, valid_preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'Macro Precision: {macro_precision:.4f}')
print(f'Macro Recall: {macro_recall:.4f}')
print(f'Macro F1: {macro_f1:.4f}')
print('Confusion matrix:')
print(cm)
