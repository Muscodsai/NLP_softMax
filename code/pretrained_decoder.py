from dataset import load_dataset, get_dataset_X_y, create_predict_dataset
import pandas as pd
from typing import Literal
from dotenv import load_dotenv
import os
import dspy
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import time

# Import data from dataset (will be needed later for few shot prompting examples)
train, test = load_dataset()

# Initialise constants and variables, and select model API to use
MAX_TRAINING_EXAMPLES = 50
model_name = 'huggingface/allenai/Olmo-3-7B-Instruct'

load_dotenv()
decoder_api_key = os.getenv("DECODER_API_KEY")
decoder_model = dspy.LM(model_name, api_key=decoder_api_key)
dspy.configure(lm=decoder_model)

classes = ['project_development', 'coursework', 'discussion_forums', 'university_admin', 'opportunities_events']


# Signature class for DSPY email classification
class ClassifyEmail(dspy.Signature):
    """Select the most likely category that this email is in."""

    subject: str = dspy.InputField()
    body: str = dspy.InputField()
    label: Literal['project_development', 'coursework', 'discussion_forums', 'university_admin', 'opportunities_events'] = dspy.OutputField()


# Metric used by BootstrapFewShot to determine if a prediction is correct
def accuracy_metric(expected, predicted, trace=None):
    return expected.label == predicted.label


# Set up examples for few shot prompting
few_shot_examples = []

for i, row in enumerate(train.itertuples()):
    if i >= MAX_TRAINING_EXAMPLES:
        # We don't need the entire dataset to bootstrap few shot prompts
        break

    example = dspy.Example(subject=row.subject, body=row.body, label=row.label).with_inputs('subject', 'body')
    few_shot_examples.append(example)

# Optimiser for few shot prompts
few_shot_optimiser = dspy.BootstrapFewShot(accuracy_metric)

# Create functions which predict the email type using various prompting methods
classify_email_0shot = dspy.Predict(ClassifyEmail)
classify_email_0shot.save("models/decoder_0_shot.json")

classify_email_CoT = dspy.ChainOfThought(ClassifyEmail)
classify_email_CoT.save("models/decoder_CoT.json")

if os.path.isfile("models/decoder_few_shot.json"):
    classify_email_few_shot = dspy.Predict(ClassifyEmail)
    classify_email_few_shot.load("models/decoder_few_shot.json")
else:
    classify_email_few_shot = few_shot_optimiser.compile(classify_email_0shot, trainset=few_shot_examples)
    classify_email_few_shot.save("models/decoder_few_shot.json")

if os.path.isfile("models/decoder_few_shot_CoT.json"):
    classify_email_few_shot_CoT = dspy.ChainOfThought(ClassifyEmail)
    classify_email_few_shot_CoT.load("models/decoder_few_shot_CoT.json")
else:
    classify_email_few_shot_CoT = few_shot_optimiser.compile(classify_email_CoT, trainset=few_shot_examples)
    classify_email_few_shot_CoT.save("models/decoder_few_shot_CoT.json")


def predict_label(data: pd.DataFrame, mode: str) -> tuple[str, pd.DataFrame]:
    """
    Predicts the label using a pre-trained decoder model with prompt engineering

    Parameters
        data: feature matrix of data to predict
        mode: the prompting method to use (eg. 0_shot, few_shot, chain_of_thought)

    Returns
        label: predicted label
        prompt: the optimised prompt which DSPY created and sent to the model
        reasoning (for chain_of_thought): the model's reasoning behind selecting the label
    """
    match mode:
        case "0_shot":
            # Zero shot prompting. Simply give the model an email and ask it to
            # classify it.
            response = classify_email_0shot(subject=data["subject"][0], body=data["body"][0])
            label = response.label
            extra_info = pd.DataFrame(
                {
                    "prompt": [dspy.settings.lm.history[-1]["messages"]],
                }
            )
        case "few_shot":
            # Few shot prompting. Provide the model with a few examples of
            # correctly classified emails before asking it to classify the
            # input email
            response = classify_email_few_shot(subject=data["subject"][0], body=data["body"][0])
            label = response.label
            extra_info = pd.DataFrame(
                {
                    "prompt": [dspy.settings.lm.history[-1]["messages"]],
                }
            )
        case "chain_of_thought":
            # Chain of thought prompting. Get the model to work through its
            # reasoning step by step before predicting the email type,
            # potentially improving results
            response = classify_email_CoT(subject=data["subject"][0], body=data["body"][0])
            label = response.label
            extra_info = pd.DataFrame(
                {
                    "prompt": [dspy.settings.lm.history[-1]["messages"]],
                    "reasoning": [response.reasoning],
                }
            )
        case "few_shot_CoT":
            # Combines both few shot prompting and chain of thought. The few
            # shot examples will include examples of reasoning, which are
            # generated during the bootstrap process.
            response = classify_email_few_shot_CoT(subject=data["subject"][0], body=data["body"][0])
            label = response.label
            extra_info = pd.DataFrame(
                {
                    "prompt": [dspy.settings.lm.history[-1]["messages"]],
                    "reasoning": [response.reasoning],
                }
            )
        case _:
            # The mode passed to the function is invalid
            label = "Error: Mode must be either 0_shot, few_shot, chain_of_thought, or few_shot_CoT"
            extra_info = pd.DataFrame(
                {
                    "error": ["Error"],
                }
            )

    return label, extra_info


# ---------------------------------------------------------------------------
# TESTING
# - Note: Do not run these tests too often, the API we're using has a limited
#         number of requests per month
# ---------------------------------------------------------------------------
def run_testing():
    modes = ["0_shot", "few_shot", "chain_of_thought", "few_shot_CoT"]

    actual_labels = []
    preds_0_shot = []
    preds_few_shot = []
    preds_CoT = []
    preds_few_shot_CoT = []

    prediction_index = [preds_0_shot, preds_few_shot, preds_CoT, preds_few_shot_CoT]

    num_test_cases = test.shape[0]

    print("Starting tests...")
    print(f"Number of test examples: {num_test_cases}")

    # Test every email in our test set with each of the 4 prompting modes
    for i, row in enumerate(test.itertuples()):
        print(f"Testing example {i} of {num_test_cases}...")

        skip = False
        body_words = row.body.split()

        # Check if the test example is suitable (OLMo does not like certain inputs
        # if the token is too long)
        for word in body_words:
            if len(word) > 100:
                print(f"Skipping example {i} of {num_test_cases}: Contains a token which is too long")
                skip = True
                break

        if not skip:
            try:
                input = create_predict_dataset(row.subject, row.body)
                actual_label = row.label
                pred_0_shot = predict_label(input, "0_shot")[0]
                pred_few_shot = predict_label(input, "few_shot")[0]
                pred_CoT = predict_label(input, "chain_of_thought")[0]
                pred_few_shot_CoT = predict_label(input, "few_shot_CoT")[0]
            except:
                # Ignore inputs that result in a gateway timeout, and continue
                continue

            # Timeout to prevent API from being overloaded
            time.sleep(0.1)

        actual_labels.append(actual_label)
        preds_0_shot.append(pred_0_shot)
        preds_few_shot.append(pred_few_shot)
        preds_CoT.append(pred_CoT)
        preds_few_shot_CoT.append(pred_few_shot_CoT)

    # Calculate metrics for each of the 4 prompting modes
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    print("Computing metrics...")

    for i in range(len(modes)):
        accuracies.append(accuracy_score(actual_labels, prediction_index[i]))
        precisions.append(precision_score(actual_labels, prediction_index[i], average="macro", labels=classes, zero_division=0))
        recalls.append(recall_score(actual_labels, prediction_index[i], average="macro", labels=classes, zero_division=0))
        f1s.append(f1_score(actual_labels, prediction_index[i], average="macro", labels=classes, zero_division=0))

    # Print and save metrics, and save confusion matrices
    metric_template = """OLMO test metrics for the {} prompting method:
    Accuracy: {}
    Precision: {}
    Recall: {}
    F1 Score: {}"""

    for i in range(len(modes)):
        metric_string = metric_template.format(modes[i], accuracies[i], precisions[i], recalls[i], f1s[i])

        print()
        print(metric_string)

        with open(f"metrics/olmo_{modes[i]}_metrics.txt", "w", encoding="utf-8") as file:
            file.write(metric_string)

        confusion = confusion_matrix(actual_labels, prediction_index[i], labels=classes)
        confusion_plot = ConfusionMatrixDisplay(confusion, display_labels=classes)
        fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
        confusion_plot.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(f"Confusion Matrix: OLMo with\n{modes[i]} prompting", fontsize=13, pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.tight_layout()
        plt.savefig(f"metrics/olmo_{modes[i]}_confusion_matrix.png", dpi=150)


# Only run the tests if this file was executed as the main program
if __name__ == "__main__":
    run_testing()
