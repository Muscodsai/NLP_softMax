from dataset import load_dataset, get_dataset_X_y, create_predict_dataset
import pandas as pd
from typing import Literal
from dotenv import load_dotenv
import os
import dspy

# Import data from dataset (will be needed later for few shot prompting examples)
train, test = load_dataset()

# Initialise constants and variables, and select model API to use
MAX_TRAINING_EXAMPLES = 50

load_dotenv()
decoder_api_key = os.getenv("DECODER_API_KEY")
decoder_model = dspy.LM('huggingface/allenai/Olmo-3-7B-Instruct', api_key=decoder_api_key)
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
classify_email_CoT = dspy.ChainOfThought(ClassifyEmail)
classify_email_few_shot = few_shot_optimiser.compile(classify_email_0shot, trainset=few_shot_examples)
classify_email_few_shot_CoT = few_shot_optimiser.compile(classify_email_CoT, trainset=few_shot_examples)


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

            print(response)
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

            print(response)
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

test_subj = "COMP9417-COMP[PHONE]_00122: Week 6 updates"
test_body="""COMP9417-COMP[PHONE]_00122
»
Forums
»
Announcements
»
Week 6 updates
Week 6 updates
by
Omar Al-Ghattas
- Monday, 24 March 2025, 3:59 PM
Hi all,
Please take note of the following
1. This week is flex week, so there are no lectures/tutorials.
2. I will still run help sessions on Wednesday midday and Saturday at 10am on Teams.
3. The group project spec is now available under the group project heading on moodle.
Best,
Omar
See this post in context
Change your forum digest preferences"""

input = create_predict_dataset(test_subj, test_body)

print("0 SHOT:")
label, ext = predict_label(input, "0_shot")
print(f"Label: {label}")
print()

print("FEW SHOT:")
label, ext = predict_label(input, "few_shot")
print(f"Label: {label}")
print()

print("CHAIN OF THOUGHT:")
label, ext = predict_label(input, "chain_of_thought")
print(f"Label: {label}")
print(f"Reasoning: {ext["reasoning"][0]}")
print()

print("FEW SHOT + CHAIN OF THOUGHT:")
label, ext = predict_label(input, "few_shot_CoT")
print(f"Label: {label}")
print(f"Reasoning: {ext["reasoning"][0]}")
