from dataset import load_dataset, get_dataset_X_y, create_predict_dataset
import pandas as pd
from typing import Literal
from dotenv import load_dotenv
import os

import dspy

train, test = load_dataset()
classes = ['project_development', 'coursework', 'discussion_forums', 'university_admin', 'opportunities_events']
load_dotenv()
decoder_api_key = os.getenv("DECODER_API_KEY")

class ClassifyEmail(dspy.Signature):
    """Select the most likely category that this email is in."""

    subject: str = dspy.InputField()
    body: str = dspy.InputField()
    label: Literal['project_development', 'coursework', 'discussion_forums', 'university_admin', 'opportunities_events'] = dspy.OutputField()

decoder_model = dspy.LM('huggingface/allenai/Olmo-3-7B-Instruct', api_key=decoder_api_key)
dspy.configure(lm=decoder_model)

classify_email_0shot = dspy.Predict(ClassifyEmail)
classify_email_CoT = dspy.ChainOfThought(ClassifyEmail)

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

def predict_label(data: pd.DataFrame, mode: str) -> tuple[str, pd.DataFrame]:
    match mode:
        case "0_shot":
            response = classify_email_0shot(subject=data["subject"][0], body=data["body"][0])
            label = response.label
            extra_info = pd.DataFrame(
                {
                    "none": ["No extra info"],
                }
            )
        case "few_shot":
            # TODO. Will need to use DSPY optimisers.
            pass
        case "chain_of_thought":
            response = classify_email_CoT(subject=data["subject"][0], body=data["body"][0])
            label = response.label
            extra_info = pd.DataFrame(
                {
                    "reasoning": [response.reasoning],
                }
            )
        case _:
            label = "Error: Mode must be either 0_shot, few_shot, or chain_of_thought"
            extra_info = pd.DataFrame(
                {
                    "none": ["No extra info"],
                }
            )

    return label, extra_info


# label, ext = predict_label(input, "0_shot")
# print(label)
# print(ext)
# label, ext = predict_label(input, "chain_of_thought")
# print(label)
# print(ext)
