import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the dataset from the csv file.
    Datset with subject, body and label columns.

    Returns
        train_data: train portion of dataset
        test_data:  test portion of dataset
    """
    data = pd.read_csv("misc/school_email_labeled.csv")
    train_data, test_data = train_test_split(data, train_size=0.8, random_state=42)
    return train_data, test_data


def get_dataset_X_y(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits dataset into features and label

    Parameters
        dataset: full dataset

    Returns
        X: features of dataset
        y: labels of dataset
    """
    return dataset.drop(columns=["label"]), dataset["label"]


def create_predict_dataset(subject: str, body: str) -> pd.DataFrame:
    """
    Creates a simple dataframe for prediction

    Parameters
        subject: subject of email
        body:    body of email

    Returns
        dataset: dataframe with features only
    """
    return pd.DataFrame({"subject": [subject], "body": [body]})


def load_test_examples(start: int, end: int) -> list[tuple[str, str]]:
    """
    Loads test data for demo evaluation

    Parameters
        start: start index to retrieve
        end:   end index to retrieve

    Returns
        examples: list of examples, which have subject and body
    """
    _, test_dataset = load_dataset()
    dataset_X, _ = get_dataset_X_y(test_dataset)
    examples = dataset_X.iloc[start:end]
    return list(examples.itertuples(index=False, name=None))
