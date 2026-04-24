from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "misc" / "school_email_labeled.csv"


def load_canonical_splits(
    test_size: float = 0.2,
    val_fraction_of_train_val: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """
    Build one canonical stratified split for fair model comparison.

    The split is:
    - 60% train
    - 20% validation
    - 20% test
    """
    all_df = pd.read_csv(DATA_PATH).fillna("")
    all_df["text"] = "Subject: " + all_df["subject"] + "\n\nBody: " + all_df["body"]

    label_encoder = LabelEncoder()
    all_df["label_id"] = label_encoder.fit_transform(all_df["label"])

    train_val_df, test_df = train_test_split(
        all_df,
        test_size=test_size,
        stratify=all_df["label_id"],
        random_state=random_state,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction_of_train_val,
        stratify=train_val_df["label_id"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        label_encoder,
    )
