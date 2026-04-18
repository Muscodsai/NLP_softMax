from dataset import load_dataset, get_dataset_X_y
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

type BaselineModel = MultinomialNB | SVC


def get_baseline_models() -> dict[str, BaselineModel]:
    """
    Retrieves baseline models

    Returns
        baseline_model: dict of baseline model name mapped to the model
    """
    return {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(probability=True, random_state=42),
    }


def fit_tf_idf_vectoriser(x: pd.Series) -> TfidfVectorizer:
    """
    Fits a tf-idf vectoriser

    Parameters
        x: series containing sentences

    Returns
        tf_idf_vectorier: fitted tf-idf vectoriser
    """
    tf_idf_vectorier = TfidfVectorizer(stop_words="english")
    return tf_idf_vectorier.fit(x)


def transform_X_tf_idf(
    X: pd.DataFrame,
    tf_idf_subject_vectoriser: TfidfVectorizer,
    tf_idf_body_vectoriser: TfidfVectorizer,
) -> np.ndarray:
    """
    Transforms initial feature matrix into tf-idf vectorised form

    Parameters
        X:                         feature matrix
        tf_idf_subject_vectoriser: subject tf-idf vectoriser
        tf_idf_body_vectoriser:    body tf-idf vectoriser

    Returns
        X_transformed: tf-idf transformed feature matrix
    """
    return sparse.hstack(
        [
            tf_idf_subject_vectoriser.transform(X["subject"]),
            tf_idf_body_vectoriser.transform(X["body"]),
        ]
    ).toarray()


def predict_label(
    predict_data: pd.DataFrame,
    model: BaselineModel,
    tf_idf_subject_vectoriser: TfidfVectorizer,
    tf_idf_body_vectoriser: TfidfVectorizer,
) -> tuple[str, pd.DataFrame]:
    """
    Predicts label using baseline model

    Parameters
        predict_data:              feature matrix of data to predict
        model:                     baseline model to use
        tf_idf_subject_vectoriser: subject tf-idf vectoriser
        tf_idf_body_vectoriser:    body tf-idf vectoriser

    Returns
        label: predicted label
        proba: proba of each label
    """
    predict_X_transformed = transform_X_tf_idf(
        predict_data, tf_idf_subject_vectoriser, tf_idf_body_vectoriser
    )
    label = model.predict(predict_X_transformed)[0]
    prediction_detailed = pd.DataFrame(
        {
            "class": model.classes_,
            "proba": model.predict_proba(predict_X_transformed)[0],
        }
    )
    return label, prediction_detailed


def train_baseline() -> tuple[
    dict[str, BaselineModel], TfidfVectorizer, TfidfVectorizer
]:
    """
    Trains all baseline models

    Returns
        trained_baseline_models:   dict mapping baseline model name to the trained model
        tf_idf_subject_vectoriser: subject tf-idf vectoriser
        tf_idf_body_vectoriser:    body tf-idf vectoriser
    """
    train_data, test_data = load_dataset()
    train_X, train_y = get_dataset_X_y(train_data)
    test_X, test_y = get_dataset_X_y(test_data)

    tf_idf_subject_vectoriser = fit_tf_idf_vectoriser(train_X["subject"])
    tf_idf_body_vectoriser = fit_tf_idf_vectoriser(train_X["body"])

    train_X_transformed = transform_X_tf_idf(
        train_X, tf_idf_subject_vectoriser, tf_idf_body_vectoriser
    )
    test_X_transformed = transform_X_tf_idf(
        test_X, tf_idf_subject_vectoriser, tf_idf_body_vectoriser
    )

    trained_baseline_models: dict[str, BaselineModel] = {}
    for model_name, model in get_baseline_models().items():
        model.fit(train_X_transformed, train_y)

        pred_test_y = model.predict(test_X_transformed)
        print(classification_report(test_y, pred_test_y))

        trained_baseline_models[model_name] = model

    return trained_baseline_models, tf_idf_subject_vectoriser, tf_idf_body_vectoriser


if __name__ == "__main__":
    train_baseline()
