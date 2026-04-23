import pandas as pd
from nltk.metrics.agreement import AnnotationTask


def find_kappa_agreement():
    """
    Finds the agreement score between the annotations

    Returns
        kappa: kappa agreement score
    """
    annotations = pd.read_csv("misc/annotation.csv")

    annotation_data = []
    for (ind, annotation_A_label), (_, annotation_B_label) in zip(
        annotations["annotation_A"].items(), annotations["annotation_B"].items()
    ):
        annotation_data.append(("A", ind, annotation_A_label))
        annotation_data.append(("B", ind, annotation_B_label))

    annotation_task = AnnotationTask(data=annotation_data)
    return annotation_task.kappa()


if __name__ == "__main__":
    print(find_kappa_agreement())
