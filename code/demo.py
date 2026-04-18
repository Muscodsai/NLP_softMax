from baseline import predict_label, train_baseline
from dataset import create_predict_dataset, load_test_examples
import gradio as gr


def demo_baseline() -> gr.Tab:
    """
    Loads baseline model gradio demo in a tab
    """
    fitted_models, tf_idf_subject_vectoriser, tf_idf_body_vectoriser = train_baseline()

    with gr.Tab("Baseline Model Demo") as demo:

        def demo_predict(baseline_type: str | None, subject: str, body: str):
            """
            Predict the demo data

            Parameters
                baseline_type: baseline model type
                subject:       subject to predict
                body:          body to predict

            Returns
                label: predicted label
                proba: proba of each label
            """
            if baseline_type is None:
                return None, None
            return predict_label(
                create_predict_dataset(subject, body),
                fitted_models[baseline_type],
                tf_idf_subject_vectoriser,
                tf_idf_body_vectoriser,
            )

        gr.Interface(
            fn=demo_predict,
            inputs=[
                gr.Radio(list(fitted_models.keys()), label="Baseline Model Type"),
                gr.Textbox(label="Email Subject", lines=2),
                gr.Textbox(label="Email Body", lines=8),
            ],
            outputs=[
                gr.Text(label="Predicted Label"),
                gr.DataFrame(column_count=0, label="Class Proba"),
            ],
            examples=[
                [list(fitted_models.keys())[0], example[0], example[1]]
                for example in load_test_examples(0, 2)
            ],
        )
    return demo


def run_demo():
    """
    Runs a gradio demo
    """
    with gr.Blocks() as demo:
        demo_baseline()

        with gr.Tab("..."):
            pass

    demo.launch()


if __name__ == "__main__":
    run_demo()
