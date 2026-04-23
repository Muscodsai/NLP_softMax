import baseline
import pretrained_decoder
from dataset import create_predict_dataset, load_test_examples
import gradio as gr

baseline_models = ["Naive Bayes", "SVM"]
decoder_modes = ["0_shot", "few_shot", "chain_of_thought", "few_shot_CoT"]
test_examples = load_test_examples(0, 2)


def demo_baseline() -> gr.Tab:
    """
    Loads baseline model gradio demo in a tab
    """
    models_cache = {}

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
            
            if "models" not in models_cache:
                fitted_models, tf_idf_subject_vectoriser, tf_idf_body_vectoriser = baseline.train_baseline()
                models_cache["models"] = (
                    fitted_models,
                    tf_idf_subject_vectoriser,
                    tf_idf_body_vectoriser,
                )
            else:
                fitted_models, tf_idf_subject_vectoriser, tf_idf_body_vectoriser = models_cache["models"]

            return baseline.predict_label(
                create_predict_dataset(subject, body),
                fitted_models[baseline_type],
                tf_idf_subject_vectoriser,
                tf_idf_body_vectoriser,
            )


        with gr.Row():
            with gr.Column():
                baseline_type = gr.Radio(baseline_models, label="Baseline Model Type")
                subject = gr.Textbox(label="Email Subject", lines=2)
                body = gr.Textbox(label="Email Body", lines=8)

                btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                label = gr.Text(label="Predicted Label")
                proba = gr.DataFrame(column_count=0, label="Class Proba")

        btn.click(
            demo_predict,
            inputs=[baseline_type, subject, body],
            outputs=[label, proba],
        )

        gr.Markdown("---")

        examples=[
            [model, example[0], example[1]]
            for model in baseline_models for example in test_examples
        ]

        for ex in examples:
            b = gr.Button(f"[{ex[0]} model] {ex[1][:100]}...", variant="secondary")

            def load_example(e=ex):
                return e

            b.click(
                load_example,
                outputs=[baseline_type, subject, body],
            )
        
    return demo


def demo_olmo() -> gr.Tab:
    """
    Loads OLMo model gradio demo in a tab
    """

    with gr.Tab("Pretrained OLMo Model Demo") as demo:

        def demo_predict_olmo(mode: str | None, subject: str, body: str):
            """
            Predict the demo data

            Parameters
                mode:          the prompting method to use
                subject:       subject to predict
                body:          body to predict

            Returns
                label: predicted label
                reasoning: reasoning behind predicting the label (if applicable)
            """
            if mode is None:
                return None, None
            
            decoder_input = create_predict_dataset(subject, body)
            label, ext = pretrained_decoder.predict_label(decoder_input, mode)

            if mode in ["chain_of_thought", "few_shot_CoT"]:
                reasoning = ext["reasoning"][0]
            else:
                reasoning = "The chosen prompting method does not result in reasoning"

            return label, reasoning


        with gr.Row():
            with gr.Column():
                modes = gr.Radio(decoder_modes, label="Prompting Method")
                subject = gr.Textbox(label="Email Subject", lines=2)
                body = gr.Textbox(label="Email Body", lines=8)

                btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                label = gr.Text(label="Predicted Label")
                reasoning = gr.Text(label="Reasoning")

        btn.click(
            demo_predict_olmo,
            inputs=[modes, subject, body],
            outputs=[label, reasoning],
        )

        gr.Markdown("---")

        examples=[
            [mode, example[0], example[1]]
            for mode in decoder_modes for example in test_examples
        ]

        for ex in examples:
            b = gr.Button(f"[{ex[0]} mode] {ex[1][:100]}...", variant="secondary")

            def load_example(e=ex):
                return e

            b.click(
                load_example,
                outputs=[modes, subject, body],
            )

    return demo


def run_demo():
    """
    Runs a gradio demo
    """
    with gr.Blocks() as demo:
        demo_baseline()

        demo_olmo()

        with gr.Tab("..."):
            pass

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    run_demo()
