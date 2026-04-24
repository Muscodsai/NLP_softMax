import baseline
import pretrained_decoder
from dataset import create_predict_dataset, load_test_examples
import gradio as gr
import pandas as pd
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

baseline_models = ["Naive Bayes", "SVM"]
decoder_modes = ["0_shot", "few_shot", "chain_of_thought", "few_shot_CoT"]
test_examples = load_test_examples(0, 2)
ROOT_DIR = Path(__file__).resolve().parents[1]
MODERNBERT_DIR = ROOT_DIR / "models" / "modern_BERT"
DATA_PATH = ROOT_DIR / "misc" / "school_email_labeled.csv"
GEMMA_MODEL_ID = "google/gemma-7b-it"


def patch_torch_compile_for_python_312():
    # modernBERT imports can fail on Python 3.12 unless torch.compile is patched
    if sys.version_info < (3, 12) or not hasattr(torch, "compile"):
        return

    try:
        torch.compile(lambda x: x)
    except RuntimeError as exc:
        if "Python 3.12+" not in str(exc):
            raise

        def _noop_compile(model=None, *args, **kwargs):
            if model is None:
                return lambda fn: fn
            return model

        torch.compile = _noop_compile


patch_torch_compile_for_python_312()


def build_input_text(subject: str, body: str) -> str:
    return f"Subject: {subject}\n\nBody: {body}"


def load_labels() -> list[str]:
    data = pd.read_csv(DATA_PATH).fillna("")
    return sorted(data["label"].unique().tolist())


LABELS = load_labels()
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}


def resolve_modernbert_path() -> Path:
    # Prefer the exported model directory, but fall back to the latest checkpoint if needed.
    if (MODERNBERT_DIR / "config.json").is_file():
        return MODERNBERT_DIR

    checkpoints = sorted(
        [path for path in MODERNBERT_DIR.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(
        "No ModernBERT model found under models/modern_BERT. Run code/train_modernbert.py first."
    )


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
            
            # Train the classical baselines only once per app session, then reuse them.
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


def demo_modernbert() -> gr.Tab:
    """
    Loads the fine-tuned ModernBERT demo in a tab.
    """
    model_cache = {}

    with gr.Tab("ModernBERT Demo") as demo:

        def demo_predict_modernbert(subject: str, body: str):
            # Lazily load the local checkpoint so the app can start before model weights are needed.
            if "model" not in model_cache:
                model_path = resolve_modernbert_path()
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    use_safetensors=True,
                )
                model.eval()
                model_cache["model"] = (tokenizer, model)
            else:
                tokenizer, model = model_cache["model"]

            # Use the same input text template as ModernBERT training and evaluation.
            text = build_input_text(subject, body)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                outputs = model(**inputs)

            probabilities = torch.softmax(outputs.logits[0], dim=0).cpu().numpy()
            predicted_id = int(probabilities.argmax())
            predicted_label = model.config.id2label.get(predicted_id, str(predicted_id))
            detailed = pd.DataFrame(
                {
                    "class": [model.config.id2label.get(idx, str(idx)) for idx in range(len(probabilities))],
                    "proba": probabilities,
                }
            ).sort_values("proba", ascending=False)

            return predicted_label, detailed

        with gr.Row():
            with gr.Column():
                subject = gr.Textbox(label="Email Subject", lines=2)
                body = gr.Textbox(label="Email Body", lines=8)
                btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                label = gr.Text(label="Predicted Label")
                proba = gr.DataFrame(column_count=0, label="Class Proba")

        btn.click(
            demo_predict_modernbert,
            inputs=[subject, body],
            outputs=[label, proba],
        )

        gr.Markdown("---")

        for subject_text, body_text in test_examples:
            b = gr.Button(f"{subject_text[:100]}...", variant="secondary")

            def load_example(example_subject=subject_text, example_body=body_text):
                return example_subject, example_body

            b.click(
                load_example,
                outputs=[subject, body],
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
            
            # The DSPy pipeline is already initialized in pretrained_decoder, so the demo just forwards inputs.
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


def demo_gemma() -> gr.Tab:
    """
    Loads the raw Gemma prompt-based classifier demo in a tab.
    """
    model_cache = {}

    with gr.Tab("Gemma Demo") as demo:

        def build_prompt(subject: str, body: str) -> str:
            return (
                "Classify the following email into exactly one label from:\n"
                + ", ".join(LABELS)
                + f"\n\nSubject: {subject}\n\nBody:\n{body}\n\nAnswer:"
            )

        def demo_predict_gemma(subject: str, body: str):
            # Gemma is loaded on first use because it is the heaviest demo dependency.
            if "model" not in model_cache:
                tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    GEMMA_MODEL_ID,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                )
                model.eval()
                model_cache["model"] = (tokenizer, model)
            else:
                tokenizer, model = model_cache["model"]

            # This prompt mirrors the benchmark path for the raw Gemma classifier.
            prompt = build_prompt(subject, body)
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_part = generated.split("Answer:")[-1].strip()
            predicted_label = next((label for label in LABELS if label in answer_part), "unknown")

            return predicted_label, answer_part

        with gr.Row():
            with gr.Column():
                subject = gr.Textbox(label="Email Subject", lines=2)
                body = gr.Textbox(label="Email Body", lines=8)
                btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                label = gr.Text(label="Predicted Label")
                raw_output = gr.Text(label="Raw Model Answer")

        btn.click(
            demo_predict_gemma,
            inputs=[subject, body],
            outputs=[label, raw_output],
        )

        gr.Markdown("Gemma requires approved Hugging Face access and local authentication.")
        gr.Markdown("---")

        for subject_text, body_text in test_examples:
            b = gr.Button(f"{subject_text[:100]}...", variant="secondary")

            def load_example(example_subject=subject_text, example_body=body_text):
                return example_subject, example_body

            b.click(
                load_example,
                outputs=[subject, body],
            )

    return demo


def run_demo():
    """
    Runs a gradio demo
    """
    with gr.Blocks() as demo:
        demo_baseline()
        demo_modernbert()

        demo_olmo()
        demo_gemma()

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    run_demo()
