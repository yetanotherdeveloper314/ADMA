"""
ADMA Web Application

A Gradio-based interface for detecting military assets in uploaded images.
Supports switching between multiple trained models via a dropdown.

Run locally:  python -m adma.app
"""

from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from adma.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
    MODELS_DIR,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_SHARE,
)
from adma.config import (
    EXAMPLES_DIR as _EXAMPLES_DIR,
)
from adma.detector import MilitaryAssetDetector, discover_models

EXAMPLES_DIR = Path(_EXAMPLES_DIR)

_current_detector: MilitaryAssetDetector | None = None
_current_model_name: str | None = None


def get_detector(model_name: str) -> MilitaryAssetDetector:
    global _current_detector, _current_model_name
    if _current_detector is not None and _current_model_name == model_name:
        return _current_detector

    available = discover_models()
    if model_name not in available:
        raise FileNotFoundError(f"Model '{model_name}' not found in {MODELS_DIR}/")

    _current_detector = MilitaryAssetDetector(
        str(available[model_name]), confidence=DEFAULT_CONFIDENCE
    )
    _current_model_name = model_name
    return _current_detector


def detect_assets(
    image: np.ndarray | None,
    confidence: float,
    model_name: str,
) -> tuple[np.ndarray | None, str, str, str]:
    if image is None:
        return None, "", "Upload an image to begin.", ""

    try:
        detector = get_detector(model_name)
    except FileNotFoundError as e:
        return None, "", str(e), ""

    # Gradio provides RGB numpy arrays; detector and YOLO expect BGR (OpenCV convention)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = detector.detect(image_bgr, confidence=confidence)

    # Detector returns BGR annotated image; convert back to RGB for Gradio display
    annotated_bgr = results["annotated_image"]
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    table_md = _build_results_table(results["counts"])
    summary = results["summary"]
    classes = ", ".join(detector.class_names.values())
    model_info = f"Model: {model_name}  |  Classes: {classes}"

    return annotated_rgb, table_md, summary, model_info


def _build_results_table(counts: dict) -> str:
    if not counts:
        return ""
    rows = ["| Asset Type | Count | Avg Confidence |", "| --- | --- | --- |"]
    for cls, info in counts.items():
        avg = sum(info["confidences"]) / len(info["confidences"])
        rows.append(f"| {cls} | {info['count']} | {avg:.1%} |")
    return "\n".join(rows)


def _gather_examples() -> list[str] | None:
    if not EXAMPLES_DIR.exists():
        return None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted(p for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in exts)
    return [str(p) for p in imgs] if imgs else None


def _model_choices() -> tuple[list[str], str]:
    models = list(discover_models().keys())
    if not models:
        return ["(no models trained yet)"], "(no models trained yet)"
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
    return models, default


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Military Asset Detection",
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="gray"),
        css="""
            .header { text-align: center; margin-bottom: 0.5rem; }
            .header h1 { margin-bottom: 0.25rem; }
            .header p  { color: #666; margin-top: 0; }
            .results-summary { font-family: monospace; white-space: pre-wrap; }
        """,
    ) as demo:
        gr.HTML(
            """
            <div class="header">
                <h1>Military Asset Detection System</h1>
                <p>Upload an image to detect and identify military assets.
                   Choose a model and adjust confidence to control detection.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    sources=["upload"],
                    height=420,
                )

                choices, default_model = _model_choices()
                model_dropdown = gr.Dropdown(
                    choices=choices,
                    value=default_model,
                    label="Select Model",
                    info="Switch between models trained on different datasets.",
                )

                confidence_slider = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=DEFAULT_CONFIDENCE,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Lower = more detections (may include false positives). "
                    "Higher = fewer but more certain detections.",
                )

                detect_btn = gr.Button(
                    "Detect Assets",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Detection Results",
                    type="numpy",
                    height=420,
                    interactive=False,
                )
                results_table = gr.Markdown(label="Summary Table")
                results_text = gr.Textbox(
                    label="Detection Summary",
                    lines=4,
                    interactive=False,
                    elem_classes=["results-summary"],
                )
                model_info_text = gr.Textbox(
                    label="Active Model Info",
                    lines=1,
                    interactive=False,
                )

        examples = _gather_examples()
        if examples:
            gr.Examples(
                examples=[[img] for img in examples],
                inputs=[input_image],
                label="Example Images (click to load)",
            )

        trigger_inputs = [input_image, confidence_slider, model_dropdown]
        trigger_outputs = [output_image, results_table, results_text, model_info_text]

        detect_btn.click(fn=detect_assets, inputs=trigger_inputs, outputs=trigger_outputs)
        input_image.change(fn=detect_assets, inputs=trigger_inputs, outputs=trigger_outputs)
        confidence_slider.release(fn=detect_assets, inputs=trigger_inputs, outputs=trigger_outputs)
        model_dropdown.change(fn=detect_assets, inputs=trigger_inputs, outputs=trigger_outputs)

    return demo


def main():
    """Entry point for 'adma-app' console script and direct execution."""
    app = build_app()
    app.launch(server_name=SERVER_HOST, server_port=SERVER_PORT, share=SERVER_SHARE)


if __name__ == "__main__":
    main()
