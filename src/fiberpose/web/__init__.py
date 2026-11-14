"""Gradio web interface: ``fiberpose --web``.

``build_demo(config)`` only describes the page; the caller decides how to serve it.
Analysis settings come from the same config file as the GUI and CLI.
Host, port and proxy path come from Gradio's own environment variables
(``GRADIO_SERVER_NAME``, ``GRADIO_SERVER_PORT``, ``GRADIO_ROOT_PATH``);
``FIBERPOSE_MAX_UPLOAD`` caps the upload size (default ``50mb``).
"""

from __future__ import annotations

import os
from dataclasses import replace

import gradio as gr

from ..api import FiberposeError
from ..domain import AppConfig
from ..paths import EXAMPLE_IMAGE_PATH
from . import callbacks as cb

__all__ = ["build_demo", "launch"]


def build_demo(config: AppConfig | None = None) -> gr.Blocks:
    config = config or AppConfig(use_gpu=False)

    def run(path, pixel_size, unit, invert, grad_threshold):
        if not path:
            raise gr.Error("Upload an image first.")
        settings = replace(config, watershed_grad_threshold=grad_threshold)
        try:
            result = cb.analyze_file(path, settings, pixel_size, unit, invert)
        except FiberposeError as exc:
            raise gr.Error(str(exc)) from exc
        if result.message:
            gr.Info(result.message)
        return result, cb.summary_rows(result), cb.histograms(result), cb.samples_csv(result)

    with gr.Blocks(title="Fiberpose") as demo:
        gr.Markdown("# Fiberpose\nSegmentation and morphometry of fibrous microstructures.")
        result = gr.State()  # the AnalysisResult of this browser session
        with gr.Row(equal_height=True):  # settings, each above the image it applies to
            with gr.Row(variant="panel"):  # segmentation settings
                pixel_size = gr.Number(1.0, label="Pixel size", minimum=0, min_width=80)
                unit = gr.Textbox("px", label="Unit", min_width=60)
                grad_threshold = gr.Number(
                    config.watershed_grad_threshold, label="|∇DT| threshold", minimum=0, maximum=1,
                    step=0.05, min_width=80,
                )
                invert = gr.Checkbox(label="Invert (dark fibers)", min_width=80)
            with gr.Row(variant="panel"):
                view = gr.Radio(cb.VIEWS, value="Diameters", label="View")
        with gr.Row(equal_height=True):
            image = gr.Image(type="filepath", label="Image", height=480)
            output = gr.Image(label="Result", interactive=False, height=480)
        with gr.Row():
            with gr.Column():  # under the input
                with gr.Row():
                    button = gr.Button("Analyze", variant="primary")
                    gr.HTML()  # empty half: keeps the button at half width
                gr.Examples([[str(EXAMPLE_IMAGE_PATH)]], inputs=image)
                summary = gr.Dataframe(headers=["Metric", "Value"], label="Summary", interactive=False)
                csv = gr.File(label="Measurements (CSV)")
            plots = gr.Plot(label="Distributions")  # under the result, same width

        inputs = [image, pixel_size, unit, invert, grad_threshold]
        analysed = button.click(run, inputs, [result, summary, plots, csv], api_name="analyze")
        analysed.success(cb.view_image, [result, view], output)
        view.change(cb.view_image, [result, view], output)
    return demo


def launch(config: AppConfig | None = None) -> None:
    build_demo(config).launch(max_file_size=os.environ.get("FIBERPOSE_MAX_UPLOAD", "50mb"))
