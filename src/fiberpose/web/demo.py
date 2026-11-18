"""The Gradio page: its layout, its event handlers and the function that serves it."""

from __future__ import annotations

import os
import time
from dataclasses import replace

import gradio as gr

from ..api import FiberposeError
from ..domain import AppConfig
from ..paths import EXAMPLE_IMAGE_PATHS
from . import callbacks as cb
from . import usage

__all__ = ["build_demo", "launch"]


def _download(path: str | None):
    """Update for a save button: offer `path`, or stay disabled without one."""
    return gr.DownloadButton(value=path, interactive=bool(path))


def build_demo(config: AppConfig | None = None) -> gr.Blocks:
    config = config or AppConfig(use_gpu=False)

    def run(path, pixel_size, unit, invert, edge_threshold, previous):
        if not path:
            raise gr.Error("Upload an image first.")
        settings = replace(config, watershed_grad_threshold=edge_threshold)
        filename = os.path.basename(path)  # Gradio keeps the uploaded file's own name
        example = filename in {p.name for p in EXAMPLE_IMAGE_PATHS}
        start = time.perf_counter()
        try:
            session = cb.analyze_file(path, settings, pixel_size, unit, invert, previous)
        except cb.ImageTooLarge as exc:  # a warning; whatever was analysed before stays on the page
            usage.log(ok=False, seconds=round(time.perf_counter() - start, 2), example=example, filename=filename)
            gr.Warning(str(exc))
            return (gr.skip(),) * 6
        except FiberposeError as exc:
            usage.log(ok=False, seconds=round(time.perf_counter() - start, 2), example=example, filename=filename)
            raise gr.Error(str(exc)) from exc
        height, width = session.result.frame.orig_img.shape[:2]
        usage.log(ok=True, seconds=round(time.perf_counter() - start, 2), width=width, height=height, example=example, filename=filename)
        if session.result.message:
            gr.Info(session.result.message)
        figure, svg = cb.charts(session)
        archive = cb.results_zip(session)  # also writes the CSV it contains
        csv = session.path("measurements.csv") if session.result.summary else None
        return session, cb.summary_table(session), figure, *map(_download, (svg, csv, archive))

    def show(session, view):
        path = cb.view_png(session, view)
        return path, _download(path)  # the same file is displayed and offered by "Save view"

    with gr.Blocks(title="Fiberpose") as demo:
        gr.Markdown(
            "# Fiberpose\nSegmentation and morphometry of fibrous microstructures. Upload a micrograph "
            f"up to {cb.MAX_SIDE} x {cb.MAX_SIDE} px (or click an example) and press **Analyze**. [Source code](https://github.com/vikba/Fiberpose)"
        )
        session = gr.State()  # this browser tab's analysed image (callbacks.Session)
        with gr.Row(equal_height=True):  # settings, each above the image it applies to
            with gr.Row(variant="panel"):  # segmentation settings
                pixel_size = gr.Number(1.0, label="Pixel size", minimum=0, min_width=80)
                unit = gr.Dropdown(["px", "nm", "µm", "mm"], value="px", label="Unit", allow_custom_value=True, min_width=80)
                edge_threshold = gr.Number(
                    config.watershed_grad_threshold, label="Edge threshold", minimum=0, maximum=1, step=0.05,
                    min_width=80, info="min |∇DT|; lower = wider.",
                )
                invert = gr.Checkbox(label="Invert", info="for dark fibers", min_width=80)
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
                gr.Examples([[str(path)] for path in EXAMPLE_IMAGE_PATHS], inputs=image)
                summary = gr.Markdown()
            with gr.Column():  # under the result
                with gr.Row():
                    save_view = gr.DownloadButton("Save view (PNG)", size="sm", interactive=False)
                    save_charts = gr.DownloadButton("Save charts (SVG)", size="sm", interactive=False)
                    save_csv = gr.DownloadButton("Save measurements (CSV)", size="sm", interactive=False)
                    save_all = gr.DownloadButton("Save results (ZIP)", size="sm", interactive=False)
                plots = gr.Plot(label="Distributions")

        inputs = [image, pixel_size, unit, invert, edge_threshold, session]
        for trigger in (button.click, edge_threshold.submit):  # the threshold reuses the model prediction
            analysed = trigger(run, inputs, [session, summary, plots, save_charts, save_csv, save_all])
            analysed.success(show, [session, view], [output, save_view])
        view.change(show, [session, view], [output, save_view])
    return demo


def launch(config: AppConfig | None = None) -> None:
    build_demo(config).launch(
        max_file_size=os.environ.get("FIBERPOSE_MAX_UPLOAD", "50mb"), footer_links=[]
    )
