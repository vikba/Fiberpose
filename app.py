import os
import traceback

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sources.domain import AppConfig, Frame, SampleProperties
from sources.fiber_analyzer_2D import FiberAnalyzer_2D
from sources.pipeline import PipelineController
from sources.utils import to_uint8_for_view, labels_to_color, ensure_rgb_u8


MODEL_PATH = os.getenv("FIBERPOSE_MODEL", "checkpoints/model_fiber_2_no_bg.onnx")

analyzer = FiberAnalyzer_2D(
    model_path=MODEL_PATH,
    device="cpu",
    strict=False,
    keep_segments_without_measurement_centerlines=False,
)

pipeline = PipelineController(analyzer)


def as_pil_rgb(img):
    if img is None:
        return None

    arr = np.asarray(img)

    if arr.ndim == 2:
        arr = to_uint8_for_view(arr)
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.dtype != np.uint8:
        arr = to_uint8_for_view(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

    # OpenCV-style BGR overlays may appear swapped.
    # If your overlay colors look wrong, uncomment this:
    # arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(arr[..., :3])


def make_hist(values, title, xlabel, bins=30, value_range=None):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=130)

    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size:
        ax.hist(values, bins=bins, range=value_range)
    else:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def format_stats(summary):
    if not summary:
        return "No measurements."

    units = summary.get("units", "px")

    return (
        f"Mean diameter: {summary.get('mean_diameter', np.nan):.3f} {units}\n"
        f"Std diameter: {summary.get('std_diameter', np.nan):.3f} {units}\n"
        f"Orientation index: {summary.get('orientation_index', np.nan):.3f}\n"
        f"Mean angle: {summary.get('mean_angle_deg', np.nan):.3f} deg\n"
        f"Std angle: {summary.get('std_angle_deg', np.nan):.3f} deg\n"
        f"Porosity: {100.0 * summary.get('porosity', np.nan):.2f} %\n"
        f"Samples: {summary.get('sample_count', 0)}\n"
        f"Fragments with centerlines: {summary.get('fragments_with_centerlines', 0)}"
    )

def make_segmentation_measurement_overlay(frame, alpha=0.55):
    base = frame.orig_img if frame.scale_factor == 1.0 or frame.orig_img_scaled is None else frame.orig_img_scaled
    base = ensure_rgb_u8(base)

    labels = frame.labels_2d
    if base is None or labels is None:
        return as_pil_rgb(frame.diameters_overlay)

    labels_rgb = labels_to_color(labels)
    out = base.astype(np.float32).copy()

    mask = labels > 0
    out[mask] = out[mask] * (1.0 - alpha) + labels_rgb[mask].astype(np.float32) * alpha

    out = out.clip(0, 255).astype(np.uint8)

    # draw diameter lines on top
    pts = list(frame.drawing_samples or [])
    if len(pts) > 15000:
        step = int(np.ceil(len(pts) / 15000))
        pts = pts[::step]

    for idx, (y, x, ny, nx, r, _angle_deg) in enumerate(pts):
        if idx % 10 != 0:
            continue
        if r <= 0:
            continue

        p0 = (int(round(x - nx * r)), int(round(y - ny * r)))
        p1 = (int(round(x + nx * r)), int(round(y + ny * r)))

        cv2.line(out, p0, p1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    return Image.fromarray(out)


def load_uploaded_image(image_file: str | None) -> Image.Image:
    if not image_file:
        raise ValueError("Upload an image first.")

    with Image.open(image_file) as src:
        # Use first page/frame for multi-page formats (e.g. TIFF).
        src.seek(0)
        return src.convert("RGB")


def preview_uploaded_image(image_file: str | None):
    try:
        return load_uploaded_image(image_file)
    except Exception:
        return None


def run_fiberpose(
    image_file,
    invert_input,
    pixel_size,
    unit,
    seg_threshold,
    sample_spacing,
    ridge_sigma,
    ridge_low,
    ridge_high,
    max_angle_deg,
):
    try:
        pil_img = load_uploaded_image(image_file)
        img = np.array(pil_img)

        config = AppConfig(
            model_path=MODEL_PATH,
            use_gpu=False,
            scale_label="1x",

            # Centerline extraction
            ridge_sigma=float(ridge_sigma),
            ridge_low=float(ridge_low),
            ridge_high=float(ridge_high),
            max_angle_deg=float(max_angle_deg),

            # Segmentation: radial expansion, not watershed
            segmentation_mode="radial_expansion",
            merge_expansion_factor=0.7,
            segmentation_expansion_factor=1,

            # Still used for pore/void mask calculation
            seg_threshold=int(seg_threshold),

            # Measurements
            sample_spacing=float(sample_spacing),

            strict_centerlines=False,
            keep_segments_without_measurement_centerlines=False,
        )

        sample_props = SampleProperties(
            pixel_size_x=float(pixel_size),
            pixel_size_y=float(pixel_size),
            voxel_size_z=1.0,
            unit=str(unit).strip() or "px",
            source="web",
        )

        frame = Frame.empty(
            frame_id="uploaded_image",
            z_index=0,
            orig_img=img,
        )

        seg_outcome = pipeline.ensure_segmentation(
            frame,
            config,
            invert_input=bool(invert_input),
        )
        if not seg_outcome.ok:
            message = seg_outcome.error_message or seg_outcome.info_message or "Segmentation failed."
            raise RuntimeError(message)

        outcome = pipeline.ensure_measurements(
            frame,
            config,
            sample_props,
            invert_input=bool(invert_input),
        )

        if not outcome.ok:
            message = outcome.error_message or outcome.info_message or "Processing failed."
            raise RuntimeError(message)

        dt_img = as_pil_rgb(frame.dt_u8 if frame.dt_u8 is not None else frame.dt)
        measurement_overlay = make_segmentation_measurement_overlay(frame, alpha=0.65)

        diameters = (
            np.asarray(frame.thickness_units, dtype=np.float32)
            if frame.thickness_units is not None
            else np.array([], dtype=np.float32)
        )

        angles = (
            np.asarray([s[5] for s in frame.measurement_samples], dtype=np.float32)
            if frame.measurement_samples is not None
            else np.array([], dtype=np.float32)
        )

        diameter_hist = make_hist(
            diameters,
            title="Diameter distribution",
            xlabel=f"Diameter ({sample_props.unit})",
            bins=30,
        )

        angle_hist = make_hist(
            angles,
            title="Orientation distribution",
            xlabel="Angle (deg)",
            bins=36,
            value_range=(-90, 90),
        )

        stats_text = format_stats(frame.fiber_summary)

        return dt_img, measurement_overlay, diameter_hist, angle_hist, stats_text

    except Exception as exc:
        err = f"Error: {exc}\n\n{traceback.format_exc()}"
        empty_diam = make_hist([], "Diameter distribution", "Diameter")
        empty_angle = make_hist([], "Orientation distribution", "Angle")
        return None, None, empty_diam, empty_angle, err


with gr.Blocks(title="Fiberpose") as demo:
    gr.Markdown("# Fiberpose web demo")
    gr.Markdown(
        "Neural distance-transform segmentation and morphometry of fibrous structures."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.File(
                label="Input image file",
                file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
                type="filepath",
            )
            input_preview = gr.Image(
                type="pil",
                label="Input preview",
                interactive=False,
            )

            invert_input = gr.Checkbox(
                value=False,
                label="Invert input",
            )

            pixel_size = gr.Number(
                value=1.0,
                label="Pixel size",
            )

            unit = gr.Textbox(
                value="px",
                label="Unit",
            )

            run_button = gr.Button("Run analysis", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                dt_output = gr.Image(type="pil", label="Distance transform")
                overlay_output = gr.Image(type="pil", label="Measurements overlay")

            with gr.Row():
                diameter_hist = gr.Plot(label="Diameter histogram")
                angle_hist = gr.Plot(label="Angle histogram")

            stats_output = gr.Textbox(
                label="Statistics",
                lines=10,
            )

    with gr.Accordion("Advanced parameters", open=False):
        seg_threshold = gr.Slider(
            minimum=0,
            maximum=255,
            value=1,
            step=1,
            label="Segmentation threshold",
        )

        sample_spacing = gr.Number(
            value=5.0,
            label="Measurement sample spacing, px",
        )

        ridge_sigma = gr.Number(
            value=1.0,
            label="Ridge sigma",
        )

        ridge_low = gr.Number(
            value=0.01,
            label="Ridge low",
        )

        ridge_high = gr.Number(
            value=0.4,
            label="Ridge high",
        )

        max_angle_deg = gr.Number(
            value=30.0,
            label="Max centerline angle, deg",
        )


    input_image.change(
        fn=preview_uploaded_image,
        inputs=[input_image],
        outputs=[input_preview],
    )

    run_button.click(
        fn=run_fiberpose,
        inputs=[
            input_image,
            invert_input,
            pixel_size,
            unit,
            seg_threshold,
            sample_spacing,
            ridge_sigma,
            ridge_low,
            ridge_high,
            max_angle_deg,
        ],
        outputs=[
            dt_output,
            overlay_output,
            diameter_hist,
            angle_hist,
            stats_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
