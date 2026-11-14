"""What the web page computes: plain functions on an ``AnalysisResult``, no Gradio."""

from __future__ import annotations

import tempfile

import imageio.v3 as iio
import numpy as np
from matplotlib.figure import Figure

from ..api import AnalysisResult, analyze
from ..domain import AppConfig
from ..io_export import BatchResultExporter
from ..render import render

# (shown label, render() view name)
VIEWS = [("Diameters", "Diameters"), ("Orientations", "Orientations"), ("Overlay", "Overlay"),
         ("Outline", "Segmentation outline"), ("DT", "DT")]


def analyze_file(path: str, config: AppConfig, pixel_size: float, unit: str, invert: bool) -> AnalysisResult:
    """Read the image from disk (keeping its bit depth) and run the full pipeline."""
    return analyze(iio.imread(path), config, pixel_size=pixel_size or 1.0, unit=unit, invert=invert)


def view_image(result: AnalysisResult | None, view: str) -> np.ndarray | None:
    """Image of `view`; measurement views fall back to the overlay when nothing was measured."""
    if result is None:
        return None
    image = render(result.frame, view)
    return render(result.frame, "Overlay") if image is None else image


def summary_rows(result: AnalysisResult) -> list[list]:
    rows = result.summary.items()
    return [[name, round(v, 3) if isinstance(v, float) else v] for name, v in rows if name != "porosity"]


def histograms(result: AnalysisResult) -> Figure:
    fig = Figure(figsize=(8, 7), layout="constrained")
    plots = ((result.diameters, f"Diameter ({result.unit})"), (result.angles_deg, "Orientation (deg)"))
    for ax, (values, label) in zip(fig.subplots(2, 1), plots):
        ax.hist(values[np.isfinite(values)], bins=40)
        ax.set(xlabel=label, ylabel="Samples")
    return fig


def samples_csv(result: AnalysisResult) -> str | None:
    """The same summary + per-sample table the desktop app exports, or None without measurements."""
    if not result.summary:
        return None
    path = tempfile.NamedTemporaryFile(prefix="fiberpose_", suffix=".csv", delete=False).name
    BatchResultExporter()._write_samples_csv(result.frame, path)
    return path
