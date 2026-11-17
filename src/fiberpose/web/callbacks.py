"""What the web page computes: plain functions on a ``Session``, no Gradio."""

from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
from matplotlib.figure import Figure

from ..api import AnalysisResult, FiberposeError, analyze
from ..domain import AppConfig
from ..io_export import BatchResultExporter
from ..render import render

# (shown label, render() view name)
VIEWS = [("Diameters", "Diameters"), ("Orientations", "Orientations"), ("Overlay", "Overlay"),
         ("Outline", "Segmentation outline"), ("DT", "DT")]

LABELS = {  # summary key -> name shown on the page
    "diam_mean": "Mean diameter", "diam_median": "Median diameter", "diam_mode": "Mode diameter",
    "diam_sd": "Diameter SD", "mean_angle_deg": "Mean angle", "std_angle_deg": "Angle SD",
    "orientation_index": "Orientation index", "sample_count": "Samples",
    "fragments_with_centerlines": "Fibers measured",
}
ACCENT = "#f97316"
MAX_SIDE = int(os.environ.get("FIBERPOSE_MAX_SIDE", 1920))  # largest accepted width and height, px


class ImageTooLarge(FiberposeError):
    """The image exceeds ``MAX_SIDE``; it is not analysed."""


@dataclass
class Session:
    """One analysed image and the folder its downloadable files are written to."""

    result: AnalysisResult
    source: str  # path of the input image
    folder: tempfile.TemporaryDirectory  # deleted together with the session

    def path(self, suffix: str) -> str:
        return os.path.join(self.folder.name, f"{Path(self.source).stem}_{suffix}")


def analyze_file(
    path: str, config: AppConfig, pixel_size: float, unit: str, invert: bool, previous: Session | None = None
) -> Session:
    """Run the pipeline on an image file; a `previous` session of the same file skips its unchanged stages."""
    same = previous.result if previous and previous.source == path else None
    image = same.frame.orig_img if same else iio.imread(path)  # read from disk to keep the bit depth
    height, width = image.shape[:2]
    if max(height, width) > MAX_SIDE:
        raise ImageTooLarge(
            f"Image is {width} x {height} px; the web version accepts up to {MAX_SIDE} x {MAX_SIDE} px."
        )
    result = analyze(image, config, pixel_size=pixel_size or 1.0, unit=unit, invert=invert, previous=same)
    return Session(result, path, tempfile.TemporaryDirectory(prefix="fiberpose_"))


def view_png(session: Session | None, view: str) -> str | None:
    """PNG of `view`; measurement views fall back to the overlay when nothing was measured."""
    if session is None:
        return None
    path = session.path(f"{view.replace(' ', '_').lower()}.png")
    if not os.path.exists(path):
        image = render(session.result.frame, view)
        iio.imwrite(path, render(session.result.frame, "Overlay") if image is None else image)
    return path


def summary_table(session: Session) -> str:
    """Markdown table of the summary: readable names, two decimals, units."""
    result, rows = session.result, []
    for key, label in LABELS.items():
        value = result.summary.get(key)
        if value is not None:
            unit = result.unit if key.startswith("diam") else "°" if "angle" in key else ""
            rows.append(f"| {label} | {value:.2f} {unit} |" if isinstance(value, float) else f"| {label} | {value} |")
    return "\n".join(["| Summary | |", "|---|---|", *rows]) if rows else ""


def charts(session: Session) -> tuple[Figure, str]:
    """Diameter and orientation histograms, as a figure for the page and an SVG file."""
    result = session.result
    plots = (
        (result.diameters, f"Diameter ({result.unit})", "median", result.summary.get("diam_median")),
        (result.angles_deg, "Orientation (deg)", "mean", result.summary.get("mean_angle_deg")),
    )
    with matplotlib.rc_context({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}):
        fig = Figure(figsize=(8, 7), layout="constrained")
        for ax, (values, label, statistic, marker) in zip(fig.subplots(2, 1), plots):
            ax.hist(values[np.isfinite(values)], bins=40, color=ACCENT)
            ax.set(xlabel=label, ylabel="Samples")
            if marker is not None:
                ax.axvline(marker, color="0.25", linestyle="--", label=f"{statistic} {marker:.2f}")
                ax.legend(frameon=False)
        path = session.path("charts.svg")
        fig.savefig(path)
    return fig, path


def results_zip(session: Session) -> str:
    """Everything in one archive: every view, the charts, the label map and the measurements table."""
    files = [view_png(session, view) for _label, view in VIEWS] + [charts(session)[1], session.path("labels.tif")]
    iio.imwrite(files[-1], session.result.labels)
    if session.result.summary:  # the same CSV the desktop app exports
        files.append(session.path("measurements.csv"))
        BatchResultExporter()._write_samples_csv(session.result.frame, files[-1])
    path = session.path("results.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, os.path.basename(file))
    return path
