"""Stable public entry point for analysing a single image.

External code (the web demo, notebooks, plugins) should use ``analyze`` and
``AnalysisResult`` instead of driving ``PipelineController`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import numpy as np

from .domain import AppConfig, Frame, SampleProperties
from .ml import onnx_cuda_available
from .models import resolve_model_path
from .pipeline import PipelineController
from .utils import ensure_rgb_u8

__all__ = ["AnalysisResult", "FiberposeError", "analyze"]

class FiberposeError(RuntimeError):
    """Raised when the pipeline fails on an image."""


@dataclass(frozen=True)
class AnalysisResult:
    """Outputs of ``analyze`` for one image.

    All arrays are in the coordinate frame of ``image`` (after the optional
    ``config.scale_label`` rescaling).
    """

    image: np.ndarray
    distance_transform: np.ndarray
    labels: Optional[np.ndarray]
    diameters: np.ndarray
    angles_deg: np.ndarray
    # (y, x, ny, nx, radius_px, angle_deg) per sample, for drawing diameter lines.
    samples: list = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    unit: str = "px"
    # Set when segmentation succeeded but no measurements could be taken.
    message: Optional[str] = None
    frame: Optional[Frame] = field(default=None, repr=False)


@lru_cache(maxsize=2)
def _controller(model_path: str, device: str) -> PipelineController:
    return PipelineController(model_path, device)


def analyze(
    image: np.ndarray,
    config: AppConfig | None = None,
    *,
    pixel_size: float = 1.0,
    unit: str = "px",
    invert: bool = False,
) -> AnalysisResult:
    """Run prediction, segmentation and measurements on a 2D image.

    ``config.model_path`` selects the model by file name or absolute path; when
    empty, the default pretrained model is used (downloaded on first use).
    """
    config = config or AppConfig(use_gpu=False)
    model_path = resolve_model_path(config.model_path)
    device = "cuda" if config.use_gpu and onnx_cuda_available() else "cpu"
    sample_props = SampleProperties(
        pixel_size_x=float(pixel_size),
        pixel_size_y=float(pixel_size),
        unit=str(unit).strip() or "px",
        source="api",
    )
    frame = Frame.empty(frame_id="image", orig_img=np.asarray(image))

    outcome = _controller(model_path, device).ensure_measurements(
        frame, config, sample_props, invert_input=bool(invert)
    )

    if outcome.error_message or not frame.has_segmentation:
        raise FiberposeError(outcome.error_message or outcome.info_message or "Processing failed.")

    base = frame.orig_img_scaled if frame.scale_factor != 1.0 and frame.orig_img_scaled is not None else frame.orig_img
    measured = outcome.ok
    return AnalysisResult(
        image=ensure_rgb_u8(base),
        distance_transform=np.asarray(frame.dt, dtype=np.float32),
        labels=frame.labels_2d,
        diameters=_as_array(frame.thickness_units if measured else None),
        angles_deg=_as_array([s[5] for s in frame.measurement_samples or []] if measured else None),
        samples=list(frame.drawing_samples or []) if measured else [],
        summary=dict(frame.fiber_summary or {}) if measured else {},
        unit=sample_props.unit,
        message=None if measured else outcome.info_message,
        frame=frame,
    )


def _as_array(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float32)
    return np.asarray(values, dtype=np.float32)
