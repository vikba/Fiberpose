"""View state model used by GUI presentation logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ViewMode(str, Enum):
    """Canonical view mode labels matching the UI combobox text."""

    ORIGINAL = "Original"
    DT = "DT"
    LAPLACIAN = "Laplacian"
    SEGMENTATION_CENTERLINES = "Segmentation centerlines"
    CENTERLINES = "Centerlines"
    MEASUREMENT_CENTERLINES = "Measurement centerlines"
    CENTERLINE_COMPARISON = "Centerline comparison"
    SEGMENTATION = "Segmentation"
    SEGMENTATION_OUTLINE = "Segmentation outline"
    OVERLAY = "Overlay"
    SAMPLING = "Sampling"
    MEASUREMENTS_OVERLAY = "Measurements overlay"
    GX = "gx"
    GY = "gy"
    DT_GRAD_ERROR = "DT grad error"


@dataclass
class ViewState:
    """UI-level view selection and overlay toggles."""

    mode: str = ViewMode.ORIGINAL.value
    show_pores: bool = False
    show_orientation: bool = False
    zoom: float = 1.0
