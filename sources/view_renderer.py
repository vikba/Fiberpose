"""Rendering helpers for mapping frame state to display images."""

from __future__ import annotations

from typing import Callable

import numpy as np
from skimage.segmentation import find_boundaries

from .datastructures import Frame
from .utils import ensure_rgb_u8, labels_to_color, to_uint8_for_view
from .view_state import ViewState, ViewMode


class ViewRenderer:
    """Select and build the image to show for a given view state."""

    def __init__(self, grad_error_builder: Callable[[Frame], np.ndarray | None]):
        self._grad_error_builder = grad_error_builder

    @staticmethod
    def _build_segmentation_outline_image(
        base_img: np.ndarray | None,
        labels: np.ndarray | None,
    ) -> np.ndarray | None:
        if labels is None:
            return None
        base_rgb = ensure_rgb_u8(base_img)
        if base_rgb is None or base_rgb.shape[:2] != labels.shape[:2]:
            return None

        out = base_rgb.copy()
        try:
            boundaries = find_boundaries(labels, mode="outer")
        except Exception:
            boundaries = None
        if boundaries is not None and boundaries.any():
            out[boundaries] = (255, 0, 0)
        return out

    @staticmethod
    def _centerline_view(img: np.ndarray | None, labels: np.ndarray | None) -> np.ndarray | None:
        out = ensure_rgb_u8(img)
        if out is None:
            return None
        if labels is None or out.shape[:2] != labels.shape[:2]:
            return out
        painted = out.copy()
        try:
            boundaries = find_boundaries(labels, mode="outer")
        except Exception:
            boundaries = None
        if boundaries is not None and boundaries.any():
            painted[boundaries] = (0, 0, 255)
        return painted

    @staticmethod
    def _labels_rgb(frame: Frame) -> np.ndarray | None:
        labels = frame.labels_2d
        if labels is None:
            return None
        cache_id, cache_img = frame.labels_rgb_cache
        current_id = id(labels)
        if cache_id != current_id:
            cache_img = labels_to_color(labels)
            frame.labels_rgb_cache = (current_id, cache_img)
        return cache_img

    def _segmentation_outline_view(self, frame: Frame) -> np.ndarray | None:
        labels = frame.labels_2d
        if labels is None:
            return None

        base_img = frame.orig_img if frame.scale_factor == 1.0 or frame.orig_img_scaled is None else frame.orig_img_scaled
        cache_id, cache_img = frame.segmentation_outline_cache
        current_id = id(labels)
        if cache_id != current_id or cache_img is None:
            cache_img = self._build_segmentation_outline_image(base_img, labels)
            frame.segmentation_outline_cache = (current_id, cache_img)
        return cache_img

    def render(self, frame: Frame, state: ViewState) -> np.ndarray | None:
        mode = state.mode
        labels_rgb = self._labels_rgb(frame)

        handlers: dict[str, Callable[[], np.ndarray | None]] = {
            ViewMode.ORIGINAL.value: lambda: ensure_rgb_u8(frame.orig_img),
            ViewMode.DT.value: lambda: to_uint8_for_view(frame.dt_normalized),
            ViewMode.LAPLACIAN.value: lambda: to_uint8_for_view(frame.lapl_img),
            ViewMode.SEGMENTATION_CENTERLINES.value: lambda: self._centerline_view(frame.segmentation_centerlines_overlay_img, frame.labels_2d),
            ViewMode.CENTERLINES.value: lambda: self._centerline_view(frame.segmentation_centerlines_overlay_img, frame.labels_2d),
            ViewMode.MEASUREMENT_CENTERLINES.value: lambda: self._centerline_view(frame.measurement_centerlines_overlay_img, frame.labels_2d),
            ViewMode.CENTERLINE_COMPARISON.value: lambda: ensure_rgb_u8(frame.centerlines_comparison_overlay_img),
            ViewMode.SEGMENTATION.value: lambda: labels_rgb,
            ViewMode.SEGMENTATION_OUTLINE.value: lambda: self._segmentation_outline_view(frame),
            ViewMode.OVERLAY.value: lambda: frame.overlay_img,
            ViewMode.SAMPLING.value: lambda: frame.sampling_overlay if frame.sampling_overlay is not None else ensure_rgb_u8(frame.dt_u8),
            ViewMode.GX.value: lambda: to_uint8_for_view(frame.gx),
            ViewMode.GY.value: lambda: to_uint8_for_view(frame.gy),
            ViewMode.DT_GRAD_ERROR.value: lambda: self._grad_error_builder(frame),
        }

        if mode == ViewMode.MEASUREMENTS_OVERLAY.value:
            fallback = frame.overlay_img if frame.overlay_img is not None else labels_rgb
            if state.show_pores:
                return frame.pores_overlay if frame.pores_overlay is not None else fallback
            if state.show_orientation:
                return frame.orientations_overlay if frame.orientations_overlay is not None else fallback
            return frame.diameters_overlay if frame.diameters_overlay is not None else fallback

        handler = handlers.get(mode)
        return handler() if handler is not None else None
