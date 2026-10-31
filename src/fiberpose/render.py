"""Images of a `Frame` for display and export, built on demand from the frame's data."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import value_indices
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries

from .domain import AppConfig, Frame
from .utils import ensure_rgb_u8, labels_to_color, to_uint8_for_view

# Fully saturated color for every OpenCV hue (0..179). Converted one pixel at a time:
# cvtColor's vectorized path for longer rows rounds some channels differently.
_HUE_TO_BGR = np.array(
    [cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0] for h in range(180)],
    dtype=np.uint8,
)


def render(frame: Frame, view: str, grad_quality_points: Sequence[float] | None = None) -> np.ndarray | None:
    """Image for `view`: a label of the GUI's View box, or "Diameters", "Orientations", "Pores".

    Returns None when the frame lacks the data the view needs. Images are cached in
    `frame.views` until a pipeline stage is recomputed; treat them as read-only.
    """
    if view in frame.views:
        return frame.views[view]
    labels = frame.labels_2d
    base = frame.orig_img if frame.scale_factor == 1.0 or frame.orig_img_scaled is None else frame.orig_img_scaled
    match view:
        case "Original":
            image = ensure_rgb_u8(frame.orig_img)
        case "DT":
            image = to_uint8_for_view(frame.dt_normalized)
        case "Foreground":
            image = to_uint8_for_view(frame.foreground_prob)
        case "Laplacian":
            image = to_uint8_for_view(frame.lapl_img)
        case "gx" | "gy":
            image = to_uint8_for_view(getattr(frame, view))
        case "Gradient magnitude":
            image = None if frame.gx is None else to_uint8_for_view(np.hypot(frame.gx, frame.gy))
        case "DT grad error":
            # Signed error |grad(dt)| - 1: blue <= p0, green over [p1, p2], red >= p3, linear in between.
            # Depends on the points, so it is not cached.
            if frame.dt is None:
                return None
            points = grad_quality_points or AppConfig().grad_quality_points
            if not np.all(np.diff(points) > 0):
                points = AppConfig().grad_quality_points
            err = np.sqrt(frame.gx * frame.gx + frame.gy * frame.gy) - 1.0
            image = np.stack(
                [np.interp(err, points, ramp) for ramp in ((0, 0, 0, 255), (0, 255, 255, 0), (255, 0, 0, 0))],
                axis=-1,
            ).astype(np.uint8)
            image[frame.dt <= 0] = 0
            return image
        case "Segmentation":
            image = labels_to_color(labels)
        case "Overlay":
            colors, base_rgb = render(frame, "Segmentation"), ensure_rgb_u8(base)
            if colors is None or base_rgb is None or base_rgb.shape[:2] != labels.shape:
                image = None
            else:
                image = base_rgb.copy()
                fg = labels > 0
                image[fg] = (base_rgb[fg] / 2 + colors[fg] / 2).astype(np.uint8)
        case "Segmentation outline":
            image = ensure_rgb_u8(base)
            if labels is None or image is None or image.shape[:2] != labels.shape:
                image = None
            else:
                image = image.copy()
                image[find_boundaries(labels, mode="outer")] = (255, 0, 0)
        case "Centerlines":
            if frame.fragments is None:
                image = None
            else:
                image = cv2.cvtColor(to_uint8_for_view(frame.dt_u8), cv2.COLOR_GRAY2BGR)
                for attr, color in (("segment_centerlines_xy", (255, 0, 0)), ("measurement_centerlines_xy", (0, 255, 0))):
                    lines = [p.reshape(-1, 1, 2).astype(np.int32) for f in frame.fragments for p in getattr(f, attr)]
                    if lines:
                        cv2.polylines(image, lines, isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
                image[find_boundaries(labels, mode="outer")] = (0, 0, 255)
        case "Sampling" | "Diameters":
            # Diameter lines at the sample points: all of them over the DT, every 10th over the overlay.
            if view == "Diameters" and not frame.has_measurements:
                image = None
            else:
                image = ensure_rgb_u8(frame.dt_u8) if view == "Sampling" else render(frame, "Overlay")
            samples = frame.drawing_samples or []
            if image is not None and samples:
                image = image.copy()
                h, w = image.shape[:2]
                for idx, (y, x, ny, nx, r, _angle) in enumerate(samples[:: -(-len(samples) // 15000)]):
                    if not (0 <= y < h and 0 <= x < w) or r <= 0 or (view == "Diameters" and idx % 10):
                        continue
                    x0, x1 = (min(max(int(round(v)), 0), w - 1) for v in (x - nx * r, x + nx * r))
                    y0, y1 = (min(max(int(round(v)), 0), h - 1) for v in (y - ny * r, y + ny * r))
                    cv2.line(image, (x0, y0), (x1, y1), (255, 255, 0), 1, cv2.LINE_AA)
                    if view == "Sampling":
                        cv2.circle(image, (int(round(x)), int(round(y))), 1, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        case "Orientations":
            base_rgb = ensure_rgb_u8(base)
            image = _orientations(frame, base_rgb) if frame.has_measurements and base_rgb is not None else None
        case "Pores":
            if not frame.has_measurements or frame.pore_dt is None:
                image = None
            else:
                void = frame.void_mask > 0
                norm = np.zeros_like(frame.pore_dt, dtype=np.float32)
                if np.any(void):
                    vmax = float(np.percentile(frame.pore_dt[void], 99.0)) or float(np.max(frame.pore_dt))
                    if vmax > 0:
                        norm[void] = (frame.pore_dt[void] / vmax).clip(0.0, 1.0)
                image = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
                image[~void] = (0, 0, 0)
                for pore in frame.pores or []:
                    r = int(round(pore["r_px"]))
                    if r > 0:
                        cv2.circle(image, (int(pore["x"]), int(pore["y"])), r, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        case _:
            image = None
    frame.views[view] = image
    return image


def _orientations(frame: Frame, base_rgb: np.ndarray) -> np.ndarray:
    """Color each labelled pixel by the axial mean angle of its label's 4 nearest samples, weighted by 1/distance."""
    image = base_rgb.copy()
    labels = frame.labels_2d
    height, width = labels.shape[:2]
    sample_map: dict[int, list[tuple[float, float, float]]] = {}
    for y, x, _ny, _nx, _r, angle_deg in frame.drawing_samples or []:
        iy = int(round(float(y)))
        ix = int(round(float(x)))
        if iy < 0 or iy >= height or ix < 0 or ix >= width:
            continue
        label_id = int(labels[iy, ix])
        if label_id <= 0 or not np.isfinite(angle_deg):
            continue
        sample_map.setdefault(label_id, []).append((float(y), float(x), float(angle_deg)))

    label_pixels = value_indices(labels)
    for label_id, label_samples in sample_map.items():
        ys, xs = label_pixels[label_id]
        sample_arr = np.asarray(label_samples, dtype=np.float32)
        k = min(4, sample_arr.shape[0])
        distances, indices = cKDTree(sample_arr[:, :2]).query(np.stack([ys, xs], axis=1).astype(np.float32), k=k)
        distances = np.asarray(distances, dtype=np.float32).reshape(len(ys), k)
        indices = np.asarray(indices, dtype=np.int32).reshape(len(ys), k)
        weights = 1.0 / (distances + 1e-6)
        theta2 = np.deg2rad(2.0 * sample_arr[:, 2][indices])
        c = np.sum(weights * np.cos(theta2), axis=1)
        s = np.sum(weights * np.sin(theta2), axis=1)
        image[ys, xs] = _angles_to_bgr(0.5 * np.rad2deg(np.arctan2(s, c)))
    return image


def _angles_to_bgr(angles_deg: np.ndarray) -> np.ndarray:
    """Map axial angles (deg) to (N, 3) colors; the hue is offset by 30 deg. Non-finite -> gray."""
    angles = np.asarray(angles_deg, dtype=np.float64)
    ang = np.mod(-angles + 30.0, 180.0) - 90.0
    hue = (ang + 90.0) / 180.0 * 179.0
    finite = np.isfinite(hue)
    colors = _HUE_TO_BGR[np.where(finite, hue, 0.0).astype(np.uint8)]
    colors[~finite] = (200, 200, 200)
    return colors
