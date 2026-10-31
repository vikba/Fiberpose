"""2D fiber analysis: DT prediction, centerlines, segmentation, and measurement sampling.

Every function is stateless: arrays in, results out. Per-image state lives on `Frame`.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, laplace, value_indices
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.segmentation import watershed

from .domain import FiberFragment
from .ml import OnnxFiberposeModel
from .raster import disk_mask, disk_radii, stamp_disks
from .ridge_detector import RidgeDetector
from .utils import (
    _bilinear,
    _sample_vec2,
    normals_from_dt_2d,
    resample_polyline_xy,
)


logger = logging.getLogger(__name__)

Sample = tuple[int, int, float, float, float, float]


# ------------------------------------------------------------------ #
# Public pipeline stages


def predict(model: OnnxFiberposeModel, image: np.ndarray) -> dict[str, np.ndarray | None]:
    """Predict the DT of `image` plus its derived fields, keyed like `Frame.set_prediction_outputs`."""
    h, w = image.shape[:2]
    stride = int(model.stride)
    pad_width = ((0, -h % stride), (0, -w % stride)) + ((0, 0),) * (image.ndim - 2)
    dt_pad, fg_pad = model.predict(np.pad(image, pad_width, mode="reflect"))

    dt = np.asarray(dt_pad[:h, :w], dtype=np.float32)
    dt_smooth = gaussian_filter(dt, sigma=1.5)
    gy, gx = np.gradient(dt_smooth)
    return {
        "dt": dt,
        "dt_u8": np.clip(dt, 0, 255).astype(np.uint8),
        "dt_normalized": dt / 255.0,
        "foreground_prob": None if fg_pad is None else np.asarray(fg_pad[:h, :w], dtype=np.float32),
        "gx": np.asarray(gaussian_filter(gx, sigma=1.0), dtype=np.float32),
        "gy": np.asarray(gaussian_filter(gy, sigma=1.0), dtype=np.float32),
        "lapl_img": np.asarray(laplace(dt_smooth), dtype=np.float32),
    }


def segment(
    dt: np.ndarray,
    dt_u8: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    foreground_prob: np.ndarray | None,
    *,
    segmentation_mode: str,
    watershed_grad_threshold: float,
    min_segment_foreground_mean: float,
    max_angle_deg: float,
    ridge_sigma: float,
    ridge_low: float,
    ridge_high: float,
    strict_centerlines: bool,
    keep_segments_without_measurement_centerlines: bool,
    merge_expansion_factor: float,
    segmentation_expansion_factor: float,
    min_fragment_centerline_length_px: float,
    measurement_centerline_length_trim_fraction: float,
    grad_quality_trim_fraction: float,
) -> tuple[np.ndarray, list[FiberFragment]]:
    """Segment fibers: ridges -> fragments -> uint16 labels.

    Returns the labels and the fragments that still own a label after pruning.
    """
    shape = dt_u8.shape
    detected_polylines = RidgeDetector.detect_ridges(
        image2d=dt_u8,
        sigma=float(ridge_sigma),
        low=float(ridge_low),
        high=float(ridge_high),
        polarity="light",
    )
    raw_fragments = _group_detected_polylines_into_segmentation_fragments(
        detected_polylines=detected_polylines,
        dt=dt,
        merge_expansion_factor=float(merge_expansion_factor),
        min_fragment_centerline_length_px=float(min_fragment_centerline_length_px),
        trim=3,
    )
    fragments = _filter_measurement_centerlines(
        segmentation_fragments=raw_fragments,
        dt_u8=dt_u8,
        dt=dt,
        gx=gx,
        gy=gy,
        max_angle_deg=max_angle_deg,
        win_len=25.0,
        std_thresh=0.8,
        strict=strict_centerlines,
        keep_segments_without_measurement_centerlines=keep_segments_without_measurement_centerlines,
    )

    # Drop the fragments with the shortest measurement centerlines.
    trim_count = int(np.floor(len(fragments) * _clamp_trim_fraction(measurement_centerline_length_trim_fraction)))
    if trim_count > 0:
        ordered = sorted(
            fragments,
            key=lambda seg: (float(seg.total_measurement_centerline_length_px), int(seg.fragment_id)),
        )
        drop_ids = {int(seg.fragment_id) for seg in ordered[:trim_count]}
        fragments = [seg for seg in fragments if int(seg.fragment_id) not in drop_ids]

    grad_err = np.abs(np.sqrt(gx * gx + gy * gy) - 1.0)
    # Pixels each fragment covers at the segmentation expansion, shared by both steps below.
    fragment_pixels = _fragment_pixels(fragments, float(segmentation_expansion_factor), shape)
    fragments, fragment_pixels = _filter_fragments_by_grad_quality(
        fragments=fragments,
        fragment_pixels=fragment_pixels,
        grad_err=grad_err,
        trim_fraction=_clamp_trim_fraction(grad_quality_trim_fraction),
        min_valid_pixels=8,
    )

    if segmentation_mode == "radial_expansion":
        labels_i32 = _rasterize_segmentation_fragments(fragments=fragments, fragment_pixels=fragment_pixels, shape=shape)
        labels = np.clip(labels_i32, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    else:
        markers = _render_markers(shape, fragments)
        labels = _watershed(dt_u8, gx, gy, markers, gradient_threshold=watershed_grad_threshold)
    labels = _prune_labels_by_foreground(
        labels,
        foreground_prob=foreground_prob,
        min_segment_foreground_mean=min_segment_foreground_mean,
    )

    surviving_ids = set(np.unique(labels)) - {0}
    fragments = [fragment for fragment in fragments if int(fragment.parent_fiber_id) in surviving_ids]
    logger.info("Segmented %d fragments (%d detected)", len(fragments), len(raw_fragments))
    return labels, fragments


def sample_centerlines(
    dt: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    fragments: list[FiberFragment],
    spacing_px: float = 5.0,
    scale_factor: float = 1.0,
) -> tuple[list[Sample], list[Sample]]:
    """Sample thickness and orientation along the fragments' measurement centerlines.

    Returns:
      - measurement_samples: (y, x, ny, nx, r, angle_deg) in original/display scale
      - drawing_samples: (y, x, ny, nx, r, angle_deg) in processing-image scale

    Note:
        `_xy` suffix denotes (x, y) ordering, while `_yx` denotes (y, x).
    """
    spacing = max(1.0, float(spacing_px))
    scale = float(scale_factor)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    measurement_samples: list[Sample] = []
    drawing_samples: list[Sample] = []

    def _canon_angle(ty: float, tx: float) -> float:
        n = float(np.hypot(ty, tx))
        if n == 0.0 or not np.isfinite(n):
            return float("nan")
        ty, tx = ty / n, tx / n
        ang = -np.degrees(np.arctan2(ty, tx))
        if ang > 90.0:
            ang -= 180.0
        elif ang <= -90.0:
            ang += 180.0
        if ang <= -90.0:
            ang = -89.999
        return float(ang)

    for frag in fragments:
        for seg_xy in frag.measurement_centerlines_xy:
            samples_xy = resample_polyline_xy(seg_xy.astype(np.float32), step=spacing)
            if len(samples_xy) < 2:
                continue

            samples_yx = samples_xy[:, [1, 0]].astype(np.float32)
            normals_yx = normals_from_dt_2d(
                centerline=samples_yx,
                dt=dt,
                gx=gx,
                gy=gy,
                alpha=0.5,
                iters=3,
                tau_g=1e-3,
            )

            for i, (y, x) in enumerate(samples_yx):
                ny, nx = float(normals_yx[i, 0]), float(normals_yx[i, 1])
                norm = float(np.hypot(ny, nx))
                if norm == 0.0 or not np.isfinite(norm):
                    continue
                ny /= norm
                nx /= norm

                angle_deg = _canon_angle(nx, -ny)

                v1 = float(_bilinear(dt, y - ny, x - nx))
                v2 = float(_bilinear(dt, y + ny, x + nx))
                r = 0.5 * (v1 + v2) + 1.0
                if r <= 0 or not np.isfinite(r):
                    continue

                drawing_samples.append((int(round(y)), int(round(x)), ny, nx, r, angle_deg))
                measurement_samples.append(
                    (int(round(y / scale)), int(round(x / scale)), ny, nx, r / scale, angle_deg)
                )

    return measurement_samples, drawing_samples


def void_mask_and_pore_dt(dt_u8: np.ndarray, thr_u8: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        void_mask_u8: uint8, {0,1}, where 1 marks void (pores), opened to drop thin artifacts
        pore_dt: float32, Euclidean distance (in pixels) inside voids
    """
    t = float(thr_u8) / 255.0
    dt01 = dt_u8.astype(np.float32) / 255.0
    void_mask = (~(dt01 >= t)).astype(np.uint8)
    if void_mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        void_mask = cv2.morphologyEx(void_mask, cv2.MORPH_OPEN, kernel)
    if void_mask.any():
        pore_dt = cv2.distanceTransform(void_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    else:
        pore_dt = np.zeros_like(dt01, dtype=np.float32)
    return void_mask, pore_dt


def sample_inscribed_circles(
    pore_dt: np.ndarray,
    void_mask: np.ndarray,
    scale_factor: float = 1.0,
    min_radius_px: float = 1.0,
    min_peak_px: float = 3.0,
    peak_footprint: int = 3,
) -> list[dict[str, Any]]:
    """Greedily place non-overlapping disks at pore-DT peaks, largest first."""
    scale = scale_factor or 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    H, W = pore_dt.shape[:2]
    current_void = (void_mask.astype(np.uint8) > 0).astype(np.uint8)

    # Local maxima inside the void. Zeros become 255 so background never counts as a peak.
    k = max(3, int(peak_footprint) | 1)
    arr_work = pore_dt.copy()
    arr_work[arr_work == 0] = 255
    dil = cv2.dilate(arr_work, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ys, xs = np.nonzero((arr_work == dil) & (pore_dt > 0) & current_void.astype(bool))
    if xs.size == 0:
        return []

    radii = pore_dt[ys, xs]
    keep = radii >= float(min_peak_px)
    if not np.any(keep):
        return []
    xs, ys, radii = xs[keep], ys[keep], radii[keep]

    order = np.argsort(-radii)
    xs, ys, radii = xs[order], ys[order], radii[order]

    circles: list[dict[str, Any]] = []
    for x, y, r in zip(xs, ys, radii):
        if r < float(min_radius_px):
            continue
        r_int = int(np.floor(float(r)))
        if r_int <= 0:
            continue
        if x - r_int < 0 or x + r_int >= W or y - r_int < 0 or y + r_int >= H:
            continue

        # The disk lies fully inside the image: test and carve only its window.
        window = current_void[y - r_int:y + r_int + 1, x - r_int:x + r_int + 1]
        disk = disk_mask(r_int)
        if window[disk].all():
            circles.append({'x': int(x), 'y': int(y), 'r_px': float(r), 'r_px_orig': float(r) / scale})
            window[disk] = 0
    return circles


# ------------------------------------------------------------------ #
# Fragments


def _group_detected_polylines_into_segmentation_fragments(
    *,
    detected_polylines: list[np.ndarray],
    dt: np.ndarray,
    merge_expansion_factor: float = 0.7,
    min_fragment_centerline_length_px: float = 10.0,
    trim: int = 3,
) -> list[FiberFragment]:
    """
    Group detected polylines into segmentation fragments using overlap under merge expansion.
    """
    h, w = dt.shape[:2]
    trim_n = max(0, int(trim))

    polylines: list[np.ndarray] = []
    polyline_radii: list[np.ndarray] = []
    for pts_xy_raw in detected_polylines:
        pts_xy = np.asarray(pts_xy_raw, dtype=np.float32)
        if pts_xy.ndim != 2 or pts_xy.shape[1] != 2:
            continue
        if trim_n > 0 and pts_xy.shape[0] > 2 * trim_n:
            pts_xy = pts_xy[trim_n:-trim_n]
        if pts_xy.shape[0] < 2:
            continue
        x_idx = np.clip(np.rint(pts_xy[:, 0]).astype(np.int32), 0, w - 1)
        y_idx = np.clip(np.rint(pts_xy[:, 1]).astype(np.int32), 0, h - 1)
        polylines.append(pts_xy)
        polyline_radii.append(np.asarray(dt[y_idx, x_idx], dtype=np.float32))

    if not polylines:
        return []

    group_ids = _overlap_groups(polylines, polyline_radii, max(0.0, float(merge_expansion_factor)), (h, w))
    grouped_polylines: dict[int, list[np.ndarray]] = {}
    grouped_radii: dict[int, list[np.ndarray]] = {}
    for label_id, pts_xy, radii in zip(group_ids.tolist(), polylines, polyline_radii):
        grouped_polylines.setdefault(label_id, []).append(pts_xy)
        grouped_radii.setdefault(label_id, []).append(radii)

    fragments: list[FiberFragment] = []
    for label_id in sorted(grouped_polylines):
        max_centerline_length = max(
            FiberFragment._centerline_length(pts_xy) for pts_xy in grouped_polylines[label_id]
        )
        if max_centerline_length < float(min_fragment_centerline_length_px):
            continue
        fragments.append(
            FiberFragment(
                fragment_id=len(fragments) + 1,
                segment_centerlines_xy=grouped_polylines[label_id],
                segment_radii=grouped_radii[label_id],
                parent_fiber_id=int(label_id),
            )
        )

    return fragments


def _overlap_groups(
    polylines: list[np.ndarray],
    radii: list[np.ndarray],
    expansion: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Group ids (1-based) for polylines whose expanded disks share at least one pixel.

    Groups are the connected components of the "shares a pixel" graph, numbered in
    order of their first polyline.
    """
    n = len(polylines)
    owners = np.repeat(np.arange(n), [len(p) for p in polylines])
    pixel, owner = stamp_disks(
        np.concatenate(polylines), disk_radii(np.concatenate(radii), expansion), owners, shape
    )
    # Sort by pixel; consecutive entries on the same pixel link two polylines.
    order = np.lexsort((owner, pixel))
    pixel, owner = pixel[order], owner[order]
    shared = pixel[1:] == pixel[:-1]
    edges = np.unique(owner[:-1][shared] * n + owner[1:][shared])
    graph = coo_matrix((np.ones(edges.size, dtype=np.int32), (edges // n, edges % n)), shape=(n, n))
    _, component = connected_components(graph, directed=False)

    _, first_polyline = np.unique(component, return_index=True)
    rank = np.empty_like(first_polyline)
    rank[np.argsort(first_polyline)] = np.arange(first_polyline.size)
    return rank[component] + 1


def _fragment_pixels(
    fragments: list[FiberFragment],
    segmentation_expansion_factor: float,
    shape: tuple[int, int],
) -> list[np.ndarray]:
    """Flat pixel indices each fragment covers: disks of its local radius along its centerlines."""
    if not fragments:
        return []
    points, radii, owners = [], [], []
    for index, frag in enumerate(fragments):
        for pts_xy, frag_radii in zip(frag.segment_centerlines_xy, frag.segment_radii):
            points.append(pts_xy)
            radii.append(frag_radii)
            owners.append(np.full(len(pts_xy), index))
    expansion = max(0.0, float(segmentation_expansion_factor))
    pixel, owner = stamp_disks(
        np.concatenate(points), disk_radii(np.concatenate(radii), expansion), np.concatenate(owners), shape
    )
    # stamp_disks sorts by owner, so each fragment's pixels are one contiguous run.
    return np.split(pixel, np.cumsum(np.bincount(owner, minlength=len(fragments)))[:-1])


def _rasterize_segmentation_fragments(
    *,
    fragments: list[FiberFragment],
    fragment_pixels: list[np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    """Render radial-expansion labels; where fragments overlap, the later one wins."""
    expanded_labels = np.zeros(shape, dtype=np.int32)
    flat = expanded_labels.reshape(-1)
    for frag, pixels in zip(fragments, fragment_pixels):
        flat[pixels] = int(frag.parent_fiber_id)
    return expanded_labels


def _filter_fragments_by_grad_quality(
    *,
    fragments: list[FiberFragment],
    fragment_pixels: list[np.ndarray],
    grad_err: np.ndarray,
    trim_fraction: float,
    min_valid_pixels: int = 8,
) -> tuple[list[FiberFragment], list[np.ndarray]]:
    """Score fragments by median |grad(dt)| error off their centerlines; drop unreliable/worst ones.

    Returns the kept fragments and their pixel lists, in input order.
    """
    h, w = grad_err.shape[:2]
    grad_err_flat = grad_err.reshape(-1)
    reliable: list[tuple[FiberFragment, np.ndarray]] = []
    for fragment, pixels in zip(fragments, fragment_pixels):
        centerline_pixels = [
            np.clip(np.rint(pts_xy[:, 1]).astype(np.int64), 0, h - 1) * w
            + np.clip(np.rint(pts_xy[:, 0]).astype(np.int64), 0, w - 1)
            for pts_xy in fragment.segment_centerlines_xy
        ]
        values = grad_err_flat[pixels[~np.isin(pixels, np.concatenate(centerline_pixels))]]
        values = np.asarray(values[np.isfinite(values)], dtype=np.float32)
        score = float(np.median(values)) if values.size else float("nan")
        fragment.grad_quality_score = score
        if values.size >= int(min_valid_pixels) and np.isfinite(score):
            reliable.append((fragment, pixels))

    drop_count = int(np.floor(len(reliable) * float(trim_fraction)))
    if drop_count > 0:
        ordered = sorted(reliable, key=lambda item: (-float(item[0].grad_quality_score), int(item[0].fragment_id)))
        drop_ids = {int(frag.fragment_id) for frag, _ in ordered[:drop_count]}
        reliable = [item for item in reliable if int(item[0].fragment_id) not in drop_ids]
    return [frag for frag, _ in reliable], [pixels for _, pixels in reliable]


def _filter_measurement_centerlines(
    *,
    segmentation_fragments: list[FiberFragment],
    dt_u8: np.ndarray,
    dt: np.ndarray,
    gx: np.ndarray | None,
    gy: np.ndarray | None,
    max_angle_deg: float,
    win_len: float,
    std_thresh: float,
    strict: bool,
    keep_segments_without_measurement_centerlines: bool,
) -> list[FiberFragment]:
    """Collect measurement centerlines per fragment and update fragments in place."""
    active_segmentation_fragments: list[FiberFragment] = []
    for seg in segmentation_fragments:
        seg.measurement_centerlines_xy = []
        for polyline_xy in seg.segment_centerlines_xy:
            seg_xy = polyline_xy.reshape(-1, 1, 2)
            windows_xy = _sliding_window_filter(
                seg_xy,
                image=dt_u8,
                dt=dt,
                grad_x=gx,
                grad_y=gy,
                win_len=win_len,
                std_thresh=std_thresh,
                max_angle=max_angle_deg,
                strict=bool(strict),
            )
            for window_xy in windows_xy:
                pts_xy = window_xy.reshape(-1, 2).astype(np.float32, copy=False)
                if pts_xy.shape[0] < 2:
                    continue
                seg.measurement_centerlines_xy.append(pts_xy)
        seg.recompute_total_measurement_centerline_length_px()
        if seg.has_measurement_centerline:
            active_segmentation_fragments.append(seg)
    if bool(keep_segments_without_measurement_centerlines):
        active_segmentation_fragments = list(segmentation_fragments)
    return active_segmentation_fragments


def _clamp_trim_fraction(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.05
    return float(min(0.95, max(0.0, value)))


# ------------------------------------------------------------------ #
# Labels


def _watershed(
    dt_u8: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    markers: np.ndarray,
    gradient_threshold: float,
) -> np.ndarray:
    """
    Watershed on inverted DT, constrained by thresholded gradient magnitude.

    Two dilations of the detected centerline markers are added to the
    threshold mask. The combined mask is then eroded and dilated before
    watershed.
    """

    dt_u8 = np.asarray(dt_u8)
    gx = np.asarray(gx, dtype=np.float32)
    gy = np.asarray(gy, dtype=np.float32)
    markers = np.asarray(markers)
    if gx.shape != dt_u8.shape or gy.shape != dt_u8.shape or markers.shape != dt_u8.shape:
        raise ValueError("DT, gradients, and markers must have matching shapes.")
    if not markers.any():
        return np.zeros_like(dt_u8, dtype=np.uint16)

    threshold = float(gradient_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Gradient magnitude threshold must be finite and non-negative.")

    dt_norm = dt_u8.astype(np.float32) / 255.0
    grad_magnitude = np.hypot(gx, gy)
    fiber_region_mask = np.isfinite(grad_magnitude) & (grad_magnitude >= threshold)

    morphology_kernel = np.ones((3, 3), dtype=np.uint8)
    centerline_mask = (markers > 0).astype(np.uint8)
    dilated_centerlines = cv2.dilate(
        centerline_mask,
        morphology_kernel,
        iterations=2,
    ).astype(bool)
    fiber_region_mask |= dilated_centerlines
    fiber_region_mask_u8 = fiber_region_mask.astype(np.uint8)
    fiber_region_mask_u8 = cv2.erode(
        fiber_region_mask_u8,
        morphology_kernel,
        iterations=1,
    )
    fiber_region_mask = cv2.dilate(
        fiber_region_mask_u8,
        morphology_kernel,
        iterations=1,
    ).astype(bool)

    elev = 1.0 - dt_norm
    if not fiber_region_mask.any():
        return np.zeros_like(dt_u8, dtype=np.uint16)
    marker_pixels = markers > 0
    logger.info(
        "Watershed gradient mask: threshold=%.4f mask=%d/%d marker_pixels=%d/%d",
        threshold,
        int(np.count_nonzero(fiber_region_mask)),
        int(fiber_region_mask.size),
        int(np.count_nonzero(fiber_region_mask & marker_pixels)),
        int(np.count_nonzero(marker_pixels)),
    )
    conn8 = np.ones((3, 3), dtype=bool)
    labels = watershed(
        elev,
        markers=markers,
        mask=fiber_region_mask,
        connectivity=conn8,
    ).astype(np.uint16)
    return labels


def _prune_labels_by_foreground(
    labels: np.ndarray,
    *,
    foreground_prob: np.ndarray | None,
    min_segment_foreground_mean: float = 0.5,
) -> np.ndarray:
    if foreground_prob is None:
        return labels
    labels_arr = np.asarray(labels)
    foreground_arr = np.asarray(foreground_prob, dtype=np.float32)
    if labels_arr.shape != foreground_arr.shape:
        raise ValueError("Foreground probability must match label shape.")

    threshold = float(min_segment_foreground_mean)
    if not np.isfinite(threshold):
        threshold = 0.5
    threshold = float(np.clip(threshold, 0.0, 1.0))

    pruned = labels_arr.copy()
    # value_indices lists each label's pixels in raster order, so the float32 mean
    # sums the same values in the same order as a boolean mask would.
    for label_id, pixels in value_indices(labels_arr).items():
        if label_id > 0 and float(np.mean(foreground_arr[pixels])) < threshold:
            pruned[pixels] = 0
    return pruned.astype(labels_arr.dtype, copy=False)


def _render_markers(shape: tuple[int, int], segments: list[FiberFragment]) -> np.ndarray:
    markers = np.zeros(shape, dtype=np.int32)
    for seg in segments:
        for polyline_xy in seg.segment_centerlines_xy:
            pts = np.asarray(polyline_xy, dtype=np.float32)
            pts_int = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                markers,
                [pts_int],
                isClosed=False,
                color=int(seg.parent_fiber_id),
                thickness=1,
                lineType=cv2.LINE_8,
            )
    return markers


# ------------------------------------------------------------------ #
# Measurement-centerline windows


def _sliding_window_filter(
    seg_xy: np.ndarray,
    image: np.ndarray,
    dt: np.ndarray | None = None,
    grad_x: np.ndarray | None = None,
    grad_y: np.ndarray | None = None,
    win_len: float = 15.0,
    std_thresh: float = 0.6,
    max_angle: float = 20.0,
    strict: bool = False,
    grad_alignment_cos: float = 0.95,
    radius_rel_tol: float = 0.25,
    min_valid_fraction: float = 0.7,
) -> list[np.ndarray]:
    if strict and (dt is None or grad_x is None or grad_y is None):
        raise ValueError("Strict centerline filtering requires dt, grad_x, and grad_y.")

    pts_xy = seg_xy.reshape(-1, 2).astype(np.float32)
    if pts_xy.shape[0] < 2:
        return []

    cum_dist = [0.0]
    for i in range(1, len(pts_xy)):
        cum_dist.append(cum_dist[-1] + float(np.linalg.norm(pts_xy[i] - pts_xy[i - 1])))
    cum_dist = np.asarray(cum_dist, dtype=np.float32)
    total_len = float(cum_dist[-1])
    if total_len <= 0:
        return []

    gray = image
    h, w = gray.shape[:2]

    def sample_intensity(p_xy: np.ndarray) -> float:
        x, y = int(round(p_xy[0])), int(round(p_xy[1]))
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)
        return float(gray[y, x])

    valid_windows: list[tuple[int, int]] = []
    start_idx = 0
    half_step = float(win_len) / 2.0
    max_angle = float(max_angle)

    n_pts = len(pts_xy)
    while start_idx < n_pts - 1:
        start_s = float(cum_dist[start_idx])
        target_s = start_s + float(win_len)
        end_idx = int(np.searchsorted(cum_dist, target_s, side="left"))

        if end_idx <= start_idx + 1:
            end_idx = start_idx + 1
        if end_idx >= n_pts:
            end_idx = n_pts - 1

        window_pts_xy = pts_xy[start_idx : end_idx + 1]
        intensities = np.fromiter((sample_intensity(p_xy) for p_xy in window_pts_xy), dtype=np.float32)
        if intensities.size >= 2:
            if np.std(intensities) < std_thresh:
                ds = float(cum_dist[end_idx] - cum_dist[start_idx])
                if ds > 0.0:
                    dI = abs(float(intensities[-1]) - float(intensities[0]))
                    angle_deg = np.degrees(np.arctan(dI / ds))
                    if angle_deg < max_angle:
                        if strict:
                            if _window_high_confidence(
                                window_pts_xy,
                                dt=dt,
                                grad_x=grad_x,
                                grad_y=grad_y,
                                grad_alignment_cos=grad_alignment_cos,
                                radius_rel_tol=radius_rel_tol,
                                min_valid_fraction=min_valid_fraction,
                            ):
                                valid_windows.append((start_idx, end_idx))
                        else:
                            valid_windows.append((start_idx, end_idx))

        new_target_s = start_s + half_step
        new_start_idx = int(np.searchsorted(cum_dist, new_target_s, side="left"))
        if new_start_idx <= start_idx:
            break
        start_idx = new_start_idx

    merged: list[tuple[int, int]] = []
    for win in valid_windows:
        if not merged:
            merged.append(win)
        else:
            a0, a1 = merged[-1]
            b0, b1 = win
            if b0 <= a1:
                merged[-1] = (a0, max(a1, b1))
            else:
                merged.append(win)

    out_segments: list[np.ndarray] = []
    for s_idx, e_idx in merged:
        if float(cum_dist[e_idx] - cum_dist[s_idx]) >= float(win_len):
            sub_xy = pts_xy[s_idx : e_idx + 1].copy().reshape(-1, 1, 2)
            out_segments.append(sub_xy)
    return out_segments


def _window_high_confidence(
    pts_xy: np.ndarray,
    dt: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    grad_alignment_cos: float = 0.95,
    radius_rel_tol: float = 0.1,
    min_valid_fraction: float = 0.7,
) -> bool:
    """
    Validate a candidate window against gradient alignment and radius consistency criteria.
    """
    pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
    if pts_xy.shape[0] < 2:
        return False

    n_pts = pts_xy.shape[0]
    tangents = np.zeros_like(pts_xy, dtype=np.float32)
    tangents[1:-1] = pts_xy[2:] - pts_xy[:-2]
    tangents[0] = pts_xy[1] - pts_xy[0]
    tangents[-1] = pts_xy[-1] - pts_xy[-2]

    tan_norm = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-6
    tangents /= tan_norm
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    valid = 0
    total = 0
    for idx in range(n_pts):
        px, py = float(pts_xy[idx, 0]), float(pts_xy[idx, 1])
        r = float(_bilinear(dt, py, px))
        if r <= 1e-6:
            continue

        h = 0.5 * r
        nx, ny = float(normals[idx, 0]), float(normals[idx, 1])
        if np.isclose(nx, 0.0) and np.isclose(ny, 0.0):
            continue

        p_plus = (px + nx * h, py + ny * h)
        p_minus = (px - nx * h, py - ny * h)

        g_plus = _sample_vec2(grad_x, grad_y, p_plus[1], p_plus[0])
        g_minus = _sample_vec2(grad_x, grad_y, p_minus[1], p_minus[0])

        mag_plus = float(np.linalg.norm(g_plus))
        mag_minus = float(np.linalg.norm(g_minus))
        if mag_plus <= 1e-6 or mag_minus <= 1e-6:
            continue

        g_plus_unit = g_plus / (mag_plus + 1e-6)
        g_minus_unit = g_minus / (mag_minus + 1e-6)
        align_score = -float(np.dot(g_plus_unit, g_minus_unit))

        target = 0.5 * r
        dt_plus = float(_bilinear(dt, p_plus[1], p_plus[0]))
        dt_minus = float(_bilinear(dt, p_minus[1], p_minus[0]))
        radius_ok = (
            abs(dt_plus - target) <= radius_rel_tol * max(r, 1e-6)
            and abs(dt_minus - target) <= radius_rel_tol * max(r, 1e-6)
        )

        total += 1
        if align_score >= grad_alignment_cos and radius_ok:
            valid += 1

    if total == 0:
        return False
    return (valid / float(total)) >= float(min_valid_fraction)
