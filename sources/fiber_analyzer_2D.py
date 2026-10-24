"""2D fiber analysis pipeline combining inference, centerlines, and measurements."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from skimage.segmentation import watershed, find_boundaries

from .domain import FiberFragment
from .ml import OnnxFiberposeModel
from .ridge_detector import RidgeDetector
from .utils import (
    _bilinear,
    _sample_vec2,
    normals_from_dt_2d,
    resample_polyline_xy,
    to_uint8_for_view,
)


# ---------- Paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

logger = logging.getLogger(__name__)


class FiberAnalyzer_2D:
    """
    Non-GUI class encapsulating fiber detection, watershed, and analysis.
    The analyzer now owns the full prediction + processing pipeline:
      • predict_dt(image) runs the model, computes gradients, laplacian, and centerlines
      • segment(mode, params) dispatches watershed/radial-expansion segmentation
      • analysis helpers operate on cached centerlines and gradients
    """

    def __init__(
        self,
        model_path: str,
        device: Any,
        model_kwargs: Optional[Dict[str, Any]] = None,
        strict: bool | None = None,
        keep_segments_without_measurement_centerlines: bool | None = None,
    ):
        self.model_path = model_path
        self.device = device

        defaults: Dict[str, Any] = {
            "nbase": "1,32,64,128,256",
        }
        if model_kwargs:
            defaults.update(model_kwargs)
        defaults["device"] = device
        self._model_kwargs = defaults

        self._predictor: Optional[OnnxFiberposeModel] = None
        self._strict_centerlines = bool(strict) if strict is not None else False
        self._keep_segments_without_measurement_centerlines = (
            bool(keep_segments_without_measurement_centerlines)
            if keep_segments_without_measurement_centerlines is not None
            else False
        )

        self._segment_counter: int = 0

        self.scale_factor: float = 1.0
        self._reset_prediction_state()

    # ------------------------------------------------------------------ #
    # Internal helpers / state


    def _reset_prediction_state(self) -> None:
        """Clear all cached prediction, segmentation, and analysis state."""
        self._dt: Optional[np.ndarray] = None
        self._dt_u8: Optional[np.ndarray] = None
        self._dt_norm: Optional[np.ndarray] = None
        self._foreground_prob: Optional[np.ndarray] = None
        self._gx: Optional[np.ndarray] = None
        self._gy: Optional[np.ndarray] = None
        self._laplacian: Optional[np.ndarray] = None
        self._fiber_fragments: Optional[List[FiberFragment]] = None
        self._segmentation_centerlines_rgb: Optional[np.ndarray] = None
        self._measurement_centerlines_rgb: Optional[np.ndarray] = None
        self._centerlines_comparison_rgb: Optional[np.ndarray] = None
        self._segmentation_markers: Optional[np.ndarray] = None
        self._radial_expansion_labels: Optional[np.ndarray] = None
        self._segmentation_expansion_factor: Optional[float] = None
        self._labels: Optional[np.ndarray] = None
        self._segment_counter = 0

    @staticmethod
    def _default_centerline_kwargs() -> Dict[str, Any]:
        return {
            "sigma": 1.0,
            "low": 0.01,
            "high": 0.4,
            "polarity": "light",
            "trim": 3,
            "win_len": 25.0,
            "std_thresh": 0.8,
        }

    def set_strict_centerline_filter(self, strict: bool):
        self._strict_centerlines = bool(strict)

    def set_keep_segments_without_measurement_centerlines(self, keep: bool):
        self._keep_segments_without_measurement_centerlines = bool(keep)

    def set_device(self, device: Any):
        """Update device and keep dependent state synchronized.
        
        Args:
            device (Any): Execution device identifier (for example CPU or CUDA)."""
        if device == self.device:
            return
        self.device = device
        self._model_kwargs["device"] = str(device)
        self._predictor = None  # force re-init on next predict
        logger.info("FiberAnalyzer_2D device set to %s", device)

    def set_model_path(self, model_path: str):
        """Update model path and reset predictor/predictions."""
        if model_path == self.model_path:
            return
        self.model_path = model_path
        self._predictor = None
        self._reset_prediction_state()
        logger.info("FiberAnalyzer_2D model switched to %s", model_path)

    def _ensure_predictor(self) -> OnnxFiberposeModel:
        if self._predictor is not None:
            return self._predictor
        kwargs = dict(self._model_kwargs)
        kwargs["device"] = self.device
        self._predictor = OnnxFiberposeModel(
            model_path=self.model_path,
            **kwargs,
        )
        return self._predictor

    def _infer_model_stride(self) -> int:
        stride = 16
        predictor = self._ensure_predictor()
        try:
            if hasattr(predictor, "stride"):
                s = int(predictor.stride)
                if s > 0:
                    return s
            if hasattr(predictor, "nbase"):
                levels = str(predictor.nbase).split(",")
                downs = max(0, len(levels) - 1)
                stride = 2 ** downs
        except Exception:
            pass
        return stride

    def _run_model(self, image: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:

        predictor = self._ensure_predictor()
        stride = self._infer_model_stride()
        H, W = image.shape[:2]
        pad_h = (stride - (H % stride)) % stride
        pad_w = (stride - (W % stride)) % stride
        if pad_h or pad_w:
            if image.ndim == 2:
                pad_width = ((0, pad_h), (0, pad_w))
            else:
                pad_width = ((0, pad_h), (0, pad_w), (0, 0))
            image_pad = np.pad(image, pad_width, mode="reflect")
        else:
            image_pad = image
        dt_pad, fg_pad = predictor.predict(image_pad)
        dt = dt_pad[:H, :W] if dt_pad.ndim == 2 else dt_pad[:H, :W, 0]
        foreground_prob = None
        if fg_pad is not None:
            foreground_prob = fg_pad[:H, :W] if fg_pad.ndim == 2 else fg_pad[:H, :W, 0]
            foreground_prob = np.asarray(foreground_prob, dtype=np.float32)
        return np.asarray(dt, dtype=np.float32), foreground_prob

    def _set_dt_and_gradients(
        self,
        dt: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        foreground_prob: Optional[np.ndarray] = None,
    ):
        self._dt = np.asarray(dt, dtype=np.float32)
        self._dt_u8 = np.clip(self._dt, 0, 255).astype(np.uint8)
        self._dt_norm = self._dt / 255.0
        self._foreground_prob = (
            np.asarray(foreground_prob, dtype=np.float32)
            if foreground_prob is not None
            else None
        )
        self._gx = np.asarray(gx, dtype=np.float32)
        self._gy = np.asarray(gy, dtype=np.float32)
        if self._gx.shape != self._dt.shape or self._gy.shape != self._dt.shape:
            raise ValueError("Gradient arrays must match DT shape.")
        if self._foreground_prob is not None and self._foreground_prob.shape != self._dt.shape:
            raise ValueError("Foreground probability must match DT shape.")

    # ------------------------------------------------------------------ #
    # Public pipeline

    def predict_dt(
        self,
        image: np.ndarray,
        smooth_sigma: float = 1.5,
        grad_sigma: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict DT for the input image, compute gradients, laplacian, and centerlines.

        Returns:
            dt (float32), gx (float32), gy (float32)
        """
        if image is None:
            raise ValueError("Input image is None.")

        self._reset_prediction_state()

        dt, foreground_prob = self._run_model(image)
        dt_smooth = gaussian_filter(dt, sigma=float(smooth_sigma))
        gy, gx = np.gradient(dt_smooth)
        gy = gaussian_filter(gy, sigma=float(grad_sigma))
        gx = gaussian_filter(gx, sigma=float(grad_sigma))
        lapl_img = laplace(dt_smooth)

        self._set_dt_and_gradients(dt, gx, gy, foreground_prob=foreground_prob)
        self._laplacian = lapl_img

        return self._dt, self._gx, self._gy

    def process_centerlines(
        self,
        max_angle_deg: float,
        sigma: float = 1.0,
        low: float = 0.1,
        high: float = 0.4,
        polarity: str = "light",
        trim: int = 3,
        win_len: float = 25.0,
        std_thresh: float = 0.8,
        merge_expansion_factor: float = 0.7,
        min_fragment_centerline_length_px: float = 10.0,
        measurement_centerline_length_trim_fraction: float = 0.05,
        grad_quality_trim_fraction: float = 0.0,
        segmentation_expansion_factor: float = 1.0,
    ) -> None:
        """
        Build both centerline sets from one ridge-detection pass:
          1) segmentation_centerlines from raw ridges + radial expansion
          2) measurement_centerlines by filtering segmentation_centerlines
        """
        if self._dt_u8 is None or self._dt is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")

        detected_polylines = RidgeDetector.detect_ridges(
            image2d=self._dt_u8,
            sigma=sigma,
            low=low,
            high=high,
            polarity=polarity,
        )

        raw_fragments = self._group_detected_polylines_into_segmentation_fragments(
            detected_polylines=detected_polylines,
            dt=self._dt,
            merge_expansion_factor=float(merge_expansion_factor),
            min_fragment_centerline_length_px=float(min_fragment_centerline_length_px),
            segment_id_start=self._segment_counter,
            trim=int(trim),
        )
        self._segment_counter += len(raw_fragments)

        active_fragments = self._filter_measurement_centerlines(
            segmentation_fragments=raw_fragments,
            dt_u8=self._dt_u8,
            dt=self._dt,
            gx=self._gx,
            gy=self._gy,
            max_angle_deg=max_angle_deg,
            win_len=win_len,
            std_thresh=std_thresh,
            strict=self._strict_centerlines,
            keep_segments_without_measurement_centerlines=self._keep_segments_without_measurement_centerlines,
        )
        trim_fraction = self._clamp_trim_fraction(measurement_centerline_length_trim_fraction)
        trim_count = int(np.floor(len(active_fragments) * trim_fraction))
        if trim_count > 0:
            ordered = sorted(
                active_fragments,
                key=lambda seg: (float(seg.total_measurement_centerline_length_px), int(seg.fragment_id)),
            )
            drop_ids = {int(seg.fragment_id) for seg in ordered[:trim_count]}
            active_fragments = [seg for seg in active_fragments if int(seg.fragment_id) not in drop_ids]

        grad_quality_fraction = self._clamp_trim_fraction(grad_quality_trim_fraction)
        if self._gx is None or self._gy is None:
            raise ValueError("Gradient fields are required for grad-quality filtering.")
        grad_mag = np.sqrt(self._gx * self._gx + self._gy * self._gy)
        grad_err = np.abs(grad_mag - 1.0)
        active_fragments = self._filter_fragments_by_grad_quality(
            fragments=active_fragments,
            grad_err=grad_err,
            segmentation_expansion_factor=float(segmentation_expansion_factor),
            trim_fraction=grad_quality_fraction,
            min_valid_pixels=8,
        )

        expanded_labels = self._rasterize_segmentation_fragments(
            fragments=active_fragments,
            shape=self._dt_u8.shape,
            segmentation_expansion_factor=float(segmentation_expansion_factor),
        )

        markers = (
            self._render_markers(self._dt_u8.shape, active_fragments)
            if active_fragments
            else np.zeros(self._dt_u8.shape, dtype=np.int32)
        )

        segmentation_centerlines_rgb = self._render_centerlines_rgb(
            dt_u8=self._dt_u8,
            segments=active_fragments,
            centerline_color=(255, 0, 0),
            measurement_only=False,
        )
        measurement_centerlines_rgb = self._render_centerlines_rgb(
            dt_u8=self._dt_u8,
            segments=active_fragments,
            centerline_color=(0, 255, 0),
            measurement_only=True,
        )
        centerlines_comparison_rgb = self._render_centerline_comparison_rgb(
            dt_u8=self._dt_u8,
            fragments=active_fragments,
        )

        self._fiber_fragments = active_fragments
        self._segmentation_centerlines_rgb = segmentation_centerlines_rgb
        self._measurement_centerlines_rgb = measurement_centerlines_rgb
        self._centerlines_comparison_rgb = centerlines_comparison_rgb
        self._segmentation_markers = markers
        self._radial_expansion_labels = expanded_labels
        self._segmentation_expansion_factor = float(segmentation_expansion_factor)

        logger.info(
            "Centerlines built: segmentation=%d/%d measurement=%d",
            len(active_fragments),
            len(raw_fragments),
            sum(len(seg.measurement_centerlines_xy) for seg in active_fragments),
        )

    @classmethod
    def _group_detected_polylines_into_segmentation_fragments(
        cls,
        *,
        detected_polylines: List[np.ndarray],
        dt: np.ndarray,
        merge_expansion_factor: float = 0.7,
        min_fragment_centerline_length_px: float = 10.0,
        segment_id_start: int = 0,
        trim: int = 3,
    ) -> List[FiberFragment]:
        """
        Group detected polylines into segmentation fragments using overlap under merge expansion.
        """
        h, w = dt.shape[:2]
        expansion = max(0.0, float(merge_expansion_factor))
        trim_n = max(0, int(trim))

        owner = np.zeros((h, w), dtype=np.int32)
        uf_parent: Dict[int, int] = {}

        def uf_find(x: int) -> int:
            p = uf_parent.get(x, x)
            while p != uf_parent.get(p, p):
                p = uf_parent.get(p, p)
            root = p
            p = x
            while p != uf_parent.get(p, p):
                nxt = uf_parent.get(p, p)
                uf_parent[p] = root
                p = nxt
            return root

        def uf_union(a: int, b: int) -> None:
            ra = uf_find(a)
            rb = uf_find(b)
            if ra == rb:
                return
            if ra < rb:
                uf_parent[rb] = ra
            else:
                uf_parent[ra] = rb

        polyline_records: List[tuple[int, np.ndarray, np.ndarray]] = []
        local_label = 0
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
            radii = np.asarray(dt[y_idx, x_idx], dtype=np.float32)

            local_label += 1
            uf_parent.setdefault(local_label, local_label)
            pts_xy = pts_xy.astype(np.float32, copy=False)
            polyline_records.append((local_label, pts_xy, radii))

            line_mask = np.zeros((h, w), dtype=np.uint8)
            for pt_xy, radius in zip(pts_xy, radii):
                rr = max(1, int(round(float(radius) * expansion)))
                x = int(round(float(pt_xy[0])))
                y = int(round(float(pt_xy[1])))
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                cv2.circle(line_mask, (x, y), rr, 1, thickness=-1)

            mask_bool = line_mask.astype(bool, copy=False)
            if not mask_bool.any():
                continue
            overlap_ids = np.unique(owner[mask_bool])
            overlap_ids = overlap_ids[overlap_ids > 0]
            for other_id in overlap_ids:
                uf_union(local_label, int(other_id))
            owner[mask_bool & (owner == 0)] = local_label

        if not polyline_records:
            return []

        root_to_label: Dict[int, int] = {}
        next_label = 1
        grouped_polylines: Dict[int, list[np.ndarray]] = {}
        grouped_radii: Dict[int, list[np.ndarray]] = {}
        for local_label, pts_xy, radii in polyline_records:
            root = uf_find(int(local_label))
            label_id = root_to_label.get(root)
            if label_id is None:
                label_id = next_label
                root_to_label[root] = label_id
                next_label += 1
            grouped_polylines.setdefault(label_id, []).append(pts_xy)
            grouped_radii.setdefault(label_id, []).append(radii)

        fragments: List[FiberFragment] = []
        for label_id in sorted(grouped_polylines):
            max_centerline_length = max(
                FiberFragment._centerline_length(pts_xy) for pts_xy in grouped_polylines[label_id]
            )
            if max_centerline_length < float(min_fragment_centerline_length_px):
                continue
            frag_id = int(segment_id_start + len(fragments) + 1)
            fragments.append(
                FiberFragment(
                    fragment_id=frag_id,
                    segment_centerlines_xy=grouped_polylines[label_id],
                    segment_radii=grouped_radii[label_id],
                    parent_fiber_id=int(label_id),
                )
            )

        return fragments

    @staticmethod
    def _rasterize_segmentation_fragments(
        *,
        fragments: List[FiberFragment],
        shape: Tuple[int, int],
        segmentation_expansion_factor: float = 1.0,
    ) -> np.ndarray:
        """Render radial-expansion labels from merged segmentation fragments."""
        h, w = shape
        expansion = max(0.0, float(segmentation_expansion_factor))
        expanded_labels = np.zeros((h, w), dtype=np.int32)
        for frag in fragments:
            label_id = int(frag.parent_fiber_id)
            for pts_xy, radii in zip(frag.segment_centerlines_xy, frag.segment_radii):
                for pt_xy, radius in zip(pts_xy, radii):
                    rr = max(1, int(round(float(radius) * expansion)))
                    x = int(round(float(pt_xy[0])))
                    y = int(round(float(pt_xy[1])))
                    if x < 0 or y < 0 or x >= w or y >= h:
                        continue
                    cv2.circle(expanded_labels, (x, y), rr, label_id, thickness=-1)
        return expanded_labels

    @staticmethod
    def _fragment_grad_quality_score(
        fragment: FiberFragment,
        grad_err: np.ndarray,
        segmentation_expansion_factor: float,
    ) -> tuple[float, int]:
        """Compute per-fragment grad-quality score excluding centerline pixels."""
        h, w = grad_err.shape[:2]
        expansion = max(0.0, float(segmentation_expansion_factor))

        fragment_mask = np.zeros((h, w), dtype=np.uint8)
        for pts_xy, radii in zip(fragment.segment_centerlines_xy, fragment.segment_radii):
            for pt_xy, radius in zip(pts_xy, radii):
                rr = max(1, int(round(float(radius) * expansion)))
                x = int(round(float(pt_xy[0])))
                y = int(round(float(pt_xy[1])))
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                cv2.circle(fragment_mask, (x, y), rr, 1, thickness=-1)

        centerline_mask = np.zeros((h, w), dtype=np.uint8)
        for pts_xy in fragment.segment_centerlines_xy:
            if pts_xy.size == 0:
                continue
            x_idx = np.clip(np.rint(pts_xy[:, 0]).astype(np.int32), 0, w - 1)
            y_idx = np.clip(np.rint(pts_xy[:, 1]).astype(np.int32), 0, h - 1)
            centerline_mask[y_idx, x_idx] = 1

        valid_mask = (
            fragment_mask.astype(bool, copy=False)
            & (~centerline_mask.astype(bool, copy=False))
            & np.isfinite(grad_err)
        )
        if not np.any(valid_mask):
            return float("nan"), 0

        values = np.asarray(grad_err[valid_mask], dtype=np.float32)
        if values.size == 0:
            return float("nan"), 0
        return float(np.median(values)), int(values.size)

    @classmethod
    def _filter_fragments_by_grad_quality(
        cls,
        *,
        fragments: List[FiberFragment],
        grad_err: np.ndarray,
        segmentation_expansion_factor: float,
        trim_fraction: float,
        min_valid_pixels: int = 8,
    ) -> List[FiberFragment]:
        """Drop unreliable fragments and trim worst grad-quality scores."""
        reliable_fragments: List[FiberFragment] = []
        for fragment in fragments:
            score, valid_pixels = cls._fragment_grad_quality_score(
                fragment=fragment,
                grad_err=grad_err,
                segmentation_expansion_factor=segmentation_expansion_factor,
            )
            fragment.grad_quality_score = float(score)
            if valid_pixels < int(min_valid_pixels):
                continue
            if not np.isfinite(score):
                continue
            reliable_fragments.append(fragment)

        drop_count = int(np.floor(len(reliable_fragments) * float(trim_fraction)))
        if drop_count <= 0:
            return reliable_fragments

        ordered = sorted(
            reliable_fragments,
            key=lambda seg: (-float(seg.grad_quality_score), int(seg.fragment_id)),
        )
        drop_ids = {int(seg.fragment_id) for seg in ordered[:drop_count]}
        return [seg for seg in reliable_fragments if int(seg.fragment_id) not in drop_ids]

    @classmethod
    def _filter_measurement_centerlines(
        cls,
        *,
        segmentation_fragments: List[FiberFragment],
        dt_u8: np.ndarray,
        dt: np.ndarray,
        gx: np.ndarray | None,
        gy: np.ndarray | None,
        max_angle_deg: float,
        win_len: float,
        std_thresh: float,
        strict: bool,
        keep_segments_without_measurement_centerlines: bool,
    ) -> List[FiberFragment]:
        """Collect measurement centerlines per fragment and update fragments in place."""
        active_segmentation_fragments: List[FiberFragment] = []
        for seg in segmentation_fragments:
            seg.measurement_centerlines_xy = []
            for polyline_xy in seg.segment_centerlines_xy:
                seg_xy = polyline_xy.reshape(-1, 1, 2)
                windows_xy = cls._sliding_window_filter(
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

    @staticmethod
    def _clamp_trim_fraction(value: float) -> float:
        value = float(value)
        if not np.isfinite(value):
            return 0.05
        return float(min(0.95, max(0.0, value)))

    @staticmethod
    def _render_centerlines_rgb(
        dt_u8: np.ndarray,
        segments: List[FiberFragment],
        centerline_color: Tuple[int, int, int] = (255, 0, 0),
        measurement_only: bool = False,
    ) -> np.ndarray:
        base_dt = to_uint8_for_view(dt_u8)
        centerlines_rgb = cv2.cvtColor(base_dt, cv2.COLOR_GRAY2BGR)

        for seg in segments:
            polylines = (
                seg.measurement_centerlines_xy
                if measurement_only
                else seg.segment_centerlines_xy
            )
            for polyline_xy in polylines:
                pts = np.asarray(polyline_xy, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
                if len(pts) < 2:
                    x, y = pts[0, 0]
                    cv2.circle(
                        centerlines_rgb,
                        (x, y),
                        1,
                        centerline_color,
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )
                else:
                    cv2.polylines(
                        centerlines_rgb,
                        [pts],
                        isClosed=False,
                        color=centerline_color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        return centerlines_rgb

    @staticmethod
    def _render_centerline_comparison_rgb(
        dt_u8: np.ndarray,
        fragments: List[FiberFragment],
    ) -> np.ndarray:
        """Render segmentation and measurement centerlines together for debugging."""
        base_dt = to_uint8_for_view(dt_u8)
        out = cv2.cvtColor(base_dt, cv2.COLOR_GRAY2BGR)
        for seg in fragments:
            for polyline_xy in seg.segment_centerlines_xy:
                pts = polyline_xy.reshape(-1, 1, 2).astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(out, [pts], isClosed=False, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        for seg in fragments:
            if not seg.has_measurement_centerline:
                continue
            for polyline_xy in seg.measurement_centerlines_xy:
                pts = polyline_xy.reshape(-1, 1, 2).astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(out, [pts], isClosed=False, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        return out

    def segment(
        self,
        mode: str = "watershed",
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Run segmentation from cached prediction/centerline state.

        Supported modes:
            - watershed: params include {"gradient_threshold": float}
            - radial_expansion: params include {"expansion_factor": float}
        """
        if self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")

        opts = dict(params or {})
        mode_name = str(mode or "watershed").strip().lower()

        if mode_name == "watershed":
            gradient_threshold = float(opts.get("gradient_threshold", 0.8))
            if self._segmentation_markers is None:
                raise ValueError("Centerlines not available. Call process_centerlines first.")
            labels = self.segment_watershed(
                gradient_threshold=gradient_threshold,
                min_segment_foreground_mean=float(opts.get("min_segment_foreground_mean", 0.5)),
            )
        elif mode_name == "radial_expansion":
            labels = self.segment_radial_expansion(
                expansion_factor=float(opts.get("expansion_factor", 1.0)),
                min_segment_foreground_mean=float(opts.get("min_segment_foreground_mean", 0.5)),
            )
        else:
            raise ValueError(f"Unsupported segmentation mode: {mode}")

        surviving_ids = set(np.unique(labels)) - {0}
        self._fiber_fragments = [
            fragment
            for fragment in self._fiber_fragments or []
            if int(fragment.parent_fiber_id) in surviving_ids
        ]
        self._measurement_centerlines_rgb = self._render_centerlines_rgb(
            self._dt_u8, self._fiber_fragments, centerline_color=(0, 255, 0), measurement_only=True
        )
        self._centerlines_comparison_rgb = self._render_centerline_comparison_rgb(
            self._dt_u8, self._fiber_fragments
        )
        return labels

    def segment_watershed(
        self,
        gradient_threshold: float,
        min_segment_foreground_mean: float = 0.5,
    ) -> np.ndarray:
        """
        Run watershed using a gradient-magnitude mask plus dilated centerlines.
        """

        if self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")
        if self._gx is None or self._gy is None:
            raise ValueError("Gradient fields are not available. Call predict_dt first.")
        if self._segmentation_markers is None or self._segmentation_markers.max() == 0:
            return np.zeros_like(self._dt_u8, dtype=np.uint16)

        logger.info("Running watershed segmentation")

        labels = self.watershed_gradient_mask_segmentation(
            self._dt_u8,
            self._gx,
            self._gy,
            self._segmentation_markers,
            gradient_threshold=gradient_threshold,
        )
        labels = self._prune_labels_by_foreground(
            labels,
            foreground_prob=self._foreground_prob,
            min_segment_foreground_mean=min_segment_foreground_mean,
        )
        self._update_centerline_overlay_with_watershed(self._segmentation_markers, labels)
        self._labels = labels
        return labels
    
    @staticmethod
    def watershed_gradient_mask_segmentation(
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

    def segment_radial_expansion(
        self,
        expansion_factor: float = 1.0,
        min_segment_foreground_mean: float = 0.5,
    ) -> np.ndarray:
        """Segment using the cached expanded ridge-label map."""
        if self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")
        target_factor = float(expansion_factor)
        cached_factor = self._segmentation_expansion_factor
        missing_or_stale = (
            self._radial_expansion_labels is None
            or cached_factor is None
            or (not np.isfinite(cached_factor))
            or (not np.isclose(float(cached_factor), target_factor, rtol=1e-9, atol=1e-9))
        )
        if missing_or_stale:
            raise ValueError("Centerlines not available. Call process_centerlines first.")
        labels_i32 = np.asarray(self._radial_expansion_labels, dtype=np.int32)
        labels = np.clip(labels_i32, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
        labels = self._prune_labels_by_foreground(
            labels,
            foreground_prob=self._foreground_prob,
            min_segment_foreground_mean=min_segment_foreground_mean,
        )
        self._labels = labels
        return labels

    @staticmethod
    def _prune_labels_by_foreground(
        labels: np.ndarray,
        *,
        foreground_prob: Optional[np.ndarray],
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
        for label_id in np.unique(labels_arr):
            if int(label_id) <= 0:
                continue
            label_mask = labels_arr == label_id
            if float(np.mean(foreground_arr[label_mask])) < threshold:
                pruned[label_mask] = 0
        return pruned.astype(labels_arr.dtype, copy=False)

    @staticmethod
    def _render_markers(shape: Tuple[int, int], segments: List[FiberFragment]) -> np.ndarray:
        markers = np.zeros(shape, dtype=np.int32)
        for seg in segments:
            for polyline_xy in seg.segment_centerlines_xy:
                pts = np.asarray(polyline_xy, dtype=np.float32)
                if pts.size == 0:
                    continue
                pts_int = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
                if len(pts_int) < 2:
                    cx, cy = pts_int[0, 0]
                    cv2.circle(markers, (cx, cy), 1, int(seg.parent_fiber_id), thickness=-1)
                    continue
                cv2.polylines(
                    markers,
                    [pts_int],
                    isClosed=False,
                    color=int(seg.parent_fiber_id),
                    thickness=1,
                    lineType=cv2.LINE_8,
                )
        return markers

    def get_dt(self) -> Optional[np.ndarray]:
        """Return dt in the format expected by downstream consumers.
        
        Returns:
            Optional[np.ndarray]: Array containing the transformed or derived data."""
        return self._dt

    def get_dt_u8(self) -> Optional[np.ndarray]:
        """Return dt u8 in the format expected by downstream consumers.
        
        Returns:
            Optional[np.ndarray]: Array containing the transformed or derived data."""
        return self._dt_u8

    def get_dt_norm(self) -> Optional[np.ndarray]:
        """Return dt norm in the format expected by downstream consumers.
        
        Returns:
            Optional[np.ndarray]: Array containing the transformed or derived data."""
        return self._dt_norm

    def get_foreground_prob(self) -> Optional[np.ndarray]:
        """Return model foreground probability, if the checkpoint provides it."""
        return self._foreground_prob

    def get_gradients(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return gradients in the format expected by downstream consumers.
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Array containing the transformed or derived data."""
        return self._gx, self._gy

    def get_laplacian(self) -> Optional[np.ndarray]:
        """Return laplacian in the format expected by downstream consumers.
        
        Returns:
            Optional[np.ndarray]: Array containing the transformed or derived data."""
        return self._laplacian

    def get_segmentation_centerlines_image(self) -> Optional[np.ndarray]:
        """Return segmentation centerline overlay image."""
        return self._segmentation_centerlines_rgb

    def get_measurement_centerlines_image(self) -> Optional[np.ndarray]:
        """Return measurement centerline overlay image."""
        return self._measurement_centerlines_rgb

    def get_centerlines_comparison_image(self) -> Optional[np.ndarray]:
        """Return combined centerline debug overlay image."""
        return self._centerlines_comparison_rgb



    def get_measurement_centerline_segments(self) -> List[np.ndarray]:
        if not self._fiber_fragments:
            return []
        segments_xy: List[np.ndarray] = []
        for seg in self._fiber_fragments:
            segments_xy.extend(polyline_xy.reshape(-1, 1, 2) for polyline_xy in seg.measurement_centerlines_xy)
        return segments_xy

    def get_measurement_fragment_count(self) -> int:
        if not self._fiber_fragments:
            return 0
        return sum(1 for seg in self._fiber_fragments if seg.has_measurement_centerline)


    def get_labels(self) -> Optional[np.ndarray]:
        """Return labels in the format expected by downstream consumers.
        
        Returns:
            Optional[np.ndarray]: Array containing the transformed or derived data."""
        return self._labels

    # ------------------------------------------------------------------ #
    # Watershed helpers

    def _update_centerline_overlay_with_watershed(self, markers: np.ndarray | None, labels: np.ndarray | None) -> None:
        """
        Render DT background with marker pixels (red) and watershed boundaries (blue) for visualization.
        """
        if self._dt_u8 is None and self._dt_norm is None:
            return
        if self._dt_norm is not None:
            base_gray = to_uint8_for_view(self._dt_norm)
        else:
            base_gray = np.asarray(self._dt_u8, dtype=np.uint8)
        base = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
        if labels is not None:
            try:
                boundaries = find_boundaries(labels, mode="outer")
            except Exception:
                boundaries = None
            if boundaries is not None and boundaries.any():
                base[boundaries] = (0, 0, 255)  # blue in BGR
        if markers is not None:
            marker_mask = markers.astype(np.int32) > 0
            if marker_mask.any():
                base[marker_mask] = (255, 0, 0)  # red in BGR
        self._segmentation_centerlines_rgb = base

    # --- Pore sampling via DT local maxima -----------------------------------

    def void_mask_and_pore_dt(self, thr_u8: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns:
            void_mask_u8: uint8, {0,1}, where 1 marks void (pores)
            pore_dt: float32, Euclidean distance (in pixels) inside voids
        """
        if self._dt_u8 is None:
            return None, None
        t = float(thr_u8) / 255.0
        dt01 = self._dt_u8.astype(np.float32) / 255.0
        fiber_mask = (dt01 >= t)
        void_mask = (~fiber_mask).astype(np.uint8)
        if void_mask.any():
            void_mask = self._clean_void_mask(void_mask, radius_px=2)
        if void_mask.any():
            pore_dt = cv2.distanceTransform(void_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
        else:
            pore_dt = np.zeros_like(dt01, dtype=np.float32)
        return void_mask, pore_dt

    def _clean_void_mask(self, void_mask: np.ndarray, radius_px: int = 2) -> np.ndarray:
        """Morphologically open void mask to drop thin artifacts before DT."""
        r = max(0, int(radius_px))
        if r == 0:
            return void_mask
        k = 2 * r + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(void_mask, cv2.MORPH_OPEN, kernel)

    def _local_maxima(self, arr: np.ndarray, footprint: int = 3, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Return boolean mask of local maxima in arr using grayscale dilation.
        - footprint: odd window size (3,5,7...).
        - mask: optional boolean/uint8 mask to restrict candidates (e.g., void area).
        Notes:
            • Pixels with value 0 are treated as background and ignored
              (set temporarily to 255 so they never appear as maxima).
        """
        k = max(3, int(footprint) | 1)  # ensure odd >= 3
        arr_work = arr.copy()

        arr_work[arr_work == 0] = 255

        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dil = cv2.dilate(arr_work, ker)
        lm = (arr_work == dil) & (arr > 0)

        if mask is not None:
            lm &= (mask.astype(bool))
        return lm

    def sample_inscribed_circles(
        self,
        pore_dt: np.ndarray,
        void_mask: np.ndarray,
        min_radius_px: float = 1.0,
        min_peak_px: float = 3.0,
        peak_footprint: int = 3,
        max_pores: int | None = None,
    ) -> List[dict]:
        if pore_dt is None or void_mask is None:
            return []

        scale = self.scale_factor if getattr(self, "scale_factor", None) else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        H, W = pore_dt.shape[:2]
        current_void = (void_mask.astype(np.uint8) > 0).astype(np.uint8)

        lm = self._local_maxima(pore_dt, footprint=peak_footprint, mask=current_void)
        ys, xs = np.nonzero(lm)
        if xs.size == 0:
            return []

        radii = pore_dt[ys, xs]
        keep = radii >= float(min_peak_px)
        if not np.any(keep):
            return []
        xs, ys, radii = xs[keep], ys[keep], radii[keep]

        order = np.argsort(-radii)
        xs, ys, radii = xs[order], ys[order], radii[order]

        circles: List[dict] = []
        for x, y, r in zip(xs, ys, radii):
            if r < float(min_radius_px):
                continue
            r_int = int(np.floor(float(r)))
            if r_int <= 0:
                continue
            if x - r_int < 0 or x + r_int >= W or y - r_int < 0 or y + r_int >= H:
                continue

            cmask = np.zeros_like(current_void, dtype=np.uint8)
            cv2.circle(cmask, (int(x), int(y)), r_int, 1, thickness=-1)
            if np.all(cmask <= current_void):
                true_radius = float(r) / scale
                circles.append({'x': int(x), 'y': int(y), 'r_px': float(r), 'r_px_orig': true_radius})
                current_void[cmask == 1] = 0
                if (max_pores is not None) and (len(circles) >= max_pores):
                    break
        return circles

    # ------------------------------------------------------------------ #
    # Analysis

    @staticmethod
    def analyze_from_signatures(
        dt: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        constant_segments_xy: List[np.ndarray],
        spacing_px: float = 5.0,
        scale_factor: float = 1.0,
    ) -> tuple[
        List[tuple[int, int, float, float, float, float]],
        List[tuple[int, int, float, float, float, float]],
    ]:
        """
        Pure analysis core with explicit inputs.

        Returns:
          - measurement_samples: (y, x, ny, nx, r, angle_deg) in original/display scale
          - drawing_samples: (y, x, ny, nx, r, angle_deg) in processing-image scale
        """
        if dt is None or gx is None or gy is None:
            return [], []

        segments = constant_segments_xy or []
        spacing = max(1.0, float(spacing_px))
        scale = float(scale_factor)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        measurement_samples: List[tuple[int, int, float, float, float, float]] = []
        drawing_samples: List[tuple[int, int, float, float, float, float]] = []

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

        for seg_xy in segments:
            seg_xy = np.asarray(seg_xy).reshape(-1, 2).astype(np.float32)
            if len(seg_xy) < 2:
                continue

            samples_xy = resample_polyline_xy(seg_xy, step=spacing)
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

                drawing_samples.append(
                    (int(round(y)), int(round(x)), ny, nx, r, angle_deg)
                )
                measurement_samples.append(
                    (
                        int(round(y / scale)),
                        int(round(x / scale)),
                        ny,
                        nx,
                        r / scale,
                        angle_deg,
                    )
                )

        return measurement_samples, drawing_samples

    def analyze(self, spacing_px: float = 5.0):
        """
        Run thickness/orientation sampling on current centerline segments.

        Returns:
          - measurement_samples: (y, x, ny, nx, r, angle_deg) in original/display scale
          - drawing_samples: (y, x, ny, nx, r, angle_deg) in processing-image scale

        Note:
            `_xy` suffix denotes (x, y) ordering, while `_yx` denotes (y, x).
        """
        if self._fiber_fragments:
            constant_segments_xy = []
            for seg in self._fiber_fragments:
                constant_segments_xy.extend(
                    polyline_xy.reshape(-1, 1, 2) for polyline_xy in seg.measurement_centerlines_xy
                )
        else:
            constant_segments_xy = []

        return self.analyze_from_signatures(
            dt=self._dt,
            gx=self._gx,
            gy=self._gy,
            constant_segments_xy=constant_segments_xy,
            spacing_px=spacing_px,
            scale_factor=self.scale_factor,
        )

    # ------------------------------------------------------------------ #
    # Local utilities (used internally and optionally by GUI helpers)


    @staticmethod
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
    ) -> List[np.ndarray]:
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

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        h, w = gray.shape[:2]

        def sample_intensity(p_xy: np.ndarray) -> float:
            x, y = int(round(p_xy[0])), int(round(p_xy[1]))
            x = min(max(x, 0), w - 1)
            y = min(max(y, 0), h - 1)
            return float(gray[y, x])

        valid_windows: List[Tuple[int, int]] = []
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
                                if FiberAnalyzer_2D._window_high_confidence(
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

        merged: List[Tuple[int, int]] = []
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

        out_segments: List[np.ndarray] = []
        for s_idx, e_idx in merged:
            if float(cum_dist[e_idx] - cum_dist[s_idx]) >= float(win_len):
                sub_xy = pts_xy[s_idx : e_idx + 1].copy().reshape(-1, 1, 2)
                out_segments.append(sub_xy)
        return out_segments

    @staticmethod
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
