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
        """Initialize fiber analyzer 2 d and establish required runtime state.
        
        Args:
            model_path (str): Filesystem location describing where model path is read or written.
            device (Any): Execution target device (for example CPU or CUDA).
            model_kwargs (Optional[Dict[str, Any]]): Value for model kwargs interpreted by this routine.
            strict (bool | None): Boolean flag that toggles strict behavior.
            keep_segments_without_measurement_centerlines (bool | None): Keep/remove behavior for segments
                that do not produce measurement centerlines."""
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

        self._centerline_kwargs: Dict[str, Any] = self._default_centerline_kwargs()
        self._last_max_angle: float = 0.0
        self._segment_counter: int = 0

        self.scale_factor: float = 1.0
        self._reset_prediction_state()

    # ------------------------------------------------------------------ #
    # Internal helpers / state

    def _has_dt_prediction(self) -> bool:
        """Return whether a distance-transform prediction is currently cached."""
        return self._dt is not None

    def _reset_prediction_state(self) -> None:
        """Clear all cached prediction, segmentation, and analysis state."""
        self._dt: Optional[np.ndarray] = None
        self._dt_u8: Optional[np.ndarray] = None
        self._dt_norm: Optional[np.ndarray] = None
        self._gx: Optional[np.ndarray] = None
        self._gy: Optional[np.ndarray] = None
        self._laplacian: Optional[np.ndarray] = None
        self._fiber_fragments: Optional[List[FiberFragment]] = None
        self._measurement_normals_xy: Optional[List[np.ndarray]] = None
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
        """Provide an internal helper for default centerline kwargs.
        
        Returns:
            Dict[str, Any]: Mapping containing named default centerline kwargs outputs and metadata."""
        return {
            "sigma": 1.0,
            "low": 0.01,
            "high": 0.4,
            "polarity": "light",
            "trim": 3,
            "tol": 3.0,
            "win_len": 25.0,
            "std_thresh": 0.8,
        }

    def set_strict_centerline_filter(self, strict: bool):
        """Update strict centerline filter and synchronize dependent runtime state.
        
        Args:
            strict (bool): Boolean flag that toggles strict behavior."""
        self._strict_centerlines = bool(strict)

    def set_keep_segments_without_measurement_centerlines(self, keep: bool):
        """Update whether segmentation keeps fragments lacking measurement centerlines.
        
        Args:
            keep (bool): Boolean flag that toggles keep behavior."""
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
        """Provide an internal helper for ensure predictor.
        
        Returns:
            OnnxFiberposeModel: Result object produced by the ensure predictor routine."""
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
        """Provide an internal helper for infer model stride.
        
        Returns:
            int: Result object produced by the infer model stride routine."""
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

    def _run_model(self, image: np.ndarray) -> np.ndarray:

        """Provide an internal helper for run model.
        
        Args:
            image (np.ndarray): Array-like data carrying image values for analysis.
        
        Returns:
            np.ndarray: Array containing the run model output aligned with downstream expectations."""
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
        dt_pad = predictor.predict(image_pad)
        dt = dt_pad[:H, :W] if dt_pad.ndim == 2 else dt_pad[:H, :W, 0]
        return np.asarray(dt, dtype=np.float32)

    def _set_dt_and_gradients(self, dt: np.ndarray, gx: np.ndarray, gy: np.ndarray):
        """Provide an internal helper for set dt and gradients.
        
        Args:
            dt (np.ndarray): Value for dt interpreted by this routine.
            gx (np.ndarray): Value for gx interpreted by this routine.
            gy (np.ndarray): Value for gy interpreted by this routine.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        self._dt = np.asarray(dt, dtype=np.float32)
        self._dt_u8 = np.clip(self._dt, 0, 255).astype(np.uint8)
        self._dt_norm = self._dt / 255.0
        self._gx = np.asarray(gx, dtype=np.float32)
        self._gy = np.asarray(gy, dtype=np.float32)
        if self._gx.shape != self._dt.shape or self._gy.shape != self._dt.shape:
            raise ValueError("Gradient arrays must match DT shape.")

    # ------------------------------------------------------------------ #
    # Public pipeline

    def predict_dt(
        self,
        image: np.ndarray,
        max_angle_deg: float,
        centerline_kwargs: Optional[Dict[str, Any]] = None,
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

        dt = self._run_model(image)
        dt_smooth = gaussian_filter(dt, sigma=float(smooth_sigma))
        gy, gx = np.gradient(dt_smooth)
        gy = gaussian_filter(gy, sigma=float(grad_sigma))
        gx = gaussian_filter(gx, sigma=float(grad_sigma))
        lapl_img = laplace(dt_smooth)

        self._set_dt_and_gradients(dt, gx, gy)
        self._laplacian = lapl_img

        if centerline_kwargs:
            self._centerline_kwargs = centerline_kwargs
        else:
            self._centerline_kwargs = self._default_centerline_kwargs()

        self._last_max_angle = float(max_angle_deg)

        return self._dt, self._gx, self._gy

    def process_centerlines(
        self,
        max_angle_deg: float,
        sigma: float = 1.0,
        low: float = 0.1,
        high: float = 0.4,
        polarity: str = "light",
        trim: int = 3,
        tol: float = 3.0,
        win_len: float = 25.0,
        std_thresh: float = 0.8,
        merge_expansion_factor: float = 0.7,
        min_fragment_centerline_length_px: float = 10.0,
        segmentation_expansion_factor: float = 1.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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

        segmentation_fragments = self._group_detected_polylines_into_segmentation_fragments(
            detected_polylines=detected_polylines,
            dt=self._dt,
            merge_expansion_factor=float(merge_expansion_factor),
            min_fragment_centerline_length_px=float(min_fragment_centerline_length_px),
            segment_id_start=self._segment_counter,
            trim=int(trim),
        )
        self._segment_counter += len(segmentation_fragments)

        active_fragments = self._filter_measurement_centerlines(
            segmentation_fragments=segmentation_fragments,
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

        measurement_segments_xy: List[np.ndarray] = []
        measurement_normals_xy: List[np.ndarray] = []
        for seg in active_fragments:
            for polyline_xy in seg.measurement_centerlines_xy:
                measurement_segments_xy.append(polyline_xy.reshape(-1, 1, 2))
                measurement_normals_xy.append(self._polyline_normals(polyline_xy))
        self._fiber_fragments = active_fragments
        self._measurement_normals_xy = measurement_normals_xy
        self._segmentation_centerlines_rgb = segmentation_centerlines_rgb
        self._measurement_centerlines_rgb = measurement_centerlines_rgb
        self._centerlines_comparison_rgb = centerlines_comparison_rgb
        self._segmentation_markers = markers
        self._radial_expansion_labels = expanded_labels
        self._segmentation_expansion_factor = float(segmentation_expansion_factor)

        logger.info(
            "Centerlines built: segmentation=%d/%d measurement=%d",
            len(active_fragments),
            len(segmentation_fragments),
            len(measurement_segments_xy),
        )
        return measurement_segments_xy, self._measurement_normals_xy, measurement_centerlines_rgb, markers

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
            seg.measurement_radii = []
            seg.measurement_source_indices = []
            for source_index, polyline_xy in enumerate(seg.segment_centerlines_xy):
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
                    x_idx = np.clip(np.rint(pts_xy[:, 0]).astype(np.int32), 0, dt.shape[1] - 1)
                    y_idx = np.clip(np.rint(pts_xy[:, 1]).astype(np.int32), 0, dt.shape[0] - 1)
                    radii = np.asarray(dt[y_idx, x_idx], dtype=np.float32)
                    seg.measurement_centerlines_xy.append(pts_xy)
                    seg.measurement_radii.append(radii)
                    seg.measurement_source_indices.append(int(source_index))
            if seg.has_measurement_centerline:
                active_segmentation_fragments.append(seg)
        if bool(keep_segments_without_measurement_centerlines):
            active_segmentation_fragments = list(segmentation_fragments)
        return active_segmentation_fragments

    @staticmethod
    def _render_centerlines_rgb(
        dt_u8: np.ndarray,
        segments: List[FiberFragment],
        centerline_color: Tuple[int, int, int] = (255, 0, 0),
        measurement_only: bool = False,
    ) -> np.ndarray:
        """Provide an internal helper for render centerlines rgb.
        
        Args:
            dt_u8 (np.ndarray): Value for dt u8 interpreted by this routine.
            segments (List[Any]): Value for segments interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the render centerlines rgb output aligned with downstream expectations."""
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
            - watershed: params = {"threshold_u8": int}
            - radial_expansion: params = {"expansion_factor": float}
        """
        if self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")

        opts = dict(params or {})
        mode_name = str(mode or "watershed").strip().lower()

        if mode_name == "watershed":
            thr_u8 = int(opts.get("threshold_u8", 128))
            if self._segmentation_markers is None or self._segmentation_markers.max() == 0:
                self.process_centerlines(self._last_max_angle, **self._centerline_kwargs)
            return self.segment_watershed(thr_u8=thr_u8)

        if mode_name == "radial_expansion":
            return self.segment_radial_expansion(
                expansion_factor=float(opts.get("expansion_factor", 1.0)),
            )

        raise ValueError(f"Unsupported segmentation mode: {mode}")

    def segment_watershed(self, thr_u8: int) -> np.ndarray:
        """
        Run watershed segmentation using cached DT and markers.
        """

        if self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")
        if self._segmentation_markers is None or self._segmentation_markers.max() == 0:
            return np.zeros_like(self._dt_u8, dtype=np.uint16)

        logger.info("Running watershed segmentation")

        labels = self.watershed_dt_segmentation(self._dt_u8, self._segmentation_markers, thr_u8=thr_u8)
        self._update_centerline_overlay_with_watershed(self._segmentation_markers, labels)
        self._labels = labels
        return labels
    
    @staticmethod
    def watershed_dt_segmentation(dt_u8: np.ndarray, markers: np.ndarray, thr_u8: int) -> np.ndarray:
        """
        Watershed segmentation on dt implementation
        """

        t = float(thr_u8) / 255.0
        dt_norm = dt_u8.astype(np.float32) / 255.0
        fiber_region_mask = dt_norm >= t
        elev = 1.0 - dt_norm
        if not fiber_region_mask.any():
            return np.zeros_like(dt_u8, dtype=np.uint16)
        conn8 = np.ones((3, 3), dtype=bool)
        labels = watershed(
            elev,
            markers=markers,
            mask=fiber_region_mask,
            connectivity=conn8,
        ).astype(np.uint16)
        return labels

    def segment_radial_expansion(self, expansion_factor: float = 1.0) -> np.ndarray:
        """Segment using the cached expanded ridge-label map."""
        if self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")
        target_factor = float(expansion_factor)
        cached_factor = self._segmentation_expansion_factor
        needs_rebuild = (
            self._radial_expansion_labels is None
            or cached_factor is None
            or (not np.isfinite(cached_factor))
            or (not np.isclose(float(cached_factor), target_factor, rtol=1e-9, atol=1e-9))
        )
        if needs_rebuild:
            self.process_centerlines(
                self._last_max_angle,
                **self._centerline_kwargs,
                segmentation_expansion_factor=target_factor,
            )
        labels_i32 = np.asarray(self._radial_expansion_labels, dtype=np.int32, copy=False)
        labels = np.clip(labels_i32, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
        self._labels = labels
        return labels

    @staticmethod
    def _render_markers(shape: Tuple[int, int], segments: List[FiberFragment]) -> np.ndarray:
        """Provide an internal helper for render markers.
        
        Args:
            shape (Tuple[int, int]): Value for shape interpreted by this routine.
            segments (List[FiberFragment]): Value for segments interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the render markers output aligned with downstream expectations."""
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

    def get_centerlines_image(self) -> Optional[np.ndarray]:
        """Backward-compatible alias for segmentation centerline image."""
        return self.get_segmentation_centerlines_image()

    def get_segmentation_centerline_segments(self) -> List[np.ndarray]:
        if not self._fiber_fragments:
            return []
        segments_xy: List[np.ndarray] = []
        for seg in self._fiber_fragments:
            segments_xy.extend(polyline_xy.reshape(-1, 1, 2) for polyline_xy in seg.segment_centerlines_xy)
        return segments_xy

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

    def get_centerline_segments(self) -> List[np.ndarray]:
        """Backward-compatible alias for measurement centerline segments."""
        return self.get_measurement_centerline_segments()

    def get_centerline_normals(self) -> List[np.ndarray]:
        """Return centerline normals in the format expected by downstream consumers.
        
        Returns:
            List[np.ndarray]: Array containing the transformed or derived data."""
        return self._measurement_normals_xy or []

    def get_markers(self) -> Optional[np.ndarray]:
        """Return markers in the format expected by downstream consumers.
        
        Returns:
            Optional[np.ndarray]: Array containing the transformed or derived data."""
        return self._segmentation_markers

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
        """Orchestrate sample inscribed circles within the current module workflow.
        
        Args:
            pore_dt (np.ndarray): Value for pore dt interpreted by this routine.
            void_mask (np.ndarray): Array-like data carrying void mask values for analysis.
            min_radius_px (float): Numeric control parameter for min radius px during computation.
            min_peak_px (float): Value for min peak px interpreted by this routine.
            peak_footprint (int): Value for peak footprint interpreted by this routine.
            max_pores (int | None): Value for max pores interpreted by this routine.
        
        Returns:
            List[dict]: List of sample inscribed circles results produced for downstream iteration."""
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
                r0 = float(_bilinear(dt, float(y), float(x)))
                if r0 <= 0:
                    continue

                ny, nx = float(normals_yx[i, 0]), float(normals_yx[i, 1])
                norm = float(np.hypot(ny, nx))
                if norm == 0.0 or not np.isfinite(norm):
                    continue
                ny /= norm
                nx /= norm

                angle_deg = _canon_angle(nx, -ny)

                x1 = x - nx * (r0 / 2.0)
                y1 = y - ny * (r0 / 2.0)
                x2 = x + nx * (r0 / 2.0)
                y2 = y + ny * (r0 / 2.0)
                v1 = float(_bilinear(dt, float(y1), float(x1)))
                v2 = float(_bilinear(dt, float(y2), float(x2)))
                r = float((v1 + v2 + r0) / 2.0)
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

    def analyze(self, constant_segments: Optional[List[np.ndarray]] = None, spacing_px: float = 5.0):
        """
        Run thickness/orientation sampling on current centerline segments.

        Returns:
          - measurement_samples: (y, x, ny, nx, r, angle_deg) in original/display scale
          - drawing_samples: (y, x, ny, nx, r, angle_deg) in processing-image scale

        Note:
            `_xy` suffix denotes (x, y) ordering, while `_yx` denotes (y, x).
        """
        if constant_segments is not None:
            constant_segments_xy = constant_segments
        elif self._fiber_fragments:
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
    def _split_and_trim_contour(
        contour: np.ndarray,
        junction_points: List[Tuple[int, int]],
        trim: int = 3,
        tol: float = 3.0,
    ) -> List[np.ndarray]:
        """Provide an internal helper for split and trim contour.
        
        Args:
            contour (np.ndarray): Array-like data carrying contour values for analysis.
            junction_points (List[Tuple[int, int]]): Value for junction points interpreted by this routine.
            trim (int): Value for trim interpreted by this routine.
            tol (float): Value for tol interpreted by this routine.
        
        Returns:
            List[np.ndarray]: Array containing the split and trim contour output aligned with downstream expectations."""
        pts_xy = contour.reshape(-1, 2)
        split_indices: List[int] = []
        for (jx, jy) in junction_points:
            dists = np.sqrt((pts_xy[:, 0] - jx) ** 2 + (pts_xy[:, 1] - jy) ** 2)
            idxs = np.where(dists < tol)[0]
            if idxs.size > 0:
                min_idx = idxs[np.argmin(dists[idxs])]
                split_indices.append(int(min_idx))
        split_indices = sorted(set(split_indices))

        segments: List[np.ndarray] = []
        boundaries = [0] + split_indices + [len(pts_xy) - 1]
        for i in range(len(boundaries) - 1):
            seg_xy = pts_xy[boundaries[i] : boundaries[i + 1] + 1]
            if len(seg_xy) > 2 * trim:
                seg_xy = seg_xy[trim:-trim]
            else:
                continue
            if len(seg_xy) >= 2:
                segments.append(seg_xy.reshape(-1, 1, 2))
        return segments

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
        """Provide an internal helper for sliding window filter.
        
        Args:
            seg_xy (np.ndarray): Value for seg xy interpreted by this routine.
            image (np.ndarray): Array-like data carrying image values for analysis.
            dt (np.ndarray | None): Value for dt interpreted by this routine.
            grad_x (np.ndarray | None): Value for grad x interpreted by this routine.
            grad_y (np.ndarray | None): Value for grad y interpreted by this routine.
            win_len (float): Value for win len interpreted by this routine.
            std_thresh (float): Value for std thresh interpreted by this routine.
            max_angle (float): Numeric control parameter for max angle during computation.
            strict (bool): Boolean flag that toggles strict behavior.
            grad_alignment_cos (float): Value for grad alignment cos interpreted by this routine.
            radius_rel_tol (float): Numeric control parameter for radius rel tol during computation.
            min_valid_fraction (float): Value for min valid fraction interpreted by this routine.
        
        Returns:
            List[np.ndarray]: Array containing the sliding window filter output aligned with downstream expectations.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
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
            """Orchestrate sample intensity within the current module workflow.
            
            Args:
                p_xy (np.ndarray): Value for p xy interpreted by this routine.
            
            Returns:
                float: Result object produced by the sample intensity routine."""
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

    @staticmethod
    def _polyline_normals(
        pts_xy: np.ndarray,
        window: int = 8,
        angle_thresh_deg: float = 60.0,
        side: str = "left",
    ) -> np.ndarray:
        """Provide an internal helper for polyline normals.
        
        Args:
            pts_xy (np.ndarray): Value for pts xy interpreted by this routine.
            window (int): Value for window interpreted by this routine.
            angle_thresh_deg (float): Numeric control parameter for angle thresh deg during computation.
            side (str): Value for side interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the polyline normals output aligned with downstream expectations.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
        n = len(pts_xy)
        if n < 2:
            raise ValueError("Need at least 2 points for an open polyline.")

        eps = 1e-12
        seg = pts_xy[1:] - pts_xy[:-1]
        seg_norm = np.linalg.norm(seg, axis=1, keepdims=True) + eps
        u = seg / seg_norm

        def rot90(v: np.ndarray) -> np.ndarray:
            """Orchestrate rot90 within the current module workflow.
            
            Args:
                v (np.ndarray): Value for v interpreted by this routine.
            
            Returns:
                np.ndarray: Array containing the rot90 output aligned with downstream expectations."""
            return np.stack([-v[:, 1], v[:, 0]], axis=1)

        def unit(v: np.ndarray) -> np.ndarray:
            """Orchestrate unit within the current module workflow.
            
            Args:
                v (np.ndarray): Value for v interpreted by this routine.
            
            Returns:
                np.ndarray: Array containing the unit output aligned with downstream expectations."""
            nrm = np.linalg.norm(v, axis=1, keepdims=True) + eps
            return v / nrm

        angle_thresh = np.deg2rad(angle_thresh_deg)
        normals = np.zeros((n, 2), dtype=np.float32)

        t0 = u[0:1]
        tn = u[-1:]
        normals[0] = rot90(t0)[0]
        normals[-1] = rot90(tn)[0]

        for i in range(1, n - 1):
            d_prev = u[i - 1 : i]
            d_next = u[i : i + 1]
            cosang = np.clip(np.sum(d_prev * d_next, axis=1), -1.0, 1.0)[0]
            ang = np.arccos(cosang)

            if ang >= angle_thresh:
                t = unit(d_prev + d_next)
                if np.linalg.norm(t) < 1e-6:
                    t = d_next
            else:
                seg_lo = max(0, i - window)
                seg_hi = min(n - 2, i + window - 1)
                if seg_lo > seg_hi:
                    seg_lo, seg_hi = i - 1, i
                inds = np.arange(seg_lo, seg_hi + 1)
                center = i - 0.5
                dist = np.abs(inds + 0.5 - center)
                weights = (window + 1) - dist
                weights = np.clip(weights, 1.0, None)
                weights = (weights / (weights.sum() + eps)).reshape(-1, 1)
                t = unit((u[inds] * weights).sum(axis=0, keepdims=True))

            normals[i] = rot90(t)[0]

        if side == "right":
            normals = -normals
        return normals.astype(np.float32)
