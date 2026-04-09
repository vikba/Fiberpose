from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from skimage.segmentation import watershed, find_boundaries

from .clustering import stitch_and_cluster
from .datastructures import FiberFragment
from .models import FiberposeModel
from .ridge_detector import RidgeDetector
from .segmentation_fastest_ascend import segment_fastest_ascend
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
      • segment(mode, params) dispatches watershed/ridge/fastest-ascend segmentation
      • analysis helpers operate on cached centerlines and gradients
    """

    def __init__(
        self,
        ckpt_path: str,
        device: Any,
        model_kwargs: Optional[Dict[str, Any]] = None,
        strict: bool | None = None,
    ):
        self.ckpt_path = ckpt_path
        self.device = device

        defaults: Dict[str, Any] = {
            "nbase": "1,32,64,128,256",
            "nout": 3,
            "sz": 3,
            "dilation": 1,
            "pool_kernel": 2,
            "scale": 2,
            "residual_on": True,
            "style_on": True,
            "concat_style": False,
            "checkpointing": False,
            "dropout": False,
            "cpu": False,
            "amp": True,
            "fp16": True,
        }
        if model_kwargs:
            defaults.update(model_kwargs)
        defaults["device"] = device
        self._model_kwargs = defaults

        self._predictor: Optional[FiberposeModel] = None
        self._strict_centerlines = bool(strict) if strict is not None else False

        self._centerline_kwargs: Dict[str, Any] = self._default_centerline_kwargs()
        self._last_max_angle: float = 0.0
        self._constant_segments_info: Optional[List[Dict[str, Any]]] = None
        self._segment_counter: int = 0
        self._cluster_fibers: bool = False
        self._cluster_connection_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        self._cluster_config: Dict[str, Any] = {
            "d_max": 150.0,
            "threshold": 10.0,
            "max_neighbors": None,
            "length_weight": 0.1,
            "radius_weight": 0.0,
            "skip_endpoint_px": 5.0,
        }

        self.scale_factor: float = 1.0
        self.reset_prediction_state()

    # ------------------------------------------------------------------ #
    # Internal helpers / state

    def dt_predicted(self):
        if self._dt is not None:
            return True
        return False

    def reset_prediction_state(self):
        self._dt: Optional[np.ndarray] = None
        self._dt_u8: Optional[np.ndarray] = None
        self._dt_norm: Optional[np.ndarray] = None
        self._gx: Optional[np.ndarray] = None
        self._gy: Optional[np.ndarray] = None
        self._laplacian: Optional[np.ndarray] = None
        self._constant_segments: Optional[List[FiberFragment]] = None
        self._normals_xy: Optional[List[np.ndarray]] = None
        self._centerlines_rgb: Optional[np.ndarray] = None
        self._markers: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._constant_segments_info = None
        self._segment_counter = 0
        self._cluster_connection_pairs = []
        self._style_vec: Optional[np.ndarray] = None

    @staticmethod
    def _default_centerline_kwargs() -> Dict[str, Any]:
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
        self._strict_centerlines = bool(strict)

    def enable_fiber_clustering(self, enabled: bool, **cluster_kwargs: Any) -> None:
        """Toggle clustering of centerline fragments before segmentation."""
        self._cluster_fibers = bool(enabled)
        if cluster_kwargs:
            self._cluster_config.update(cluster_kwargs)
        # force downstream consumers to refresh markers if clustering state changes
        self._markers = None

    def get_cluster_config(self) -> Dict[str, Any]:
        """Return a shallow copy of the current clustering configuration."""
        return dict(self._cluster_config)

    def set_device(self, device: Any):
        if device == self.device:
            return
        self.device = device
        self._model_kwargs["device"] = device
        self._predictor = None  # force re-init on next predict
        logger.info("FiberAnalyzer_2D device set to %s", device)

    def set_checkpoint_path(self, ckpt_path: str):
        """Update model checkpoint and reset predictor/predictions."""
        if ckpt_path == self.ckpt_path:
            return
        self.ckpt_path = ckpt_path
        self._predictor = None
        self.reset_prediction_state()
        logger.info("FiberAnalyzer_2D checkpoint switched to %s", ckpt_path)

    def _ensure_predictor(self) -> FiberposeModel:
        if self._predictor is not None:
            return self._predictor
        kwargs = dict(self._model_kwargs)
        kwargs["device"] = self.device
        self._predictor = FiberposeModel(
            ckpt_path=self.ckpt_path,
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

    def _run_model(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:

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
        dt_pad, style_vec = predictor.predict(image_pad, return_style=True)
        dt = dt_pad[:H, :W] if dt_pad.ndim == 2 else dt_pad[:H, :W, 0]
        return np.asarray(dt, dtype=np.float32), style_vec

    def _set_dt_and_gradients(self, dt: np.ndarray, gx: np.ndarray, gy: np.ndarray):
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

        self.reset_prediction_state()

        dt, style_vec = self._run_model(image)
        dt_smooth = gaussian_filter(dt, sigma=float(smooth_sigma))
        gy, gx = np.gradient(dt_smooth)
        gy = gaussian_filter(gy, sigma=float(grad_sigma))
        gx = gaussian_filter(gx, sigma=float(grad_sigma))
        lapl_img = laplace(dt_smooth)

        self._set_dt_and_gradients(dt, gx, gy)
        self._laplacian = lapl_img
        self._style_vec = style_vec

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
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Stateful wrapper around the stateless `compute_centerlines(...)` core.

        Uses cached DT generated by `predict_dt(...)` and updates internal caches:
        `_constant_segments`, `_normals_xy`, `_centerlines_rgb`, `_markers`, etc.
        """
        if self._dt_u8 is None or self._dt is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")

        fragments, markers = self.compute_centerlines(
            dt_u8=self._dt_u8,
            dt=self._dt,
            gx=self._gx,
            gy=self._gy,
            max_angle_deg=max_angle_deg,
            sigma=sigma,
            low=low,
            high=high,
            polarity=polarity,
            trim=trim,
            tol=tol,
            win_len=win_len,
            std_thresh=std_thresh,
            strict=self._strict_centerlines,
            segment_id_start=self._segment_counter,
        )

        self._segment_counter = int(self._segment_counter + len(fragments))
        constant_segments_xy: List[np.ndarray] = [seg.points_xy.reshape(-1, 1, 2) for seg in fragments]
        normals_xy = [self._polyline_normals(seg.points_xy) for seg in fragments]

        cluster_connections: List[Tuple[np.ndarray, np.ndarray]] = []
        if fragments and self._cluster_fibers:
            cluster_connections = self._cluster_segments(
                segments=fragments,
                cluster_config=self._cluster_config,
            )
        self._normalize_parent_ids(fragments)
        centerlines_rgb = self._render_centerlines_rgb(
            dt_u8=self._dt_u8,
            segments=fragments,
            cluster_connections=cluster_connections,
        )

        self._constant_segments = fragments
        self._constant_segments_info = [
            {
                "segment_id": seg.fragment_id,
                "parent_fiber_id": seg.parent_fiber_id,
                "points_xy": seg.points_xy,
                "radii": seg.radii,
            }
            for seg in fragments
        ]
        self._normals_xy = normals_xy
        self._centerlines_rgb = centerlines_rgb
        self._markers = markers
        self._cluster_connection_pairs = cluster_connections

        logger.info("Detected constant segments: %d", len(constant_segments_xy))
        return constant_segments_xy, normals_xy, centerlines_rgb, markers

    @classmethod
    def compute_centerlines(
        cls,
        dt_u8: np.ndarray,
        dt: np.ndarray,
        gx: np.ndarray | None,
        gy: np.ndarray | None,
        *,
        max_angle_deg: float,
        sigma: float = 1.0,
        low: float = 0.1,
        high: float = 0.4,
        polarity: str = "light",
        trim: int = 3,
        tol: float = 3.0,
        win_len: float = 25.0,
        std_thresh: float = 0.8,
        strict: bool = False,
        segment_id_start: int = 0,
        markers_from_raw_ridges: bool = False,
    ) -> Tuple[List[FiberFragment], np.ndarray]:
        """
        Stateless centerline extraction core.

        Returns:
            fragments: list[FiberFragment] (filtered windows)
            markers: int32 marker image for watershed seeding.
                     If `markers_from_raw_ridges` is True, markers come directly from ridge
                     detector polylines; otherwise, they use the filtered fragments (default).
        """
        if dt_u8 is None or dt is None:
            raise ValueError("dt_u8 and dt are required.")
        if strict and (gx is None or gy is None):
            raise ValueError("Strict centerline filtering requires gx and gy.")

        lines, junctions = RidgeDetector.detect_ridges(
            image2d=dt_u8,
            sigma=sigma,
            low=low,
            high=high,
            polarity=polarity,
        )

        # After ridge detection: convert ridge polylines to contours, split at junctions,
        # trim noisy ends, then slide a window along each segment to filter by orientation
        # consistency (and optionally gradient alignment). Accepted windows become fragments.
        contours = [np.stack((line.pts[:, 1], line.pts[:, 0]), axis=1).reshape(-1, 1, 2) for line in lines]
        junction_points = [(int(round(float(j.pt[1]))), int(round(float(j.pt[0])))) for j in junctions]

        processed_segments: List[Tuple[np.ndarray, int]] = []
        for source_idx, contour in enumerate(contours):
            segs_xy = cls._split_and_trim_contour(contour, junction_points, trim=trim, tol=tol)
            if segs_xy:
                processed_segments.extend((seg, source_idx) for seg in segs_xy)
            else:
                pts_xy = contour.reshape(-1, 2)
                if len(pts_xy) > 2 * trim:
                    trimmed_xy = pts_xy[trim:-trim].reshape(-1, 1, 2)
                    processed_segments.append((trimmed_xy, source_idx))

        fragments: List[FiberFragment] = []
        segment_counter = int(segment_id_start)
        for seg_xy, source_id in processed_segments:
            filtered_windows = cls._sliding_window_filter(
                seg_xy,
                image=dt_u8,
                dt=dt,
                grad_x=gx,
                grad_y=gy,
                win_len=win_len,
                std_thresh=std_thresh,
                max_angle=max_angle_deg,
                strict=strict,
            )
            for window_xy in filtered_windows:
                pts_xy = window_xy.reshape(-1, 2).astype(np.float32)
                if pts_xy.size == 0:
                    continue
                # Store per-point DT radii so strict ridge segmentation can
                # threshold and dilate centerline runs by local thickness.
                x_idx = np.clip(np.rint(pts_xy[:, 0]).astype(np.int32), 0, dt.shape[1] - 1)
                y_idx = np.clip(np.rint(pts_xy[:, 1]).astype(np.int32), 0, dt.shape[0] - 1)
                radii = np.asarray(dt[y_idx, x_idx], dtype=np.float32)
                segment_counter += 1
                fragments.append(
                    FiberFragment(
                        fragment_id=segment_counter,
                        centerline_xy=pts_xy,
                        radii=radii,
                        parent_fiber_id=source_id + 1,
                    )
                )

        markers = (
            cls._render_markers_from_ridge_lines(dt_u8.shape, lines)
            if markers_from_raw_ridges
            else (cls._render_markers(dt_u8.shape, fragments) if fragments else np.zeros(dt_u8.shape, dtype=np.int32))
        )
        return fragments, markers

    @staticmethod
    def _render_centerlines_rgb(
        dt_u8: np.ndarray,
        segments: List[FiberFragment],
        cluster_connections: List[Tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> np.ndarray:
        base_dt = to_uint8_for_view(dt_u8)
        centerlines_rgb = cv2.cvtColor(base_dt, cv2.COLOR_GRAY2BGR)

        centerline_color = (255, 0, 0)
        for seg in segments:
            pts = seg.points_xy.reshape(-1, 1, 2).astype(np.int32)
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

        if cluster_connections:
            connection_color = (0, 255, 0)
            for p_xy, q_xy in cluster_connections:
                px = int(round(float(p_xy[0])))
                py = int(round(float(p_xy[1])))
                qx = int(round(float(q_xy[0])))
                qy = int(round(float(q_xy[1])))
                cv2.line(centerlines_rgb, (px, py), (qx, qy), connection_color, 1, cv2.LINE_AA)

        return centerlines_rgb

    def segment(
        self,
        mode: str = "watershed",
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Dispatch segmentation by mode.

        Supported modes:
            - watershed: params = {"threshold_u8": int}
            - ridge: params = {"radius_thresh_px": float, "dilate_extra_px": float, "min_run_points": int}
            - fastest_ascend: params = {"sigma": float, "dist_thresh": float, "flux_merge_thresh": float}
        """
        if self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")

        opts = dict(params or {})
        mode_name = str(mode or "watershed").strip().lower()

        if mode_name == "watershed":
            thr_u8 = int(opts.get("threshold_u8", 128))
            if self._markers is None or self._markers.max() == 0:
                self.process_centerlines(self._last_max_angle, **self._centerline_kwargs)
            return self.segment_watershed(thr_u8=thr_u8)

        if mode_name == "ridge":
            radius_thresh_px = opts.get("radius_thresh_px")
            if radius_thresh_px is None:
                raise ValueError("Ridge segmentation requires radius_thresh_px.")
            return self.segment_strict_ridge(
                radius_thresh_px=float(radius_thresh_px),
                dilate_extra_px=float(opts.get("dilate_extra_px", 0.0)),
                min_run_points=int(opts.get("min_run_points", 1)),
            )

        if mode_name == "fastest_ascend":
            return self.segment_fastest_ascend(
                sigma=float(opts.get("sigma", 1.0)),
                dist_thresh=float(opts.get("dist_thresh", 0.2)),
                flux_merge_thresh=float(opts.get("flux_merge_thresh", 5.0)),
            )

        raise ValueError(f"Unsupported segmentation mode: {mode}")

    def segment_watershed(self, thr_u8: int) -> np.ndarray:
        """
        Run watershed segmentation using cached DT and markers.
        """

        if self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")
        if self._markers is None or self._markers.max() == 0:
            return np.zeros_like(self._dt_u8, dtype=np.uint16)

        logger.info("Running watershed segmentation")

        labels = self.watershed_dt_segmentation(self._dt_u8, self._markers, thr_u8=thr_u8)
        self._update_centerline_overlay_with_watershed(self._markers, labels)
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

    def segment_strict_ridge(
        self,
        radius_thresh_px: float,
        dilate_extra_px: float = 0.0,
        min_run_points: int = 1,
    ) -> np.ndarray:
        """
        Strict ridge segmentation: expand qualified centerline runs using DT radii.
        """
        if self._dt is None or self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")

        if not self._constant_segments_info:
            labels = np.zeros_like(self._dt_u8, dtype=np.uint16)
            self._labels = labels
            return labels

        mask = np.zeros(self._dt_u8.shape, dtype=np.int32)
        radius_thresh_px = float(radius_thresh_px)
        extra = float(dilate_extra_px)

        for info in self._constant_segments_info:
            pts_xy = info.get("points_xy")
            radii = info.get("radii")
            if pts_xy is None or radii is None:
                continue
            pts_xy = np.asarray(pts_xy, dtype=np.float32)
            radii = np.asarray(radii, dtype=np.float32)
            if pts_xy.shape[0] == 0 or radii.shape[0] != pts_xy.shape[0]:
                continue

            keep_mask = radii >= radius_thresh_px
            if not np.any(keep_mask):
                continue

            run_start = None
            for idx, keep in enumerate(keep_mask):
                if keep and run_start is None:
                    run_start = idx
                elif (not keep) and (run_start is not None):
                    if idx - run_start >= max(1, int(min_run_points)):
                        self._draw_centerline_run(
                            mask,
                            pts_xy[run_start:idx],
                            radii[run_start:idx],
                            extra,
                            value=int(info.get("parent_fiber_id", info.get("segment_id", 0))),
                        )
                    run_start = None
            if run_start is not None:
                if len(pts_xy) - run_start >= max(1, int(min_run_points)):
                    self._draw_centerline_run(
                        mask,
                        pts_xy[run_start:],
                        radii[run_start:],
                        extra,
                        value=int(info.get("parent_fiber_id", info.get("segment_id", 0))),
                    )

        if not mask.any():
            labels = np.zeros_like(self._dt_u8, dtype=np.uint16)
            self._labels = labels
            return labels

        labels = mask.astype(np.uint16, copy=False)
        self._labels = labels
        return labels

    def segment_fastest_ascend(
        self,
        sigma: float = 1.0,
        dist_thresh: float = 0.2,
        flux_merge_thresh: float = 5.0,
    ) -> np.ndarray:
        """
        Segment using steepest-ascent flow over the predicted DT.
        """
        if self._dt_u8 is None:
            raise ValueError("Distance transform not available. Call predict_dt first.")

        labels_i32 = segment_fastest_ascend(
            self._dt_u8,
            sigma=float(sigma),
            dist_thresh=float(dist_thresh),
            flux_merge_thresh=float(flux_merge_thresh),
        )
        labels_i32 = np.asarray(labels_i32, dtype=np.int32, copy=False)
        max_uint16 = np.iinfo(np.uint16).max
        max_label = int(labels_i32.max()) if labels_i32.size else 0
        if max_label > max_uint16:
            logger.warning(
                "Fastest-ascend produced %d labels; clipping to uint16 max=%d",
                max_label,
                max_uint16,
            )
        labels = np.clip(labels_i32, 0, max_uint16).astype(np.uint16, copy=False)
        self._labels = labels
        return labels

    @staticmethod
    def _draw_centerline_run(
        mask: np.ndarray,
        pts_xy: np.ndarray,
        radii: np.ndarray,
        extra_radius_px: float = 0.0,
        value: int = 255,
    ) -> None:
        """
        Paint a contiguous run of centerline samples onto a binary mask using their DT radii.
        """
        if pts_xy.size == 0:
            return
        h, w = mask.shape[:2]
        extra_radius_px = float(extra_radius_px)

        for pt_xy, radius in zip(pts_xy, radii):
            r_eff = float(radius) + extra_radius_px
            if r_eff <= 0.0:
                continue
            cx = int(round(float(pt_xy[0])))
            cy = int(round(float(pt_xy[1])))
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            radius_px = max(1, int(round(r_eff)))
            cv2.circle(mask, (cx, cy), radius_px, int(value), thickness=-1)

    # ------------------------------------------------------------------ #
    # Clustering helpers

    @staticmethod
    def _cluster_segments(
        segments: List[FiberFragment],
        cluster_config: Dict[str, Any],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not segments:
            return []

        try:
            stitch_and_cluster(segments, **cluster_config)
        except Exception as exc:
            logger.warning("Fiber clustering failed: %s", exc)
            return []

        connections: List[Tuple[np.ndarray, np.ndarray]] = []
        seen_pairs: set[tuple[int, int]] = set()
        skip_px = float(cluster_config.get("skip_endpoint_px", 0.0))
        for frag in segments:
            for ep in frag.endpoints():
                other = ep.connected_to
                if other is None:
                    continue
                fid = frag.fragment_id
                oid = other.fragment.fragment_id
                key = (min(fid, oid), max(fid, oid))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                connections.append((ep.shifted_pos(skip_px).copy(), other.shifted_pos(skip_px).copy()))
        return connections

    @staticmethod
    def _normalize_parent_ids(segments: List[FiberFragment]) -> Dict[int, int]:
        remap: Dict[int, int] = {}
        next_id = 1
        for seg in segments:
            pid = int(seg.parent_fiber_id)
            if pid <= 0:
                pid = int(seg.fragment_id)
            if pid not in remap:
                remap[pid] = next_id
                next_id += 1
            seg.parent_fiber_id = remap[pid]
        return remap

    @staticmethod
    def _render_markers(shape: Tuple[int, int], segments: List[FiberFragment]) -> np.ndarray:
        markers = np.zeros(shape, dtype=np.int32)
        for seg in segments:
            pts = np.asarray(seg.points_xy, dtype=np.float32)
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

    @staticmethod
    def _render_markers_from_ridge_lines(shape: Tuple[int, int], lines: List[Any]) -> np.ndarray:
        """
        Render markers directly from RidgeDetector polylines (y,x ordering).
        """
        markers = np.zeros(shape, dtype=np.int32)
        for idx, line in enumerate(lines, start=1):
            pts = np.asarray(line.pts, dtype=np.float32)
            if pts.size == 0:
                continue
            pts_xy = pts[:, [1, 0]]
            pts_int = np.round(pts_xy).astype(np.int32).reshape(-1, 1, 2)
            if len(pts_int) < 2:
                cx, cy = pts_int[0, 0]
                cv2.circle(markers, (cx, cy), 1, idx, thickness=-1)
                continue
            cv2.polylines(
                markers,
                [pts_int],
                isClosed=False,
                color=idx,
                thickness=1,
                lineType=cv2.LINE_8,
            )
        return markers

    # ------------------------------------------------------------------ #
    # Accessors

    def get_dt(self) -> Optional[np.ndarray]:
        return self._dt

    def get_dt_u8(self) -> Optional[np.ndarray]:
        return self._dt_u8

    def get_dt_norm(self) -> Optional[np.ndarray]:
        return self._dt_norm

    def get_style_vector(self) -> Optional[np.ndarray]:
        return self._style_vec

    def get_gradients(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._gx, self._gy

    def get_laplacian(self) -> Optional[np.ndarray]:
        return self._laplacian

    def get_centerlines_image(self) -> Optional[np.ndarray]:
        return self._centerlines_rgb

    def get_cluster_connections(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return copies of clustered connection endpoints (x,y) pairs, if any."""
        if not self._cluster_connection_pairs:
            return []
        return [(p.copy(), q.copy()) for p, q in self._cluster_connection_pairs]

    def get_centerline_segments(self) -> List[np.ndarray]:
        if not self._constant_segments:
            return []
        return [seg.points_xy.reshape(-1, 1, 2) for seg in self._constant_segments]

    def get_centerline_normals(self) -> List[np.ndarray]:
        return self._normals_xy or []

    def get_markers(self) -> Optional[np.ndarray]:
        return self._markers

    def get_labels(self) -> Optional[np.ndarray]:
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
        self._centerlines_rgb = base

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
        List[tuple[int, int, float]],
        List[float],
        List[tuple[int, int, float, float, float, float, float, float]],
    ]:
        """
        Pure analysis core with explicit inputs.

        Returns:
          - radius_samples: list of (y, x, r)
          - angles_samples_deg: list[float]
          - overlay_samples: list of (y, x, r, ny, nx, ty, tx, angle_deg)
        """
        if dt is None or gx is None or gy is None:
            return [], [], []

        segments = constant_segments_xy or []
        # Keep sampling stable: never allow sub-pixel step below 1 px.
        spacing = max(1.0, float(spacing_px))
        # `scale_factor` maps processing resolution back to display/original resolution.
        scale = float(scale_factor)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        radius_samples: List[tuple[int, int, float]] = []
        angles_samples_deg: List[float] = []
        overlay_samples: List[tuple[int, int, float, float, float, float, float, float]] = []

        def _canon_angle(ty: float, tx: float):
            # Convert tangent direction to canonical orientation angle in image coords:
            # - use clockwise-positive convention via leading minus
            # - fold 180-degree equivalent directions into (-90, 90]
            n = float(np.hypot(ty, tx))
            if n == 0.0 or not np.isfinite(n):
                return None
            ty, tx = ty / n, tx / n
            ang = -np.degrees(np.arctan2(ty, tx))
            if ang > 90.0:
                ang -= 180.0
                ty, tx = -ty, -tx
            elif ang <= -90.0:
                ang += 180.0
                ty, tx = -ty, -tx
            if ang <= -90.0:
                ang = -89.999
            return ty, tx, ang

        for seg_xy in segments:
            seg_xy = np.asarray(seg_xy).reshape(-1, 2).astype(np.float32)
            if len(seg_xy) < 2:
                continue
            # Uniform arc-length sampling along each centerline segment.
            samples_xy = resample_polyline_xy(seg_xy, step=spacing)
            if len(samples_xy) < 2:
                continue
            samples_yx = samples_xy[:, [1, 0]].astype(np.float32)
            # Estimate local normal from DT gradients; tangent is the perpendicular.
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
                #approximate radius at the ridge center
                r0 = float(_bilinear(dt, float(y), float(x)))
                if r0 <= 0:
                    continue
                ny, nx = float(normals_yx[i, 0]), float(normals_yx[i, 1])
                ty, tx = nx, -ny
                canon = _canon_angle(ty, tx)
                if canon is None:
                    ty_c, tx_c, ang = float("nan"), float("nan"), float("nan")
                else:
                    ty_c, tx_c, ang = canon
                angles_samples_deg.append(ang)

                norm = np.sqrt(nx**2 + ny**2)
                nx /= norm
                ny /= norm

                x1 = x - nx*(r0/2)
                y1 = y - ny*(r0/2)

                x2 = x + nx*(r0/2)
                y2 = y + ny*(r0/2) 

                v1 = float(_bilinear(dt, float(y1), float(x1)))
                v2 = float(_bilinear(dt, float(y2), float(x2)))

                r = (v1 + v2 + r0) / 2

                # Keep per-sample geometry for overlays/diagnostics.
                overlay_samples.append((int(round(y)), int(round(x)), r, ny, nx, ty_c, tx_c, float(ang)))
                # Export histogram samples in original scale.
                radius_samples.append(
                    (
                        int(round(y / scale)),
                        int(round(x / scale)),
                        r / scale,
                    )
                )

        return radius_samples, angles_samples_deg, overlay_samples

    def analyze(self, constant_segments: Optional[List[np.ndarray]] = None, spacing_px: float = 5.0):
        """
        Backward-compatible wrapper over `analyze_from_signatures(...)`.

        Returns:
          - radius_samples: list of (y, x, r)
          - angles_samples_deg: list[float]
          - overlay_samples: list of (y, x, r, ny, nx, ty, tx, angle_deg)

        Note:
            `_xy` suffix denotes (x, y) ordering, while `_yx` denotes (y, x).
        """
        if constant_segments is not None:
            constant_segments_xy = constant_segments
        elif self._constant_segments:
            constant_segments_xy = [seg.points_xy.reshape(-1, 1, 2) for seg in self._constant_segments]
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

    @staticmethod
    def _polyline_normals(
        pts_xy: np.ndarray,
        window: int = 8,
        angle_thresh_deg: float = 60.0,
        side: str = "left",
    ) -> np.ndarray:
        pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
        n = len(pts_xy)
        if n < 2:
            raise ValueError("Need at least 2 points for an open polyline.")

        eps = 1e-12
        seg = pts_xy[1:] - pts_xy[:-1]
        seg_norm = np.linalg.norm(seg, axis=1, keepdims=True) + eps
        u = seg / seg_norm

        def rot90(v: np.ndarray) -> np.ndarray:
            return np.stack([-v[:, 1], v[:, 0]], axis=1)

        def unit(v: np.ndarray) -> np.ndarray:
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
