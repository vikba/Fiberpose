"""Pipeline orchestration for frame and batch processing."""

from __future__ import annotations

# ---- merged from frame_pipeline_service.py ----


import json
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .domain import AppConfig, Frame, SampleProperties
from .fiber_analyzer_2D import FiberAnalyzer_2D
from .utils import ensure_rgb_u8, labels_to_color


@dataclass
class PipelineOutcome:
    """Result envelope for pipeline stage execution."""

    ok: bool
    error_title: str | None = None
    error_message: str | None = None
    info_title: str | None = None
    info_message: str | None = None


class PipelineController:
    """Encapsulates prediction->segmentation->measurement orchestration for one frame."""

    def __init__(self, analyzer: FiberAnalyzer_2D | None = None):
        self.analyzer = analyzer

    def set_analyzer(self, analyzer: FiberAnalyzer_2D | None) -> None:
        self.analyzer = analyzer

    @staticmethod
    def _effective_scale_factor(config: AppConfig) -> float:
        token = str(config.scale_label or "1x").strip().lower().replace("×", "x")
        if token.endswith("x"):
            token = token[:-1]
        try:
            scale = float(token)
        except Exception:
            scale = 1.0
        if not np.isfinite(scale) or scale <= 0:
            return 1.0
        return float(scale)

    def _processing_image(self, frame: Frame, config: AppConfig) -> np.ndarray | None:
        image = frame.orig_img
        if image is None:
            return None

        scale = self._effective_scale_factor(config)
        if math.isclose(scale, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            frame.scale_factor = 1.0
            frame.orig_img_scaled = None
            if self.analyzer is not None:
                self.analyzer.scale_factor = 1.0
            return image

        needs_rescale = (
            frame.orig_img_scaled is None
            or not math.isclose(float(frame.scale_factor), float(scale), rel_tol=1e-9, abs_tol=1e-9)
        )
        if needs_rescale:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            frame.orig_img_scaled = cv2.resize(
                image,
                None,
                fx=float(scale),
                fy=float(scale),
                interpolation=interp,
            )
        frame.scale_factor = float(scale)
        if self.analyzer is not None:
            self.analyzer.scale_factor = float(scale)
        return frame.orig_img_scaled

    @staticmethod
    def _image_for_inference(image: np.ndarray | None, invert_input: bool) -> np.ndarray | None:
        if image is None or not bool(invert_input):
            return image
        try:
            return cv2.bitwise_not(image)
        except Exception:
            return np.max(image) - image

    @staticmethod
    def _create_overlay(
        base_img: np.ndarray | None,
        labels_img: np.ndarray | None,
        alpha: float = 0.5,
        labels_rgb: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if base_img is None or labels_img is None:
            return None
        base_rgb = ensure_rgb_u8(base_img)
        if labels_rgb is None:
            labels_rgb = labels_to_color(labels_img)
        if base_rgb is None or labels_rgb is None:
            return None
        if base_rgb.shape[:2] != labels_rgb.shape[:2]:
            return None
        out = base_rgb.astype(np.float32, copy=True)
        over = labels_rgb.astype(np.float32, copy=False)
        mask = labels_img > 0
        if mask.any():
            out[mask] = out[mask] * (1.0 - alpha) + over[mask] * alpha
        return out.clip(0, 255).astype(np.uint8)

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
        from skimage.segmentation import find_boundaries

        out = base_rgb.copy()
        try:
            boundaries = find_boundaries(labels, mode="outer")
        except Exception:
            boundaries = None
        if boundaries is not None and boundaries.any():
            out[boundaries] = (255, 0, 0)
        return out

    @staticmethod
    def _configured_centerline_kwargs(config: AppConfig) -> dict[str, Any]:
        kwargs = dict(FiberAnalyzer_2D._default_centerline_kwargs())
        kwargs["sigma"] = float(config.ridge_sigma)
        kwargs["low"] = float(config.ridge_low)
        kwargs["high"] = float(config.ridge_high)
        return kwargs

    @classmethod
    def centerline_params(cls, config: AppConfig) -> dict[str, Any]:
        return {
            "max_angle_deg": float(config.max_angle_deg),
            "centerline_kwargs": cls._configured_centerline_kwargs(config),
            "strict_centerlines": bool(config.strict_centerlines),
            "keep_segments_without_measurement_centerlines": bool(
                config.keep_segments_without_measurement_centerlines
            ),
            "merge_expansion_factor": float(config.merge_expansion_factor),
            "min_fragment_centerline_length_px": float(config.min_fragment_centerline_length_px),
            "measurement_centerline_length_trim_fraction": FiberAnalyzer_2D._clamp_trim_fraction(
                float(config.measurement_centerline_length_trim_fraction)
            ),
            "grad_quality_trim_fraction": FiberAnalyzer_2D._clamp_trim_fraction(
                float(config.grad_quality_trim_fraction)
            ),
            "segmentation_expansion_factor": float(config.segmentation_expansion_factor),
        }

    @staticmethod
    def measurement_params(config: AppConfig) -> dict[str, Any]:
        return {
            "sample_spacing": max(1.0, float(config.sample_spacing)),
            "seg_threshold": int(config.seg_threshold),
        }

    @classmethod
    def prediction_signature(cls, frame: Frame, config: AppConfig, *, invert_input: bool) -> str | None:
        image = frame.orig_img
        if image is None:
            return None
        payload = {
            "image_id": int(id(image)),
            "shape": tuple(int(v) for v in image.shape),
            "dtype": str(image.dtype),
            "model_path": str(config.model_path),
            "invert_input": bool(invert_input),
            "scale_factor": float(cls._effective_scale_factor(config)),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @classmethod
    def segmentation_signature(cls, frame: Frame, config: AppConfig, *, invert_input: bool) -> str | None:
        prediction_sig = cls.prediction_signature(frame, config, invert_input=invert_input)
        if prediction_sig is None:
            return None
        payload = {
            "prediction_sig": prediction_sig,
            "centerline_params": cls.centerline_params(config),
            "segmentation_params": cls._segmentation_params(config),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @classmethod
    def measurement_signature(
        cls,
        frame: Frame,
        config: AppConfig,
        sample_props: SampleProperties,
        *,
        invert_input: bool,
    ) -> str | None:
        segmentation_sig = cls.segmentation_signature(frame, config, invert_input=invert_input)
        if segmentation_sig is None:
            return None
        props = sample_props if sample_props is not None else SampleProperties.defaults()
        payload = {
            "segmentation_sig": segmentation_sig,
            "measurement_params": cls.measurement_params(config),
            "pixel_size_x": float(props.pixel_size_x),
            "pixel_size_y": float(props.pixel_size_y),
            "voxel_size_z": float(props.voxel_size_z),
            "unit": str(props.unit),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @staticmethod
    def invalidate_prediction(frame: Frame, analyzer: FiberAnalyzer_2D | None = None) -> None:
        if analyzer is not None:
            analyzer._reset_prediction_state()
        frame.clear_prediction()

    @staticmethod
    def invalidate_segmentation(frame: Frame) -> None:
        frame.clear_segmentation()

    @staticmethod
    def invalidate_measurements(frame: Frame) -> None:
        frame.clear_measurements()

    def _hydrate_analyzer_from_prediction(
        self,
        frame: Frame,
    ) -> None:
        if self.analyzer is None or frame.dt is None or frame.gx is None or frame.gy is None:
            return
        self.analyzer._reset_prediction_state()
        self.analyzer.scale_factor = float(frame.scale_factor)
        self.analyzer._set_dt_and_gradients(
            frame.dt,
            frame.gx,
            frame.gy,
            foreground_prob=frame.foreground_prob,
        )
        self.analyzer._laplacian = frame.lapl_img

    def predict(
        self,
        frame: Frame,
        config: AppConfig,
        *,
        invert_input: bool,
    ) -> dict[str, Any]:
        if self.analyzer is None:
            raise RuntimeError("Fiber analyzer has not been initialized.")
        processing_img = self._processing_image(frame, config)
        if processing_img is None:
            raise RuntimeError("Missing input image.")
        self.analyzer.predict_dt(
            image=self._image_for_inference(processing_img, invert_input=invert_input),
        )
        dt = self.analyzer.get_dt()
        gx, gy = self.analyzer.get_gradients()
        if dt is None or gx is None or gy is None:
            raise RuntimeError("Distance transform or gradients unavailable.")
        return {
            "dt": np.asarray(dt, dtype=np.float32, copy=False),
            "gx": np.asarray(gx, dtype=np.float32, copy=False),
            "gy": np.asarray(gy, dtype=np.float32, copy=False),
            "dt_u8": np.asarray(self.analyzer.get_dt_u8(), dtype=np.uint8, copy=False),
            "dt_normalized": np.asarray(self.analyzer.get_dt_norm(), dtype=np.float32, copy=False),
            "foreground_prob": (
                np.asarray(self.analyzer.get_foreground_prob(), dtype=np.float32, copy=False)
                if self.analyzer.get_foreground_prob() is not None
                else None
            ),
            "lapl_img": np.asarray(self.analyzer.get_laplacian(), dtype=np.float32, copy=False),
        }

    def build_centerlines(self, frame: Frame, config: AppConfig) -> dict[str, Any]:
        if self.analyzer is None:
            raise RuntimeError("Fiber analyzer has not been initialized.")
        params = self.centerline_params(config)
        self._hydrate_analyzer_from_prediction(frame)
        self.analyzer.set_strict_centerline_filter(bool(params["strict_centerlines"]))
        self.analyzer.set_keep_segments_without_measurement_centerlines(
            bool(params["keep_segments_without_measurement_centerlines"])
        )
        self.analyzer.process_centerlines(
            max_angle_deg=float(params["max_angle_deg"]),
            **dict(params["centerline_kwargs"]),
            merge_expansion_factor=float(params["merge_expansion_factor"]),
            segmentation_expansion_factor=float(params["segmentation_expansion_factor"]),
            min_fragment_centerline_length_px=float(params["min_fragment_centerline_length_px"]),
            measurement_centerline_length_trim_fraction=float(
                params["measurement_centerline_length_trim_fraction"]
            ),
            grad_quality_trim_fraction=float(params["grad_quality_trim_fraction"]),
        )
        return {
            "segmentation_centerlines_overlay_img": self.analyzer.get_segmentation_centerlines_image(),
            "measurement_centerlines_overlay_img": self.analyzer.get_measurement_centerlines_image(),
            "centerlines_comparison_overlay_img": self.analyzer.get_centerlines_comparison_image(),
        }

    @staticmethod
    def _segmentation_params(config: AppConfig) -> tuple[str, dict[str, Any]]:
        mode = str(config.segmentation_mode or "watershed")
        if mode not in ("watershed", "radial_expansion"):
            mode = "watershed"
        params = (
            {
                "gradient_threshold": float(config.watershed_grad_threshold),
                "min_segment_foreground_mean": float(config.min_segment_foreground_mean),
            }
            if mode == "watershed"
            else {
                "expansion_factor": float(config.segmentation_expansion_factor),
                "min_segment_foreground_mean": float(config.min_segment_foreground_mean),
            }
        )
        return mode, params

    def segment_stage(self, frame: Frame, config: AppConfig, centerlines: dict[str, Any]) -> dict[str, Any]:
        if self.analyzer is None:
            raise RuntimeError("Fiber analyzer has not been initialized.")
        mode, params = self._segmentation_params(config)
        labels = self.analyzer.segment(mode=str(mode), params=dict(params))
        base_img = frame.orig_img if frame.scale_factor == 1.0 or frame.orig_img_scaled is None else frame.orig_img_scaled
        if labels is not None:
            labels_rgb = labels_to_color(labels)
            labels_cache = (id(labels), labels_rgb)
            outline_cache = (id(labels), self._build_segmentation_outline_image(base_img, labels))
        else:
            labels_rgb = None
            labels_cache = (None, None)
            outline_cache = (None, None)
        return {
            "labels_2d": labels,
            "segmentation_centerlines_overlay_img": centerlines["segmentation_centerlines_overlay_img"],
            "measurement_centerlines_overlay_img": self.analyzer.get_measurement_centerlines_image(),
            "centerlines_comparison_overlay_img": self.analyzer.get_centerlines_comparison_image(),
            "overlay_img": self._create_overlay(base_img, labels, alpha=0.5, labels_rgb=labels_rgb),
            "labels_rgb_cache": labels_cache,
            "segmentation_outline_cache": outline_cache,
        }

    @staticmethod
    def _compute_orientation_stats(angles_samples_deg: np.ndarray) -> tuple[float, float, float]:
        if angles_samples_deg is None or getattr(angles_samples_deg, "size", 0) == 0:
            return float("nan"), float("nan"), float("nan")

        angles_rad = np.deg2rad(2.0 * angles_samples_deg)
        mean_cos = float(np.mean(np.cos(angles_rad)))
        mean_sin = float(np.mean(np.sin(angles_rad)))
        orientation_index = float(np.clip(np.hypot(mean_cos, mean_sin), 0.0, 1.0))

        mean_angle_rad = 0.5 * float(np.arctan2(mean_sin, mean_cos))
        mean_angle_deg = float(np.rad2deg(mean_angle_rad))
        if mean_angle_deg <= -90.0:
            mean_angle_deg += 180.0

        if orientation_index > 0 and np.isfinite(orientation_index):
            r = float(np.clip(orientation_index, 1e-8, 1.0))
            std_angle_rad = float(np.sqrt(max(0.0, -2.0 * np.log(r))))
            std_angle_deg = 0.5 * float(np.rad2deg(std_angle_rad))
        else:
            std_angle_deg = float("nan")

        return orientation_index, mean_angle_deg, std_angle_deg

    def _build_measurement_outputs(
        self,
        *,
        measurement_samples: list[Any],
        pores: list[dict[str, Any]],
        porosity: float,
        pixel_size: float,
        units_label: str,
        pore_scale: float,
    ) -> dict[str, Any]:
        if measurement_samples:
            radii_px = np.array([sample[4] for sample in measurement_samples], dtype=np.float32)
            angles_arr = np.array([sample[5] for sample in measurement_samples], dtype=np.float32)
            thickness_px = 2.0 * radii_px
            thickness_units = thickness_px * float(pixel_size)
            xs = np.array([sample[1] for sample in measurement_samples], dtype=np.int32)
            ys = np.array([sample[0] for sample in measurement_samples], dtype=np.int32)
            samples_table = np.column_stack([xs, ys, thickness_px, angles_arr]).astype(object)
        else:
            angles_arr = np.array([], dtype=np.float32)
            thickness_units = np.array([], dtype=np.float32)
            samples_table = None

        angles_arr = angles_arr[np.isfinite(angles_arr)] if angles_arr.size else angles_arr
        safe_scale = float(pore_scale) if np.isfinite(pore_scale) and pore_scale > 0 else 1.0
        pore_diams_units = np.array(
            [2.0 * p.get("r_px_orig", p["r_px"] / safe_scale) * float(pixel_size) for p in pores],
            dtype=np.float32,
        )

        if thickness_units.size:
            diam_mean = float(np.mean(thickness_units))
            diam_sd = float(np.std(thickness_units))
            diam_median = float(np.median(thickness_units))
            if np.ptp(thickness_units) == 0:
                diam_mode = float(thickness_units[0])
            else:
                hist, edges = np.histogram(thickness_units, bins=40)
                peak = int(np.argmax(hist))
                diam_mode = float((edges[peak] + edges[peak + 1]) / 2.0)
        else:
            diam_mean = diam_sd = diam_mode = diam_median = float("nan")

        orientation_index, mean_angle_deg, std_angle_deg = self._compute_orientation_stats(angles_arr)
        fragments_with_centerlines = self.analyzer.get_measurement_fragment_count() if self.analyzer is not None else 0

        fiber_summary = {
            "diam_mean": diam_mean,
            "diam_sd": diam_sd,
            "diam_mode": diam_mode,
            "diam_median": diam_median,
            "orientation_index": float(orientation_index),
            "mean_angle_deg": float(mean_angle_deg),
            "std_angle_deg": float(std_angle_deg),
            "porosity": float(porosity),
            "sample_count": len(measurement_samples),
            "fragments_with_centerlines": int(fragments_with_centerlines),
            "units": units_label,
        }
        return {
            "samples_table": samples_table,
            "thickness_units": thickness_units,
            "pore_diams_units": pore_diams_units,
            "fiber_summary": fiber_summary,
        }

    def measure(self, frame: Frame, config: AppConfig, sample_props: SampleProperties) -> dict[str, Any]:
        if self.analyzer is None:
            raise RuntimeError("Fiber analyzer has not been initialized.")

        measurement_params = self.measurement_params(config)
        spacing_px_input = float(measurement_params["sample_spacing"])
        current_scale = frame.scale_factor if frame.scale_factor and np.isfinite(frame.scale_factor) else 1.0
        spacing_px_scaled = max(1.0, spacing_px_input * current_scale)
        pixel_size = sample_props.effective_pixel_size_xy()
        units_label = sample_props.unit if sample_props.unit else "px"

        self.analyzer.scale_factor = frame.scale_factor
        measurement_samples, drawing_samples = self.analyzer.analyze(spacing_px=spacing_px_scaled)
        thr_u8 = int(measurement_params["seg_threshold"])
        void_mask, pore_dt = self.analyzer.void_mask_and_pore_dt(thr_u8)
        porosity = float(void_mask.mean()) if void_mask is not None else float("nan")
        pores = self.analyzer.sample_inscribed_circles(
            pore_dt=pore_dt,
            void_mask=void_mask,
            min_radius_px=1.5 * current_scale,
            min_peak_px=3.0 * current_scale,
            peak_footprint=5,
            max_pores=None,
        )

        outputs = self._build_measurement_outputs(
            measurement_samples=measurement_samples,
            pores=pores,
            porosity=porosity,
            pixel_size=pixel_size,
            units_label=units_label,
            pore_scale=frame.scale_factor,
        )
        outputs.update(
            {
                "measurement_samples": measurement_samples,
                "drawing_samples": drawing_samples,
                "porosity": porosity,
                "void_mask": void_mask,
                "pore_dt": pore_dt,
                "pores": pores,
            }
        )
        return outputs

    @staticmethod
    def _clip_point(y: float, x: float, height: int, width: int) -> tuple[int, int]:
        iy = int(round(float(y)))
        ix = int(round(float(x)))
        iy = min(max(0, iy), max(0, height - 1))
        ix = min(max(0, ix), max(0, width - 1))
        return iy, ix

    @classmethod
    def _draw_line(
        cls,
        img: np.ndarray | None,
        p0: tuple[float, float],
        p1: tuple[float, float],
        color: tuple[int, int, int] = (255, 255, 0),
        thickness: int = 1,
    ) -> None:
        if img is None:
            return
        height, width = img.shape[:2]
        y0, x0 = cls._clip_point(*p0, height, width)
        y1, x1 = cls._clip_point(*p1, height, width)
        cv2.line(img, (x0, y0), (x1, y1), color, int(thickness), cv2.LINE_AA)

    @staticmethod
    def _draw_circle(
        img: np.ndarray,
        center_xy: tuple[int, int],
        radius_px: int,
        color: tuple[int, int, int] = (255, 255, 0),
        thickness: int = 1,
        line_type: int = cv2.LINE_AA,
    ) -> None:
        cv2.circle(img, center_xy, int(radius_px), color, thickness=thickness, lineType=line_type)

    @staticmethod
    def _pore_dt_to_rgb(pore_dt: np.ndarray | None, void_mask_u8: np.ndarray | None) -> np.ndarray | None:
        if pore_dt is None or void_mask_u8 is None:
            return None
        vm = void_mask_u8.astype(np.uint8) > 0
        norm = np.zeros_like(pore_dt, dtype=np.float32)
        if np.any(vm):
            vmax = float(np.percentile(pore_dt[vm], 99.0))
            if vmax <= 0:
                vmax = float(np.max(pore_dt))
            if vmax > 0:
                norm[vm] = (pore_dt[vm] / vmax).clip(0.0, 1.0)
        base_gray = (norm * 255.0).astype(np.uint8)
        base_rgb = cv2.applyColorMap(base_gray, cv2.COLORMAP_TURBO)
        base_rgb[~vm] = (0, 0, 0)
        return base_rgb

    @staticmethod
    def _angle_to_bgr(angle_deg: float) -> tuple[int, int, int]:
        if not np.isfinite(angle_deg):
            return (200, 200, 200)
        ang = ((-angle_deg + 30.) % 180.0) - 90.0
        hue = (ang + 90.0) / 180.0 * 179.0
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    @staticmethod
    def _weighted_axial_average_angles(
        sample_angles_deg: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        distances = np.asarray(distances, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32)

        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        neighbor_angles = sample_angles_deg[indices]
        weights = 1.0 / (distances + 1e-6)

        theta2 = np.deg2rad(2.0 * neighbor_angles)
        c = np.sum(weights * np.cos(theta2), axis=1)
        s = np.sum(weights * np.sin(theta2), axis=1)

        return 0.5 * np.rad2deg(np.arctan2(s, c))

    def _fill_orientation_overlay_by_label(self, frame: Frame) -> None:
        img = frame.orientations_overlay
        labels = frame.labels_2d
        samples = frame.drawing_samples or []
        if img is None or labels is None or not samples:
            return

        sample_map: dict[int, list[tuple[float, float, float]]] = {}
        height, width = labels.shape[:2]
        for y, x, _ny, _nx, _r, angle_deg in samples:
            iy = int(round(float(y)))
            ix = int(round(float(x)))
            if iy < 0 or iy >= height or ix < 0 or ix >= width:
                continue
            label_id = int(labels[iy, ix])
            if label_id <= 0 or not np.isfinite(angle_deg):
                continue
            sample_map.setdefault(label_id, []).append((float(y), float(x), float(angle_deg)))

        for label_id, label_samples in sample_map.items():
            mask = labels == label_id
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue

            sample_arr = np.asarray(label_samples, dtype=np.float32)
            tree = cKDTree(sample_arr[:, :2])
            query_coords = np.stack([ys, xs], axis=1).astype(np.float32)

            k = min(4, sample_arr.shape[0])
            distances, indices = tree.query(query_coords, k=k)

            angles = sample_arr[:, 2]
            pixel_angles = self._weighted_axial_average_angles(angles, distances, indices)
            pixel_colors = np.asarray(
                [self._angle_to_bgr(float(angle)) for angle in pixel_angles],
                dtype=np.uint8,
            )
            img[ys, xs] = pixel_colors

    def build_measurement_overlays(self, frame: Frame) -> None:
        sampling_base = ensure_rgb_u8(frame.dt_u8 if frame.dt_u8 is not None else frame.dt_normalized)
        if sampling_base is None and frame.overlay_img is not None:
            sampling_base = ensure_rgb_u8(frame.overlay_img)

        base_img = frame.orig_img if frame.scale_factor == 1.0 or frame.orig_img_scaled is None else frame.orig_img_scaled
        base_rgb = ensure_rgb_u8(base_img)

        frame.sampling_overlay = sampling_base.copy() if sampling_base is not None else None
        frame.diameters_overlay = (
            frame.overlay_img.copy() if frame.overlay_img is not None else (base_rgb.copy() if base_rgb is not None else None)
        )
        frame.orientations_overlay = base_rgb.copy() if base_rgb is not None else None
        frame.pores_overlay = self._pore_dt_to_rgb(frame.pore_dt, frame.void_mask)

        pts = list(frame.drawing_samples or [])
        if len(pts) > 15000:
            step = int(np.ceil(len(pts) / 15000))
            pts = pts[::step]

        if pts and frame.sampling_overlay is not None:
            height, width = frame.sampling_overlay.shape[:2]
            for idx, (y, x, ny, nx, r, _angle_deg) in enumerate(pts):
                if not (0 <= y < height and 0 <= x < width) or r <= 0:
                    continue

                p0 = (y - ny * r, x - nx * r)
                p1 = (y + ny * r, x + nx * r)
                self._draw_line(frame.sampling_overlay, p0, p1, color=(255, 255, 0))
                self._draw_circle(
                    frame.sampling_overlay,
                    (int(round(x)), int(round(y))),
                    radius_px=1,
                    color=(255, 0, 0),
                    thickness=-1,
                )
                if frame.diameters_overlay is not None and (idx % 10) == 0:
                    self._draw_line(frame.diameters_overlay, p0, p1, color=(255, 255, 0))

        self._fill_orientation_overlay_by_label(frame)

        if frame.pores_overlay is not None and frame.pores:
            frame.pores_overlay = frame.pores_overlay.copy()
            for pore in frame.pores:
                x, y = int(pore["x"]), int(pore["y"])
                r = int(round(pore["r_px"]))
                if r > 0:
                    self._draw_circle(frame.pores_overlay, (x, y), r, color=(255, 255, 255), thickness=1)

    def ensure_prediction(self, frame: Frame, config: AppConfig, *, invert_input: bool) -> PipelineOutcome:
        if frame.orig_img is None:
            return PipelineOutcome(False, info_title="Load original", info_message="Load an image before using this operation.")
        if self.analyzer is None:
            return PipelineOutcome(False, error_title="Analyzer unavailable", error_message="Fiber analyzer has not been initialized.")

        target_sig = self.prediction_signature(frame, config, invert_input=invert_input)
        if target_sig is None:
            return PipelineOutcome(False)

        if frame.has_prediction and frame.dt is not None and frame.prediction_sig == target_sig:
            self._hydrate_analyzer_from_prediction(frame)
            return PipelineOutcome(True)

        try:
            prediction = self.predict(frame, config, invert_input=invert_input)
        except Exception as exc:
            frame.fail(str(exc))
            return PipelineOutcome(False, error_title="Prediction failed", error_message=str(exc))

        frame.set_prediction_outputs(
            dt=prediction["dt"],
            gx=prediction["gx"],
            gy=prediction["gy"],
            dt_u8=prediction["dt_u8"],
            dt_normalized=prediction["dt_normalized"],
            foreground_prob=prediction["foreground_prob"],
            lapl_img=prediction["lapl_img"],
            orig_img_scaled=frame.orig_img_scaled,
            scale_factor=frame.scale_factor,
        )
        frame.prediction_sig = target_sig
        return PipelineOutcome(True)

    def ensure_segmentation_centerlines(self, frame: Frame, config: AppConfig, *, invert_input: bool) -> PipelineOutcome:
        ready = self.ensure_prediction(frame, config, invert_input=invert_input)
        if not ready.ok:
            return ready
        if self.analyzer is None:
            return PipelineOutcome(False, error_title="Analyzer unavailable", error_message="Fiber analyzer has not been initialized.")

        try:
            outputs = self.build_centerlines(frame, config)
        except Exception as exc:
            frame.fail(str(exc))
            return PipelineOutcome(False, error_title="Centerline computation failed", error_message=str(exc))

        frame.segmentation_centerlines_overlay_img = outputs["segmentation_centerlines_overlay_img"]
        frame.measurement_centerlines_overlay_img = outputs["measurement_centerlines_overlay_img"]
        frame.centerlines_comparison_overlay_img = outputs["centerlines_comparison_overlay_img"]
        return PipelineOutcome(True)

    def ensure_segmentation(self, frame: Frame, config: AppConfig, *, invert_input: bool) -> PipelineOutcome:
        if frame.orig_img is None:
            return PipelineOutcome(False, info_title="Load original", info_message="Please load an original image first.")
        if self.analyzer is None:
            return PipelineOutcome(False, error_title="Analyzer unavailable", error_message="Fiber analyzer has not been initialized.")

        frame.error_message = None
        target_sig = self.segmentation_signature(frame, config, invert_input=invert_input)
        if target_sig is None:
            return PipelineOutcome(False)
        if (
            frame.has_segmentation
            and frame.labels_2d is not None
            and frame.segmentation_sig == target_sig
            and self.analyzer.get_labels() is frame.labels_2d
        ):
            return PipelineOutcome(True)

        self.invalidate_segmentation(frame)
        centerline_state = self.ensure_segmentation_centerlines(frame, config, invert_input=invert_input)
        if not centerline_state.ok:
            return centerline_state

        try:
            outputs = self.segment_stage(
                frame,
                config,
                {
                    "segmentation_centerlines_overlay_img": frame.segmentation_centerlines_overlay_img,
                    "measurement_centerlines_overlay_img": frame.measurement_centerlines_overlay_img,
                    "centerlines_comparison_overlay_img": frame.centerlines_comparison_overlay_img,
                },
            )
        except Exception as exc:
            frame.fail(str(exc))
            return PipelineOutcome(False, error_title="Segmentation failed", error_message=str(exc))

        frame.set_segmentation_outputs(
            labels_2d=outputs["labels_2d"],
            segmentation_centerlines_overlay_img=outputs["segmentation_centerlines_overlay_img"],
            measurement_centerlines_overlay_img=outputs["measurement_centerlines_overlay_img"],
            centerlines_comparison_overlay_img=outputs["centerlines_comparison_overlay_img"],
            overlay_img=outputs["overlay_img"],
            sampling_overlay=frame.sampling_overlay,
            labels_rgb_cache=outputs["labels_rgb_cache"],
            segmentation_outline_cache=outputs["segmentation_outline_cache"],
        )
        frame.segmentation_sig = target_sig
        return PipelineOutcome(True)

    def ensure_measurements(
        self,
        frame: Frame,
        config: AppConfig,
        sample_props: SampleProperties,
        *,
        invert_input: bool,
    ) -> PipelineOutcome:
        target_sig = self.measurement_signature(frame, config, sample_props, invert_input=invert_input)
        if (
            target_sig is not None
            and frame.has_measurements
            and frame.fiber_summary is not None
            and frame.measurement_sig == target_sig
        ):
            return PipelineOutcome(True)

        segmentation_ready = self.ensure_segmentation(frame, config, invert_input=invert_input)
        if not segmentation_ready.ok:
            return segmentation_ready
        if self.analyzer is None:
            return PipelineOutcome(False, info_title="No analyzer", info_message="Measurement data unavailable.")
        if target_sig is None:
            return PipelineOutcome(False)

        self.invalidate_measurements(frame)
        if not self.analyzer.get_measurement_centerline_segments():
            return PipelineOutcome(False, info_title="No centerlines", info_message="No measurement centerlines available.")

        frame.error_message = None
        try:
            outputs = self.measure(frame=frame, config=config, sample_props=sample_props)
        except Exception as exc:
            frame.fail(str(exc))
            return PipelineOutcome(False, error_title="Analysis failed", error_message=str(exc))

        frame.set_measurement_outputs(
            samples_table=outputs["samples_table"],
            fiber_summary=outputs["fiber_summary"],
            measurement_samples=outputs["measurement_samples"],
            drawing_samples=outputs["drawing_samples"],
            thickness_units=outputs["thickness_units"],
            porosity=outputs["porosity"],
            pore_diams_units=outputs["pore_diams_units"],
            void_mask=outputs["void_mask"],
            pore_dt=outputs["pore_dt"],
            pores=outputs["pores"],
            sampling_overlay=frame.sampling_overlay,
        )
        self.build_measurement_overlays(frame)
        frame.measurement_sig = target_sig
        return PipelineOutcome(True)

# ---- merged from batch_service.py ----


from dataclasses import dataclass
from typing import Callable

import numpy as np

from .domain import BatchResult, Frame, ImageBatch


@dataclass
class BatchCallbacks:
    """Callbacks invoked during batch run."""

    on_started: Callable[[int], None]
    on_frame_started: Callable[[int, int], None]
    on_frame_finished: Callable[[int, int, Frame], None]
    on_finished: Callable[[BatchResult], None]


class BatchService:
    """Runs frame processing over an `ImageBatch` and assembles results."""

    def run_batch(
        self,
        image_batch: ImageBatch,
        compute_measurements: bool,
        process_frame: Callable[[np.ndarray, str, int, Frame | None, bool], Frame],
        callbacks: BatchCallbacks,
        cached_result: BatchResult | None = None,
    ) -> BatchResult:
        total_images = len(image_batch)
        callbacks.on_started(total_images)

        z_indices = image_batch.metadata.get("z_indices", list(range(total_images)))
        filenames = image_batch.metadata.get("filenames", [str(i) for i in range(total_images)])

        results: list[Frame] = []

        for idx, frame in enumerate(image_batch.frames):
            frame_id = str(filenames[idx]) if idx < len(filenames) else str(idx)
            z_index = int(z_indices[idx]) if idx < len(z_indices) else idx
            callbacks.on_frame_started(idx, total_images)
            cached_frame = None
            if cached_result is not None and 0 <= int(z_index) < len(cached_result.frame_results):
                cached_frame = cached_result.frame_at(int(z_index))
            result = process_frame(frame, frame_id, z_index, cached_frame, compute_measurements)
            results.append(result)
            callbacks.on_frame_finished(idx + 1, total_images, result)

        batch_result = BatchResult(frame_results=results)
        callbacks.on_finished(batch_result)
        return batch_result
