"""Pipeline orchestration for frame and batch processing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from . import analysis
from .domain import AppConfig, BatchResult, Frame, ImageBatch, SampleProperties
from .ml import OnnxFiberposeModel

# Config fields passed to `analysis.segment`; changing any of them invalidates the segmentation.
_SEGMENTATION_FIELDS = (
    "segmentation_mode",
    "watershed_mask_threshold",
    "min_segment_foreground_mean",
    "max_angle_deg",
    "ridge_sigma",
    "ridge_low",
    "ridge_high",
    "strict_centerlines",
    "keep_segments_without_measurement_centerlines",
    "merge_expansion_factor",
    "segmentation_expansion_factor",
    "min_fragment_centerline_length_px",
    "grad_quality_trim_fraction",
)


@dataclass
class PipelineOutcome:
    """Result envelope for pipeline stage execution."""

    ok: bool
    error_title: str | None = None
    error_message: str | None = None
    info_title: str | None = None
    info_message: str | None = None


class PipelineController:
    """Runs prediction -> segmentation -> measurement on a `Frame`, caching each stage on the frame.

    The only state kept here is the loaded model; all image results live on the frame.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model: OnnxFiberposeModel | None = None

    def set_model(self, model_path: str | None, device: str) -> None:
        """Select the model; it is loaded on the next prediction."""
        if (model_path, device) != (self.model_path, self.device):
            self.model_path, self.device, self._model = model_path, device, None

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
    def segmentation_params(config: AppConfig) -> dict[str, Any]:
        return {name: getattr(config, name) for name in _SEGMENTATION_FIELDS}

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
            "segmentation_params": cls.segmentation_params(config),
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
        fragments_with_centerlines: int,
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
        measurement_params = self.measurement_params(config)
        spacing_px_input = float(measurement_params["sample_spacing"])
        current_scale = frame.scale_factor if frame.scale_factor and np.isfinite(frame.scale_factor) else 1.0
        spacing_px_scaled = max(1.0, spacing_px_input * current_scale)
        pixel_size = sample_props.effective_pixel_size_xy()
        units_label = sample_props.unit if sample_props.unit else "px"

        measurement_samples, drawing_samples = analysis.sample_centerlines(
            frame.dt, frame.gx, frame.gy, frame.fragments, spacing_px=spacing_px_scaled, scale_factor=frame.scale_factor
        )
        void_mask, pore_dt = analysis.void_mask_and_pore_dt(frame.dt_u8, int(measurement_params["seg_threshold"]))
        porosity = float(void_mask.mean())
        pores = analysis.sample_inscribed_circles(
            pore_dt=pore_dt,
            void_mask=void_mask,
            scale_factor=frame.scale_factor,
            min_radius_px=1.5 * current_scale,
            min_peak_px=3.0 * current_scale,
            peak_footprint=5,
        )

        outputs = self._build_measurement_outputs(
            measurement_samples=measurement_samples,
            pores=pores,
            porosity=porosity,
            pixel_size=pixel_size,
            units_label=units_label,
            pore_scale=frame.scale_factor,
            fragments_with_centerlines=sum(f.has_measurement_centerline for f in frame.fragments),
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

    def ensure_prediction(self, frame: Frame, config: AppConfig, *, invert_input: bool) -> PipelineOutcome:
        if frame.orig_img is None:
            return PipelineOutcome(False, info_title="Load original", info_message="Load an image before using this operation.")

        target_sig = self.prediction_signature(frame, config, invert_input=invert_input)
        if frame.has_prediction and frame.prediction_sig == target_sig:
            return PipelineOutcome(True)

        try:
            if self._model is None:
                self._model = OnnxFiberposeModel(str(self.model_path), device=self.device)
            image = self._image_for_inference(self._processing_image(frame, config), invert_input=invert_input)
            prediction = analysis.predict(self._model, image)
        except Exception as exc:
            frame.fail(str(exc))
            return PipelineOutcome(False, error_title="Prediction failed", error_message=str(exc))

        frame.set_prediction_outputs(**prediction, orig_img_scaled=frame.orig_img_scaled, scale_factor=frame.scale_factor)
        frame.prediction_sig = target_sig
        return PipelineOutcome(True)

    def ensure_segmentation(self, frame: Frame, config: AppConfig, *, invert_input: bool) -> PipelineOutcome:
        frame.error_message = None
        ready = self.ensure_prediction(frame, config, invert_input=invert_input)
        if not ready.ok:
            return ready
        target_sig = self.segmentation_signature(frame, config, invert_input=invert_input)
        if frame.has_segmentation and frame.segmentation_sig == target_sig:
            return PipelineOutcome(True)

        frame.clear_segmentation()
        try:
            labels, fragments = analysis.segment(
                frame.dt,
                frame.dt_u8,
                frame.gx,
                frame.gy,
                frame.foreground_prob,
                **self.segmentation_params(config),
            )
        except Exception as exc:
            frame.fail(str(exc))
            return PipelineOutcome(False, error_title="Segmentation failed", error_message=str(exc))

        frame.set_segmentation_outputs(labels_2d=labels, fragments=fragments)
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
        if target_sig is not None and frame.has_measurements and frame.measurement_sig == target_sig:
            return PipelineOutcome(True)

        segmentation_ready = self.ensure_segmentation(frame, config, invert_input=invert_input)
        if not segmentation_ready.ok:
            return segmentation_ready

        frame.clear_measurements()
        if not any(f.has_measurement_centerline for f in frame.fragments):
            return PipelineOutcome(False, info_title="No centerlines", info_message="No measurement centerlines available.")

        try:
            outputs = self.measure(frame=frame, config=config, sample_props=sample_props)
        except Exception as exc:
            frame.fail(str(exc))
            return PipelineOutcome(False, error_title="Analysis failed", error_message=str(exc))

        frame.set_measurement_outputs(**outputs)
        frame.measurement_sig = target_sig
        return PipelineOutcome(True)


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
