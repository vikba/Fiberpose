from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from copy import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import imageio.v3 as iio
import numpy as np
import logging
from tifffile import TiffFile
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    # --- Model & Hardware ---
    checkpoint_path: str = ""
    use_gpu: bool = True
    scale_label: str = "1x"
    
    # --- Centerline Params ---
    max_angle_deg: float = 30.0
    strict_centerlines: bool = False
    
    # --- Segmentation Params ---
    segmentation_mode: str = "watershed"
    seg_threshold: int = 1
    
    # --- Analysis & Physical Units ---
    sample_spacing: float = 5.0

    # --- 3D View Rendering ---
    render3d_max_display_voxels: int = 8_000_000
    render3d_max_display_edge: int = 320
    
    # --- Clustering ---
    cluster_enabled: bool = False
    cluster_params: dict = field(default_factory=dict)

    # --- DT gradient quality view ---
    # Signed error breakpoints for s = |grad(dt)| - 1:
    # [p0, p1, p2, p3] with p0 < p1 <= 0 <= p2 < p3.
    grad_quality_points: list[float] = field(
        default_factory=lambda: [-0.35, -0.08, 0.08, 0.35]
    )

    # --- Save/export defaults (single image, folder batch, and stack exports) ---
    save_options: dict = field(
        default_factory=lambda: {
            "dt": True,
            "labels": True,
            "overlay": True,
            "debug": True,
            "outline": False,
            "samples": True,
            "style_vector": False,
            "image_format": "png",
            "collision": "suffix",
            "write_manifest": True,
        }
    )

    def save_to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load_from_json(cls, filepath: str, *, strict: bool = False) -> "AppConfig":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Config JSON root must be an object.")
            # Backward compatibility for renamed config keys.
            if "scale_label" not in data and "scale_string" in data:
                data["scale_label"] = data.get("scale_string")
            # Backward compatibility for legacy save-option keys.
            if "save_options" not in data:
                default_save_options = cls().save_options
                merged_save_options = dict(default_save_options)
                pred_opts = data.get("single_predictions_save_options")
                meas_opts = data.get("single_measurements_save_options")
                if isinstance(pred_opts, dict):
                    for key in ("dt", "labels", "overlay", "debug", "outline", "samples", "style_vector"):
                        if key in pred_opts:
                            merged_save_options[key] = bool(pred_opts.get(key))
                if isinstance(meas_opts, dict) and "samples" in meas_opts:
                    merged_save_options["samples"] = bool(meas_opts.get("samples"))
                data["save_options"] = merged_save_options
            valid_keys = {k for k in cls.__dataclass_fields__}
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered_data)
        except FileNotFoundError:
            if strict:
                raise
            logger.info("Config file not found: %s. Using defaults.", filepath)
        except json.JSONDecodeError as exc:
            if strict:
                raise
            logger.error("Invalid JSON in config file %s: %s. Using defaults.", filepath, exc)
        except Exception:
            if strict:
                raise
            logger.exception("Failed to load config file %s. Using defaults.", filepath)
        return cls()


@dataclass
class SampleProperties:
    """Per-loaded-sample physical calibration used by analysis and UI."""

    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    voxel_size_z: float = 1.0
    unit: str = "px"
    source: str = "default"
    source_path: str = ""

    @staticmethod
    def _as_positive(value: Any, default: float = 1.0) -> float:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            try:
                value = float(value[0]) / float(value[1])
            except (TypeError, ValueError, ZeroDivisionError):
                return float(default)
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        if out <= 0 or not np.isfinite(out):
            return float(default)
        return out

    @classmethod
    def defaults(cls) -> "SampleProperties":
        return cls(
            pixel_size_x=1.0,
            pixel_size_y=1.0,
            voxel_size_z=1.0,
            unit="px",
            source="default",
            source_path="",
        )

    @classmethod
    def from_image_metadata(
        cls,
        metadata: Optional[Dict[str, Any]],
        *,
        source_path: str = "",
    ) -> "SampleProperties":
        meta = metadata or {}
        px_raw = meta.get("pixel_width")
        py_raw = meta.get("pixel_height")
        pz_raw = meta.get("voxel_depth")
        unit_raw = meta.get("unit")

        has_px = px_raw is not None and cls._as_positive(px_raw, default=-1.0) > 0
        has_py = py_raw is not None and cls._as_positive(py_raw, default=-1.0) > 0
        has_pz = pz_raw is not None and cls._as_positive(pz_raw, default=-1.0) > 0

        px = cls._as_positive(px_raw) if has_px else None
        py = cls._as_positive(py_raw) if has_py else None
        pz = cls._as_positive(pz_raw) if has_pz else 1.0

        if px is None and py is None:
            px = 1.0
            py = 1.0
        elif px is None:
            px = py
        elif py is None:
            py = px

        unit_txt = str(unit_raw).strip() if unit_raw is not None else ""
        unit = unit_txt if unit_txt else "px"

        has_metadata = bool(has_px or has_py or has_pz or unit_txt)
        return cls(
            pixel_size_x=float(px),
            pixel_size_y=float(py),
            voxel_size_z=float(pz),
            unit=unit,
            source="metadata" if has_metadata else "default",
            source_path=str(source_path or meta.get("source_path") or ""),
        )

    def effective_pixel_size_xy(self) -> float:
        x = self._as_positive(self.pixel_size_x)
        y = self._as_positive(self.pixel_size_y)
        return 0.5 * (x + y)

    def spacing_zyx(self) -> tuple[float, float, float]:
        return (
            self._as_positive(self.voxel_size_z),
            self._as_positive(self.pixel_size_y),
            self._as_positive(self.pixel_size_x),
        )

@dataclass
class ImageBatch:
    """Container for an ordered batch of frames plus lightweight metadata."""
    frames: List[np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def kind(self) -> str:
        """Returns the kind of loaded info: 'Image', 'Folder', or 'Stack'."""
        return self.metadata.get("source_type", "image")

    @property
    def name(self) -> str:
        """Returns the base name of the file or folder."""
        path = self.metadata.get("source_path", "")
        if not path:
            return "Unknown"
        return os.path.basename(os.path.normpath(path))

    @staticmethod
    def _extract_voxel_sizes(path: str) -> Dict[str, Any]:
        """Read physical calibration from TIFF tags and common metadata keys."""
        sizes = {"pixel_width": None, "pixel_height": None, "voxel_depth": None, "unit": None}

        def _positive(value: Any) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, (tuple, list)) and len(value) == 2:
                try:
                    value = float(value[0]) / float(value[1])
                except (TypeError, ValueError, ZeroDivisionError):
                    return None
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            if out <= 0 or not np.isfinite(out):
                return None
            return out

        def _unit_scale_to_um(unit_value: Any) -> Optional[float]:
            if unit_value is None:
                return None
            token = str(getattr(unit_value, "name", unit_value)).strip().lower()
            if token in {"inch", "in", "inches"}:
                return 25400.0
            if token in {"centimeter", "centimetre", "cm"}:
                return 10000.0
            if token in {"millimeter", "millimetre", "mm"}:
                return 1000.0
            if token in {"resunit.inch", "resunit.centimeter", "resunit.centimetre"}:
                if "inch" in token:
                    return 25400.0
                return 10000.0
            try:
                token_num = int(unit_value)
            except (TypeError, ValueError):
                return None
            if token_num == 2:
                return 25400.0
            if token_num == 3:
                return 10000.0
            return None

        try:
            meta = iio.immeta(path)
        except Exception:
            meta = {}

        sizes["voxel_depth"] = (
            _positive(meta.get("spacing"))
            or _positive(meta.get("PhysicalSizeZ"))
        )
        sizes["unit"] = str(
            meta.get("unit")
            or meta.get("PhysicalSizeXUnit")
            or meta.get("PhysicalSizeYUnit")
            or meta.get("PhysicalSizeZUnit")
            or ""
        ).strip() or None

        sizes["pixel_width"] = (
            _positive(meta.get("pixel_width"))
            or _positive(meta.get("PhysicalSizeX"))
        )
        sizes["pixel_height"] = (
            _positive(meta.get("pixel_height"))
            or _positive(meta.get("PhysicalSizeY"))
        )

        if (
            (sizes["pixel_width"] is None or sizes["pixel_height"] is None)
            and str(path).lower().endswith((".tif", ".tiff"))
        ):
            with TiffFile(path) as tif:
                tags = tif.pages[0].tags
                x_res = _positive(tags["XResolution"].value if "XResolution" in tags else None)
                y_res = _positive(tags["YResolution"].value if "YResolution" in tags else None)
                res_unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None
                unit_scale = _unit_scale_to_um(res_unit)

                if sizes["pixel_width"] is None and x_res is not None:
                    sizes["pixel_width"] = (1.0 / x_res) * (unit_scale or 1.0)
                if sizes["pixel_height"] is None and y_res is not None:
                    sizes["pixel_height"] = (1.0 / y_res) * (unit_scale or 1.0)
                if sizes["unit"] is None and unit_scale is not None:
                    sizes["unit"] = "um"

        # Minimal fallback for non-TIFF metadata formats.
        if sizes["pixel_width"] is None or sizes["pixel_height"] is None:
            res = meta.get("resolution")
            rx = None
            ry = None
            if isinstance(res, (tuple, list)) and len(res) >= 2:
                rx = _positive(res[0])
                ry = _positive(res[1])
            elif isinstance(res, dict):
                rx = _positive(res.get("x") or res.get("X"))
                ry = _positive(res.get("y") or res.get("Y"))

            unit_scale = _unit_scale_to_um(
                meta.get("resolution_unit")
                or meta.get("ResolutionUnit")
                or meta.get("resolutionunit")
            )
            if sizes["pixel_width"] is None and rx is not None:
                sizes["pixel_width"] = (1.0 / rx) * (unit_scale or 1.0)
            if sizes["pixel_height"] is None and ry is not None:
                sizes["pixel_height"] = (1.0 / ry) * (unit_scale or 1.0)
            if sizes["unit"] is None and unit_scale is not None:
                sizes["unit"] = "um"

        return sizes

    @staticmethod
    def _normalize_stack(data: np.ndarray) -> List[np.ndarray]:
        if data.ndim == 2:
            return [data]
        if data.ndim == 3:
            if data.shape[2] in (3, 4):
                return [data]
            return [data[z] for z in range(data.shape[0])]
        if data.ndim == 4:
            return [data[z] for z in range(data.shape[0])]
        raise ValueError("Unsupported image dimensions for batch loading.")

    @classmethod
    def from_path(cls, path: str) -> "ImageBatch":
        data = iio.imread(path)
        frames = cls._normalize_stack(data)
        source_type = "Image" if len(frames) == 1 else "Stack"
        filenames = [os.path.basename(path)] * len(frames)
        
        # --- Extract physical sizes ---
        voxel_sizes = cls._extract_voxel_sizes(path)
        
        metadata = {
            "source_type": source_type,
            "source_path": path,
            "filenames": filenames,
            "z_indices": list(range(len(frames))),
            **voxel_sizes  # Merge the extracted sizes into the metadata dictionary
        }
        return cls(frames=frames, metadata=metadata)

    @classmethod
    def from_folder(
        cls, folder: str, extensions: Optional[Iterable[str]] = None
    ) -> "ImageBatch":
        entries = sorted(os.listdir(folder))
        ext_set = {ext.lower() for ext in extensions} if extensions else None
        frames: List[np.ndarray] = []
        filenames: List[str] = []
        
        first_valid_path = None

        for name in entries:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if ext_set is not None:
                if os.path.splitext(name)[1].lower() not in ext_set:
                    continue
            try:
                data = iio.imread(path)
                if first_valid_path is None:
                    first_valid_path = path # Save the first image to extract metadata
            except Exception:
                continue
            
            image_frames = cls._normalize_stack(data)
            for idx, frame in enumerate(image_frames):
                frames.append(frame)
                if len(image_frames) == 1:
                    filenames.append(name)
                else:
                    filenames.append(f"{os.path.splitext(name)[0]}_z{idx:03d}")
                    
        # --- Extract physical sizes from the first image in the folder ---
        voxel_sizes = {}
        if first_valid_path:
            voxel_sizes = cls._extract_voxel_sizes(first_valid_path)
            
        metadata = {
            "source_type": "Folder",
            "source_path": folder,
            "filenames": filenames,
            "z_indices": list(range(len(frames))),
            **voxel_sizes
        }
        return cls(frames=frames, metadata=metadata)

    @classmethod
    def from_source(
        cls, path: str, extensions: Optional[Iterable[str]] = None
    ) -> "ImageBatch":
        if os.path.isdir(path):
            return cls.from_folder(path, extensions=extensions)
        if os.path.isfile(path):
            return cls.from_path(path)
        raise FileNotFoundError(f"Source path does not exist: {path}")
    
@dataclass
class BatchResult:
    frame_results: List[Frame]
    semantic_volume: Optional[np.ndarray] = None
    semantic_skeleton_volume: Optional[np.ndarray] = None
    stack3d_meta: Optional[Dict[str, Any]] = None

    def frame_at(self, index: int) -> Frame:
        return self.frame_results[index]

    @property
    def has_3d_input(self) -> bool:
        return len(self.frame_results) > 1

    @property
    def has_3d_outputs(self) -> bool:
        return (
            self.semantic_volume is not None
            or self.semantic_skeleton_volume is not None
        )

    def save_all(self, output_dir: str, export_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        from .exporter import BatchResultExporter

        exporter = BatchResultExporter(export_config=export_config)
        return exporter.save_all(self, output_dir, export_config=export_config)

    # Backward compatibility for callers still using the previous name.
    @property
    def merge_meta(self) -> Optional[Dict[str, Any]]:
        return self.stack3d_meta

    @merge_meta.setter
    def merge_meta(self, value: Optional[Dict[str, Any]]) -> None:
        self.stack3d_meta = value


@dataclass
class Frame:
    """Flat, stage-aware container for a single frame's processing state."""

    # Identity
    z_index: int = 0
    frame_id: str = ""

    # Stage flags
    has_prediction: bool = False
    has_segmentation: bool = False
    has_measurements: bool = False
    error_message: Optional[str] = None

    # Source and prediction buffers
    orig_img: Optional[np.ndarray] = None
    orig_img_scaled: Optional[np.ndarray] = None
    scale_factor: float = 1.0
    dt: Optional[np.ndarray] = None
    dt_u8: Optional[np.ndarray] = None
    dt_normalized: Optional[np.ndarray] = None
    lapl_img: Optional[np.ndarray] = None
    gx: Optional[np.ndarray] = None
    gy: Optional[np.ndarray] = None
    style_vec: Optional[np.ndarray] = None

    # Segmentation products
    labels_2d: Optional[np.ndarray] = None
    centerlines_overlay_img: Optional[np.ndarray] = None
    overlay_img: Optional[np.ndarray] = None
    sampling_overlay: Optional[np.ndarray] = None
    measurements_overlay: Optional[np.ndarray] = None

    # Measurement outputs
    samples_table: Optional[Any] = None
    fiber_summary: Optional[Dict[str, Any]] = None
    overlay_samples: Optional[Any] = None
    angles_samples_deg: Optional[np.ndarray] = None
    thickness_units: Optional[np.ndarray] = None
    porosity: Optional[float] = None
    pore_diams_units: Optional[np.ndarray] = None
    void_mask: Optional[np.ndarray] = None
    pore_dt: Optional[np.ndarray] = None
    pores: Optional[Any] = None

    # Render cache (local only)
    labels_rgb_cache: tuple[Optional[int], Optional[np.ndarray]] = field(
        default_factory=lambda: (None, None),
        repr=False,
    )
    # Cached "Segmentation outline" visualization: (labels-id, outline-rgb-image).
    segmentation_outline_cache: tuple[Optional[int], Optional[np.ndarray]] = field(
        default_factory=lambda: (None, None),
        repr=False,
    )

    @classmethod
    def empty(
        cls,
        *,
        frame_id: str = "",
        z_index: int = 0,
        orig_img: Optional[np.ndarray] = None,
    ) -> "Frame":
        return cls(frame_id=str(frame_id), z_index=int(z_index), orig_img=orig_img)

    def clear_prediction(self) -> None:
        self.has_prediction = False
        self.dt = None
        self.dt_u8 = None
        self.dt_normalized = None
        self.lapl_img = None
        self.gx = None
        self.gy = None
        self.style_vec = None
        self.orig_img_scaled = None
        self.scale_factor = 1.0
        self.error_message = None

    def clear_segmentation(self) -> None:
        self.has_segmentation = False
        self.labels_2d = None
        self.centerlines_overlay_img = None
        self.overlay_img = None
        self.sampling_overlay = None
        self.measurements_overlay = None
        self.labels_rgb_cache = (None, None)
        self.segmentation_outline_cache = (None, None)

    def clear_measurements(self) -> None:
        self.has_measurements = False
        self.samples_table = None
        self.fiber_summary = None
        self.overlay_samples = None
        self.angles_samples_deg = None
        self.thickness_units = None
        self.porosity = None
        self.pore_diams_units = None
        self.void_mask = None
        self.pore_dt = None
        self.pores = None

    def begin_frame(self, orig_img: Optional[np.ndarray], frame_id: str, z_index: int) -> None:
        self.frame_id = str(frame_id)
        self.z_index = int(z_index)
        self.orig_img = orig_img
        self.clear_prediction()
        self.clear_segmentation()
        self.clear_measurements()
        self.error_message = None

    def set_prediction_outputs(
        self,
        *,
        dt: Optional[np.ndarray],
        gx: Optional[np.ndarray],
        gy: Optional[np.ndarray],
        dt_u8: Optional[np.ndarray],
        dt_normalized: Optional[np.ndarray],
        lapl_img: Optional[np.ndarray],
        style_vec: Optional[np.ndarray],
        orig_img_scaled: Optional[np.ndarray],
        scale_factor: float,
        invalidate_downstream: bool = True,
    ) -> None:
        self.dt = dt
        self.gx = gx
        self.gy = gy
        self.dt_u8 = dt_u8
        self.dt_normalized = dt_normalized
        self.lapl_img = lapl_img
        self.style_vec = style_vec
        self.orig_img_scaled = orig_img_scaled
        self.scale_factor = float(scale_factor)
        self.has_prediction = self.dt is not None
        self.error_message = None
        if invalidate_downstream:
            self.clear_segmentation()
            self.clear_measurements()

    def set_segmentation_outputs(
        self,
        *,
        labels_2d: Optional[np.ndarray],
        centerlines_overlay_img: Optional[np.ndarray],
        overlay_img: Optional[np.ndarray],
        sampling_overlay: Optional[np.ndarray] = None,
        measurements_overlay: Optional[np.ndarray] = None,
        labels_rgb_cache: Optional[tuple[Optional[int], Optional[np.ndarray]]] = None,
        segmentation_outline_cache: Optional[tuple[Optional[int], Optional[np.ndarray]]] = None,
        invalidate_measurements: bool = True,
    ) -> None:
        self.labels_2d = labels_2d
        self.centerlines_overlay_img = centerlines_overlay_img
        self.overlay_img = overlay_img
        self.sampling_overlay = sampling_overlay
        self.measurements_overlay = measurements_overlay
        if labels_rgb_cache is not None:
            self.labels_rgb_cache = labels_rgb_cache
        elif labels_2d is None:
            self.labels_rgb_cache = (None, None)
        if segmentation_outline_cache is not None:
            self.segmentation_outline_cache = segmentation_outline_cache
        elif labels_2d is None:
            self.segmentation_outline_cache = (None, None)
        self.has_segmentation = self.labels_2d is not None
        self.error_message = None
        if invalidate_measurements:
            self.clear_measurements()

    def set_measurement_outputs(
        self,
        *,
        samples_table: Optional[Any],
        fiber_summary: Optional[Dict[str, Any]],
        overlay_samples: Optional[Any],
        angles_samples_deg: Optional[np.ndarray],
        thickness_units: Optional[np.ndarray],
        porosity: Optional[float],
        pore_diams_units: Optional[np.ndarray],
        void_mask: Optional[np.ndarray],
        pore_dt: Optional[np.ndarray],
        pores: Optional[Any],
        measurements_overlay: Optional[np.ndarray],
        sampling_overlay: Optional[np.ndarray],
    ) -> None:
        self.samples_table = samples_table
        self.fiber_summary = fiber_summary
        self.overlay_samples = overlay_samples
        self.angles_samples_deg = angles_samples_deg
        self.thickness_units = thickness_units
        self.porosity = porosity
        self.pore_diams_units = pore_diams_units
        self.void_mask = void_mask
        self.pore_dt = pore_dt
        self.pores = pores
        self.measurements_overlay = measurements_overlay
        self.sampling_overlay = sampling_overlay
        self.has_measurements = True
        self.error_message = None

    def fail(self, message: str) -> None:
        self.error_message = str(message)

    def snapshot(
        self,
        *,
        include_measurements: bool = True,
        frame_id: Optional[str] = None,
        z_index: Optional[int] = None,
    ) -> "Frame":
        clone: Frame = copy(self)
        if clone.fiber_summary is not None:
            clone.fiber_summary = dict(clone.fiber_summary)
        if frame_id is not None:
            clone.frame_id = str(frame_id)
        if z_index is not None:
            clone.z_index = int(z_index)
        if not include_measurements:
            clone.clear_measurements()
        return clone


@dataclass
class FiberFragment:
    """Polyline fragment with two endpoints used for stitching and segmentation."""

    fragment_id: int
    centerline_xy: np.ndarray
    radii: Optional[np.ndarray] = None
    parent_fiber_id: int = -1
    _endpoints: Tuple["FragmentEndpoint", "FragmentEndpoint"] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.centerline_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("centerline_xy must be an array of shape (N, 2).")
        if pts.shape[0] < 2:
            raise ValueError("Each fragment must contain at least two points.")
        self.centerline_xy = pts

        if self.radii is None:
            radii = np.zeros(len(pts), dtype=np.float32)
        else:
            radii = np.asarray(self.radii, dtype=np.float32).reshape(-1)
            if radii.shape[0] != pts.shape[0]:
                raise ValueError("radii length must match centerline length.")
        self.radii = radii
        self.parent_fiber_id = int(self.parent_fiber_id)

        self._endpoints = (
            FragmentEndpoint(self, "a"),
            FragmentEndpoint(self, "b"),
        )

    def endpoints(self) -> Tuple["FragmentEndpoint", "FragmentEndpoint"]:
        return self._endpoints

    @property
    def points_xy(self) -> np.ndarray:
        return self.centerline_xy

    def length(self) -> float:
        if len(self.centerline_xy) < 2:
            return 0.0
        diffs = np.diff(self.centerline_xy, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())


class FragmentEndpoint:
    """Endpoint with local direction estimation suitable for stitching heuristics."""

    __slots__ = ("fragment", "end", "connected_to", "dir", "_fit_N")

    def __init__(self, fragment: FiberFragment, end: str, fit_N: int = 100) -> None:
        self.fragment = fragment
        if end not in ("a", "b"):
            raise ValueError("Endpoint end must be either 'a' or 'b'.")
        self.end = end
        self.connected_to: Optional["FragmentEndpoint"] = None
        self._fit_N = int(max(2, fit_N))
        self.dir = self._estimate_direction()

    def _points_for_fit(self) -> np.ndarray:
        pts = self.fragment.centerline_xy
        n = min(len(pts), self._fit_N)
        if self.end == "a":
            return pts[:n].copy()
        return pts[-n:][::-1].copy()

    def _estimate_direction(self) -> np.ndarray:
        pts = self._points_for_fit()
        mu = pts.mean(axis=0)
        centered = pts - mu
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            direction = Vt[0]
        except np.linalg.LinAlgError:
            direction = pts[-1] - pts[0]
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        seed = pts[0] - pts[1] if len(pts) >= 2 else np.array([1.0, 0.0], np.float32)
        if np.dot(direction, seed) < 0:
            direction = -direction
        return direction.astype(np.float32)

    def index(self) -> int:
        return 0 if self.end == "a" else len(self.fragment.centerline_xy) - 1

    def pos(self) -> np.ndarray:
        return self.fragment.centerline_xy[self.index()]

    def shifted_pos(self, offset_px: float = 0.0) -> np.ndarray:
        """
        Return a point offset from the endpoint along the fragment by offset_px.
        Falls back to the true endpoint if the fragment is too short.
        """
        offset_px = float(max(0.0, offset_px))
        if offset_px <= 0.0:
            return self.pos()

        pts = self.fragment.centerline_xy
        # travel inward from this endpoint
        if self.end == "a":
            walk_pts = pts
        else:
            walk_pts = pts[::-1]

        if len(walk_pts) < 2:
            return self.pos()

        seg_vecs = np.diff(walk_pts, axis=0)
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        accum = 0.0
        for i, seg_len in enumerate(seg_lens):
            next_accum = accum + float(seg_len)
            if offset_px <= next_accum and seg_len > 0:
                t = (offset_px - accum) / seg_len
                return (1.0 - t) * walk_pts[i] + t * walk_pts[i + 1]
            accum = next_accum

        # ran out of length; return the far end in walked order (which is the same coordinate space)
        return walk_pts[-1]

    def radius(self) -> float:
        return float(self.fragment.radii[self.index()])

    def last_n_points(self, n: int) -> np.ndarray:
        pts = self.fragment.centerline_xy
        if self.end == "a":
            return pts[: min(n, len(pts))].copy()
        return pts[-min(n, len(pts)) :][::-1].copy()

    def same_fragment(self, other: "FragmentEndpoint") -> bool:
        return self.fragment.fragment_id == other.fragment.fragment_id

    def is_facing(
        self,
        other: "FragmentEndpoint",
        *,
        cos_min: float = 0.95,
    ) -> bool:
        p1 = self.pos().astype(np.float32)
        p2 = other.pos().astype(np.float32)
        v = p2 - p1
        d = float(np.linalg.norm(v))
        if d < 1e-9:
            return False
        u12 = v / d
        u21 = -u12
        d1 = self.dir / (np.linalg.norm(self.dir) + 1e-12)
        d2 = other.dir / (np.linalg.norm(other.dir) + 1e-12)
        return (float(np.dot(d1, u12)) >= cos_min) and (float(np.dot(d2, u21)) >= cos_min)

    def disconnect(self) -> None:
        if self.connected_to is not None:
            other = self.connected_to
            self.connected_to = None
            if other.connected_to is self:
                other.connected_to = None

    def connect(self, other: "FragmentEndpoint") -> None:
        if other is self or self.same_fragment(other):
            return
        if self.connected_to is other and other.connected_to is self:
            return
        self.disconnect()
        other.disconnect()
        self.connected_to = other
        other.connected_to = self
