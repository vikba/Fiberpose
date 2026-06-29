"""Core data models for configuration, batches, frame outputs, and fiber geometry."""

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
    """Store configuration values for app."""
    model_path: str = ""
    use_gpu: bool = True
    scale_label: str = "1x"
    
    # --- Centerline Params ---
    max_angle_deg: float = 30.0
    ridge_sigma: float = 1.0
    ridge_low: float = 0.01
    ridge_high: float = 0.4
    min_fragment_centerline_length_px: float = 10.0
    measurement_centerline_length_trim_fraction: float = 0.05
    grad_quality_trim_fraction: float = 0.0
    strict_centerlines: bool = False
    keep_segments_without_measurement_centerlines: bool = False
    
    # --- Segmentation Params ---
    segmentation_mode: str = "watershed"
    merge_expansion_factor: float = 0.7
    segmentation_expansion_factor: float = 1.0
    seg_threshold: int = 1
    min_segment_foreground_mean: float = 0.5
    
    # --- Analysis & Physical Units ---
    sample_spacing: float = 5.0

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
            "foreground": True,
            "labels": True,
            "overlay": True,
            "diameters": True,
            "orientations": True,
            "outline": False,
            "tables": True,
            "image_format": "png",
            "collision": "suffix",
            "write_manifest": True,
        }
    )

    def save_to_json(self, filepath: str):
        """Persist to json using the configured export conventions.
        
        Args:
            filepath (str): Filesystem location used by this operation."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, strict: bool = False) -> "AppConfig":
        """Normalize a config mapping into AppConfig, with backward compatibility."""
        if not isinstance(data, dict):
            raise ValueError("Config JSON root must be an object.")
        data = dict(data)
        if "radial_expansion_factor" in data and "merge_expansion_factor" not in data:
            data["merge_expansion_factor"] = data["radial_expansion_factor"]
        valid_keys = set(cls.__dataclass_fields__)
        unknown_keys = sorted(set(data.keys()) - valid_keys)
        if unknown_keys:
            if strict:
                raise ValueError(f"Unknown config key(s): {', '.join(unknown_keys)}")
            data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**data)

    @classmethod
    def load_from_json(cls, filepath: str, *, strict: bool = False) -> "AppConfig":
        """Load from json from storage and normalize it for use.
        
        Args:
            filepath (str): Filesystem location describing where filepath is read or written.
            strict (bool): Boolean flag that toggles strict behavior.
        
        Returns:
            'AppConfig': Result object produced by the load from json routine.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data, strict=strict)
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
        """Provide an internal helper for as positive.
        
        Args:
            value (Any): Value for value interpreted by this routine.
            default (float): Value for default interpreted by this routine.
        
        Returns:
            float: Result object produced by the as positive routine."""
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
        """Orchestrate defaults within the current module workflow.
        
        Returns:
            'SampleProperties': Result object produced by the defaults routine."""
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
        """Construct an instance from image metadata while validating inputs.
        
        Args:
            metadata (Optional[Dict[str, Any]]): Value for metadata interpreted by this routine.
            source_path (str): Filesystem location describing where source path is read or written.
        
        Returns:
            'SampleProperties': Result object produced by the from image metadata routine."""
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
        """Orchestrate effective pixel size xy within the current module workflow.
        
        Returns:
            float: Result object produced by the effective pixel size xy routine."""
        x = self._as_positive(self.pixel_size_x)
        y = self._as_positive(self.pixel_size_y)
        return 0.5 * (x + y)

    def spacing_zyx(self) -> tuple[float, float, float]:
        """Orchestrate spacing zyx within the current module workflow.
        
        Returns:
            tuple[float, float, float]: Tuple grouping the spacing zyx outputs in a stable positional order."""
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
        """Report how many items are currently tracked by this container.
        
        Returns:
            int: Number of items currently available in the container."""
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
            """Provide an internal helper for positive.
            
            Args:
                value (Any): Value for value interpreted by this routine.
            
            Returns:
                Optional[float]: Result object produced by the positive routine."""
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
            """Provide an internal helper for unit scale to um.
            
            Args:
                unit_value (Any): Value for unit value interpreted by this routine.
            
            Returns:
                Optional[float]: Result object produced by the unit scale to um routine."""
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
        """Provide an internal helper for normalize stack.
        
        Args:
            data (np.ndarray): Value for data interpreted by this routine.
        
        Returns:
            List[np.ndarray]: Array containing the normalize stack output aligned with downstream expectations.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
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
        """Construct an instance from path while validating inputs.
        
        Args:
            path (str): Filesystem location describing where path is read or written.
        
        Returns:
            'ImageBatch': Result object produced by the from path routine."""
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
        """Construct an instance from folder while validating inputs.
        
        Args:
            folder (str): Filesystem location describing where folder is read or written.
            extensions (Optional[Iterable[str]]): Value for extensions interpreted by this routine.
        
        Returns:
            'ImageBatch': Result object produced by the from folder routine."""
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
            for image_frame in image_frames:
                frames.append(image_frame)
                filenames.append(name)
                    
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
        """Construct an instance from source while validating inputs.
        
        Args:
            path (str): Filesystem location describing where path is read or written.
            extensions (Optional[Iterable[str]]): Value for extensions interpreted by this routine.
        
        Returns:
            'ImageBatch': Result object produced by the from source routine.
        
        Raises:
            FileNotFoundError: If an expected file or directory cannot be located."""
        if os.path.isdir(path):
            return cls.from_folder(path, extensions=extensions)
        if os.path.isfile(path):
            return cls.from_path(path)
        raise FileNotFoundError(f"Source path does not exist: {path}")
    
@dataclass
class BatchResult:
    """Capture outputs and metadata produced by batch."""
    frame_results: List[Frame]

    def frame_at(self, index: int) -> Frame:
        """Orchestrate frame at within the current module workflow.
        
        Args:
            index (int): Zero-based index selecting which index to operate on.
        
        Returns:
            Frame: Result object produced by the frame at routine."""
        return self.frame_results[index]

    def save_all(self, output_dir: str, export_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Persist all using the configured export conventions.
        
        Args:
            output_dir (str): Filesystem location used by this operation.
            export_config (Dict[str, Any] | None): Configuration values controlling runtime behavior.
        
        Returns:
            Dict[str, Any]: Mapping of computed keys and values."""
        from .io_export import BatchResultExporter

        exporter = BatchResultExporter(export_config=export_config)
        return exporter.save_all(self, output_dir, export_config=export_config)

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
    prediction_sig: Optional[str] = None
    segmentation_centerline_sig: Optional[str] = None
    measurement_centerline_sig: Optional[str] = None
    segmentation_sig: Optional[str] = None
    measurement_sig: Optional[str] = None
    error_message: Optional[str] = None

    # Source and prediction buffers
    orig_img: Optional[np.ndarray] = None
    orig_img_scaled: Optional[np.ndarray] = None
    scale_factor: float = 1.0
    dt: Optional[np.ndarray] = None
    dt_u8: Optional[np.ndarray] = None
    dt_normalized: Optional[np.ndarray] = None
    foreground_prob: Optional[np.ndarray] = None
    lapl_img: Optional[np.ndarray] = None
    gx: Optional[np.ndarray] = None
    gy: Optional[np.ndarray] = None

    # Segmentation products
    labels_2d: Optional[np.ndarray] = None
    segmentation_centerlines_overlay_img: Optional[np.ndarray] = None
    measurement_centerlines_overlay_img: Optional[np.ndarray] = None
    centerlines_comparison_overlay_img: Optional[np.ndarray] = None
    overlay_img: Optional[np.ndarray] = None
    sampling_overlay: Optional[np.ndarray] = None
    diameters_overlay: Optional[np.ndarray] = None
    orientations_overlay: Optional[np.ndarray] = None
    pores_overlay: Optional[np.ndarray] = None

    # Measurement outputs
    samples_table: Optional[Any] = None
    fiber_summary: Optional[Dict[str, Any]] = None
    measurement_samples: Optional[Any] = None
    drawing_samples: Optional[Any] = None
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
        """Orchestrate empty within the current module workflow.
        
        Args:
            frame_id (str): Array-like data carrying frame id values for analysis.
            z_index (int): Coordinate component representing the z index axis value.
            orig_img (Optional[np.ndarray]): Value for orig img interpreted by this routine.
        
        Returns:
            'Frame': Result object produced by the empty routine."""
        return cls(frame_id=str(frame_id), z_index=int(z_index), orig_img=orig_img)

    def clear_prediction(self) -> None:
        """Clear cached prediction state that is no longer valid.
        
        Returns:
            None: Result object produced by the clear prediction routine."""
        self.has_prediction = False
        self.dt = None
        self.dt_u8 = None
        self.dt_normalized = None
        self.foreground_prob = None
        self.lapl_img = None
        self.gx = None
        self.gy = None
        self.orig_img_scaled = None
        self.scale_factor = 1.0
        self.prediction_sig = None
        self.segmentation_centerline_sig = None
        self.measurement_centerline_sig = None
        self.segmentation_sig = None
        self.measurement_sig = None
        self.error_message = None

    def clear_segmentation(self) -> None:
        """Clear cached segmentation state that is no longer valid.
        
        Returns:
            None: Result object produced by the clear segmentation routine."""
        self.has_segmentation = False
        self.labels_2d = None
        self.segmentation_centerlines_overlay_img = None
        self.measurement_centerlines_overlay_img = None
        self.centerlines_comparison_overlay_img = None
        self.overlay_img = None
        self.sampling_overlay = None
        self.diameters_overlay = None
        self.orientations_overlay = None
        self.pores_overlay = None
        self.labels_rgb_cache = (None, None)
        self.segmentation_outline_cache = (None, None)
        self.segmentation_centerline_sig = None
        self.measurement_centerline_sig = None
        self.segmentation_sig = None
        self.measurement_sig = None

    def clear_measurements(self) -> None:
        """Clear cached measurements state that is no longer valid.
        
        Returns:
            None: Result object produced by the clear measurements routine."""
        self.has_measurements = False
        self.samples_table = None
        self.fiber_summary = None
        self.measurement_samples = None
        self.drawing_samples = None
        self.thickness_units = None
        self.porosity = None
        self.pore_diams_units = None
        self.void_mask = None
        self.pore_dt = None
        self.pores = None
        self.diameters_overlay = None
        self.orientations_overlay = None
        self.pores_overlay = None
        self.measurement_sig = None

    def begin_frame(self, orig_img: Optional[np.ndarray], frame_id: str, z_index: int) -> None:
        """Start frame and initialize per-run state tracking.
        
        Args:
            orig_img (Optional[np.ndarray]): Value for orig img interpreted by this routine.
            frame_id (str): Array-like data carrying frame id values for analysis.
            z_index (int): Coordinate component representing the z index axis value.
        
        Returns:
            None: Result object produced by the begin frame routine."""
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
        foreground_prob: Optional[np.ndarray],
        lapl_img: Optional[np.ndarray],
        orig_img_scaled: Optional[np.ndarray],
        scale_factor: float,
        invalidate_downstream: bool = True,
    ) -> None:
        """Update prediction outputs and synchronize dependent runtime state.
        
        Args:
            dt (Optional[np.ndarray]): Value for dt interpreted by this routine.
            gx (Optional[np.ndarray]): Value for gx interpreted by this routine.
            gy (Optional[np.ndarray]): Value for gy interpreted by this routine.
            dt_u8 (Optional[np.ndarray]): Value for dt u8 interpreted by this routine.
            dt_normalized (Optional[np.ndarray]): Value for dt normalized interpreted by this routine.
            foreground_prob (Optional[np.ndarray]): Foreground probability predicted by the model, if available.
            lapl_img (Optional[np.ndarray]): Value for lapl img interpreted by this routine.
            orig_img_scaled (Optional[np.ndarray]): Numeric control parameter for orig img scaled during computation.
            scale_factor (float): Numeric control parameter for scale factor during computation.
            invalidate_downstream (bool): Value for invalidate downstream interpreted by this routine.
        
        Returns:
            None: Result object produced by the set prediction outputs routine."""
        self.dt = dt
        self.gx = gx
        self.gy = gy
        self.dt_u8 = dt_u8
        self.dt_normalized = dt_normalized
        self.foreground_prob = foreground_prob
        self.lapl_img = lapl_img
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
        segmentation_centerlines_overlay_img: Optional[np.ndarray],
        measurement_centerlines_overlay_img: Optional[np.ndarray],
        centerlines_comparison_overlay_img: Optional[np.ndarray],
        overlay_img: Optional[np.ndarray],
        sampling_overlay: Optional[np.ndarray] = None,
        labels_rgb_cache: Optional[tuple[Optional[int], Optional[np.ndarray]]] = None,
        segmentation_outline_cache: Optional[tuple[Optional[int], Optional[np.ndarray]]] = None,
        invalidate_measurements: bool = True,
    ) -> None:
        """Update segmentation outputs and synchronize dependent runtime state.
        
        Args:
            labels_2d (Optional[np.ndarray]): Array-like data carrying labels 2d values for analysis.
            centerlines_overlay_img (Optional[np.ndarray]): Value for centerlines overlay img interpreted by this routine.
            overlay_img (Optional[np.ndarray]): Value for overlay img interpreted by this routine.
            sampling_overlay (Optional[np.ndarray]): Value for sampling overlay interpreted by this routine.
            labels_rgb_cache (Optional[tuple[Optional[int], Optional[np.ndarray]]]): Array-like data carrying labels rgb cache values for analysis.
            segmentation_outline_cache (Optional[tuple[Optional[int], Optional[np.ndarray]]]): Value for segmentation outline cache interpreted by this routine.
            invalidate_measurements (bool): Value for invalidate measurements interpreted by this routine.
        
        Returns:
            None: Result object produced by the set segmentation outputs routine."""
        self.labels_2d = labels_2d
        self.segmentation_centerlines_overlay_img = segmentation_centerlines_overlay_img
        self.measurement_centerlines_overlay_img = measurement_centerlines_overlay_img
        self.centerlines_comparison_overlay_img = centerlines_comparison_overlay_img
        self.overlay_img = overlay_img
        self.sampling_overlay = sampling_overlay
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
        measurement_samples: Optional[Any],
        drawing_samples: Optional[Any],
        thickness_units: Optional[np.ndarray],
        porosity: Optional[float],
        pore_diams_units: Optional[np.ndarray],
        void_mask: Optional[np.ndarray],
        pore_dt: Optional[np.ndarray],
        pores: Optional[Any],
        sampling_overlay: Optional[np.ndarray],
    ) -> None:
        """Update measurement outputs and synchronize dependent runtime state."""
        self.samples_table = samples_table
        self.fiber_summary = fiber_summary
        self.measurement_samples = measurement_samples
        self.drawing_samples = drawing_samples
        self.thickness_units = thickness_units
        self.porosity = porosity
        self.pore_diams_units = pore_diams_units
        self.void_mask = void_mask
        self.pore_dt = pore_dt
        self.pores = pores
        self.sampling_overlay = sampling_overlay
        self.has_measurements = True
        self.error_message = None

    def fail(self, message: str) -> None:
        """Orchestrate fail within the current module workflow.
        
        Args:
            message (str): Value for message interpreted by this routine.
        
        Returns:
            None: Result object produced by the fail routine."""
        self.error_message = str(message)

    def snapshot(
        self,
        *,
        include_measurements: bool = True,
        frame_id: Optional[str] = None,
        z_index: Optional[int] = None,
    ) -> "Frame":
        """Orchestrate snapshot within the current module workflow.
        
        Args:
            include_measurements (bool): Value for include measurements interpreted by this routine.
            frame_id (Optional[str]): Array-like data carrying frame id values for analysis.
            z_index (Optional[int]): Coordinate component representing the z index axis value.
        
        Returns:
            'Frame': Result object produced by the snapshot routine."""
        clone: Frame = copy(self)
        def _clone_array(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.copy()
            return value

        clone.dt = _clone_array(clone.dt)
        clone.dt_u8 = _clone_array(clone.dt_u8)
        clone.dt_normalized = _clone_array(clone.dt_normalized)
        clone.foreground_prob = _clone_array(clone.foreground_prob)
        clone.lapl_img = _clone_array(clone.lapl_img)
        clone.gx = _clone_array(clone.gx)
        clone.gy = _clone_array(clone.gy)

        clone.labels_2d = _clone_array(clone.labels_2d)
        clone.segmentation_centerlines_overlay_img = _clone_array(clone.segmentation_centerlines_overlay_img)
        clone.measurement_centerlines_overlay_img = _clone_array(clone.measurement_centerlines_overlay_img)
        clone.centerlines_comparison_overlay_img = _clone_array(clone.centerlines_comparison_overlay_img)
        clone.overlay_img = _clone_array(clone.overlay_img)
        clone.sampling_overlay = _clone_array(clone.sampling_overlay)
        clone.diameters_overlay = _clone_array(clone.diameters_overlay)
        clone.orientations_overlay = _clone_array(clone.orientations_overlay)
        clone.pores_overlay = _clone_array(clone.pores_overlay)

        clone.samples_table = _clone_array(clone.samples_table)
        clone.thickness_units = _clone_array(clone.thickness_units)
        clone.pore_diams_units = _clone_array(clone.pore_diams_units)
        clone.void_mask = _clone_array(clone.void_mask)
        clone.pore_dt = _clone_array(clone.pore_dt)

        labels_cache_id, labels_cache_img = clone.labels_rgb_cache
        clone.labels_rgb_cache = (labels_cache_id, _clone_array(labels_cache_img))
        outline_cache_id, outline_cache_img = clone.segmentation_outline_cache
        clone.segmentation_outline_cache = (outline_cache_id, _clone_array(outline_cache_img))

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
    """Unified 2D fragment carrying both segmentation and measurement centerlines."""

    fragment_id: int
    segment_centerlines_xy: list[np.ndarray]
    segment_radii: Optional[list[np.ndarray]] = None
    parent_fiber_id: int = -1
    measurement_centerlines_xy: list[np.ndarray] = field(default_factory=list)
    measurement_radii: list[np.ndarray] = field(default_factory=list)
    measurement_source_indices: list[int] = field(default_factory=list)
    total_measurement_centerline_length_px: float = 0.0
    grad_quality_score: float = float("nan")

    @staticmethod
    def _normalize_centerline(points_raw: np.ndarray, label: str) -> np.ndarray:
        points_xy = np.asarray(points_raw, dtype=np.float32)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError(f"{label} must be an array of shape (N, 2).")
        if points_xy.shape[0] < 2:
            raise ValueError(f"{label} must contain at least two points.")
        return points_xy

    @staticmethod
    def _normalize_radii(radii_raw: Optional[np.ndarray], n_points: int, label: str) -> np.ndarray:
        if radii_raw is None:
            return np.zeros(n_points, dtype=np.float32)
        radii = np.asarray(radii_raw, dtype=np.float32).reshape(-1)
        if radii.shape[0] != n_points:
            raise ValueError(f"{label} length must match centerline length.")
        return radii

    @staticmethod
    def _centerline_length(points_xy: np.ndarray) -> float:
        if points_xy.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(points_xy, axis=0), axis=1).sum())

    def recompute_total_measurement_centerline_length_px(self) -> float:
        self.total_measurement_centerline_length_px = float(
            sum(self._centerline_length(points_xy) for points_xy in self.measurement_centerlines_xy)
        )
        return self.total_measurement_centerline_length_px

    def __post_init__(self) -> None:
        if not self.segment_centerlines_xy:
            raise ValueError("FiberFragment must contain at least one segmentation centerline.")

        centerlines_xy = [
            self._normalize_centerline(points_raw, "Segmentation centerline")
            for points_raw in self.segment_centerlines_xy
        ]
        if not self.segment_radii:
            centerlines_radii = [
                np.zeros(points_xy.shape[0], dtype=np.float32) for points_xy in centerlines_xy
            ]
        else:
            if len(self.segment_radii) != len(centerlines_xy):
                raise ValueError("segment_radii length must match segment_centerlines_xy length.")
            centerlines_radii = [
                self._normalize_radii(radii_raw, points_xy.shape[0], "Segmentation radii")
                for points_xy, radii_raw in zip(centerlines_xy, self.segment_radii)
            ]

        pairs = list(zip(centerlines_xy, centerlines_radii))
        pairs.sort(key=lambda pair: self._centerline_length(pair[0]), reverse=True)
        self.segment_centerlines_xy = [points_xy for points_xy, _ in pairs]
        self.segment_radii = [radii for _, radii in pairs]

        measurement_centerlines_xy = [
            self._normalize_centerline(points_raw, "Measurement centerline")
            for points_raw in (self.measurement_centerlines_xy or [])
        ]
        if not measurement_centerlines_xy:
            if self.measurement_radii:
                raise ValueError("measurement_radii requires measurement_centerlines_xy.")
            if self.measurement_source_indices:
                raise ValueError("measurement_source_indices requires measurement_centerlines_xy.")
            self.measurement_centerlines_xy = []
            self.measurement_radii = []
            self.measurement_source_indices = []
            self.total_measurement_centerline_length_px = 0.0
        else:
            if not self.measurement_radii:
                measurement_radii = [
                    np.zeros(points_xy.shape[0], dtype=np.float32)
                    for points_xy in measurement_centerlines_xy
                ]
            else:
                if len(self.measurement_radii) != len(measurement_centerlines_xy):
                    raise ValueError("measurement_radii length must match measurement_centerlines_xy length.")
                measurement_radii = [
                    self._normalize_radii(radii_raw, points_xy.shape[0], "Measurement radii")
                    for points_xy, radii_raw in zip(measurement_centerlines_xy, self.measurement_radii)
                ]
            if len(self.measurement_source_indices) != len(measurement_centerlines_xy):
                raise ValueError(
                    "measurement_source_indices length must match measurement_centerlines_xy length."
                )
            measurement_source_indices: list[int] = []
            for source_index_raw in self.measurement_source_indices:
                source_index = int(source_index_raw)
                if source_index < 0 or source_index >= len(self.segment_centerlines_xy):
                    raise ValueError(
                        "measurement_source_indices entries must reference segmentation centerlines."
                    )
                measurement_source_indices.append(source_index)
            self.measurement_centerlines_xy = measurement_centerlines_xy
            self.measurement_radii = measurement_radii
            self.measurement_source_indices = measurement_source_indices
            self.recompute_total_measurement_centerline_length_px()

        self.fragment_id = int(self.fragment_id)
        self.parent_fiber_id = int(self.parent_fiber_id)

    @property
    def has_measurement_centerline(self) -> bool:
        return bool(self.measurement_centerlines_xy)

    @property
    def points_xy(self) -> np.ndarray:
        return self.measurement_centerlines_xy[0] if self.measurement_centerlines_xy else self.segment_centerlines_xy[0]

    @property
    def centerline_xy(self) -> np.ndarray:
        return self.points_xy

    @property
    def radii(self) -> np.ndarray:
        return self.measurement_radii[0] if self.measurement_radii else self.segment_radii[0]

    def length(self) -> float:
        return self._centerline_length(self.points_xy)
