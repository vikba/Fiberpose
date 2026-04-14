"""Export helpers for serializing batch results, artifacts, and measurement tables."""

from __future__ import annotations

import json
import os
from typing import Any

import imageio.v3 as iio
import numpy as np

class BatchResultExporter:
    """Persist batch/frame outputs using a small dict-based export config."""

    _ALL_ARTIFACTS = ("dt", "labels", "overlay", "diameters", "orientations", "outline", "tables", "style_vector")
    _PRESET_ARTIFACTS = {
        "predictions": {"dt", "labels", "overlay", "diameters", "orientations", "outline"},
        "measurements": {"tables"},
        "full": {"dt", "labels", "overlay", "diameters", "orientations", "outline", "tables", "style_vector"},
    }

    def __init__(self, export_config: dict[str, Any] | None = None):
        """Initialize batch result exporter with validated runtime dependencies and defaults.
        
        Args:
            export_config (dict[str, Any] | None): Configuration values controlling runtime behavior."""
        self.export_config = dict(export_config or {})

    def save_all(
        self,
        batch_result,
        output_dir: str,
        export_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Export configured artifacts for all frames (or as stacks) and return a save manifest."""
        cfg = self._normalize_export_config(export_config or self.export_config)
        artifacts = self._resolve_artifacts(cfg)

        os.makedirs(output_dir, exist_ok=True)

        layout = str(cfg.get("layout", "frames")).strip().lower()
        if layout == "stack":
            written_files, skipped, errors = self._save_stack_artifacts(
                batch_result=batch_result,
                output_dir=output_dir,
                cfg=cfg,
                artifacts=artifacts,
            )
        else:
            written_files, skipped, errors = self._save_framewise_artifacts(
                batch_result=batch_result,
                output_dir=output_dir,
                cfg=cfg,
                artifacts=artifacts,
            )

        manifest_path = None
        if cfg.get("write_manifest", True):
            manifest = {
                "export_config": cfg,
                "written_files": [os.path.relpath(p, output_dir) for p in written_files],
                "written_count": len(written_files),
                "skipped": skipped,
                "errors": errors,
            }
            manifest_path = self._resolve_output_path(output_dir, "manifest", "", ".json", cfg)
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)

        return {
            "output_dir": output_dir,
            "written_files": written_files,
            "written_count": len(written_files),
            "skipped": skipped,
            "errors": errors,
            "manifest_path": manifest_path,
        }

    @staticmethod
    def _skip_item(frame_index: int, artifact: str, reason: str) -> dict[str, Any]:
        """Provide an internal helper for skip item.
        
        Args:
            frame_index (int): Array-like data carrying frame index values for analysis.
            artifact (str): Value for artifact interpreted by this routine.
            reason (str): Value for reason interpreted by this routine.
        
        Returns:
            dict[str, Any]: Mapping containing named skip item outputs and metadata."""
        return {
            "frame_index": int(frame_index),
            "artifact": artifact,
            "reason": reason,
        }

    def _save_framewise_artifacts(
        self,
        *,
        batch_result,
        output_dir: str,
        cfg: dict[str, Any],
        artifacts: set[str],
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        """Provide an internal helper for save framewise artifacts.
        
        Args:
            batch_result (Any): Value for batch result interpreted by this routine.
            output_dir (str): Filesystem location describing where output dir is read or written.
            cfg (dict[str, Any]): Value for cfg interpreted by this routine.
            artifacts (set[str]): Value for artifacts interpreted by this routine.
        
        Returns:
            tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]: Tuple grouping the save framewise artifacts outputs in a stable positional order."""
        written_files: list[str] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        image_ext = ".tiff" if cfg["image_format"] == "tiff" else ".png"

        for frame_index, frame in enumerate(batch_result.frame_results):
            base_name = self._make_base_name(frame, frame_index, cfg)
            for artifact in self._ALL_ARTIFACTS:
                if artifact not in artifacts:
                    continue
                try:
                    if artifact == "dt":
                        if frame.dt_u8 is None:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_dt", image_ext, cfg)
                        iio.imwrite(path, np.asarray(frame.dt_u8, dtype=np.uint8))
                        written_files.append(path)
                    elif artifact == "labels":
                        if frame.labels_2d is None or np.max(frame.labels_2d) <= 0:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_labels", image_ext, cfg)
                        iio.imwrite(path, np.asarray(frame.labels_2d, dtype=np.uint16))
                        written_files.append(path)
                    elif artifact == "overlay":
                        if frame.overlay_img is None:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_overlay", image_ext, cfg)
                        iio.imwrite(path, np.asarray(frame.overlay_img, dtype=np.uint8))
                        written_files.append(path)
                    elif artifact == "diameters":
                        src = frame.diameters_overlay
                        if src is None:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_diameters", image_ext, cfg)
                        iio.imwrite(path, np.asarray(src, dtype=np.uint8))
                        written_files.append(path)
                    elif artifact == "orientations":
                        src = frame.orientations_overlay
                        if src is None:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_orientations", image_ext, cfg)
                        iio.imwrite(path, np.asarray(src, dtype=np.uint8))
                        written_files.append(path)
                    elif artifact == "outline":
                        outline_img = self._cached_outline_image(frame)
                        if outline_img is None:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_outline", image_ext, cfg)
                        iio.imwrite(path, np.asarray(outline_img, dtype=np.uint8))
                        written_files.append(path)
                    elif artifact == "tables":
                        if frame.samples_table is None or len(frame.samples_table) == 0:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_samples", ".csv", cfg)
                        self._write_samples_csv(frame, path)
                        written_files.append(path)
                    elif artifact == "style_vector":
                        if frame.style_vec is None:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_style", ".npy", cfg)
                        np.save(path, np.asarray(frame.style_vec, dtype=np.float32))
                        written_files.append(path)
                except Exception as exc:
                    errors.append(
                        {
                            "frame_index": int(frame_index),
                            "frame_id": str(frame.frame_id),
                            "artifact": artifact,
                            "error": str(exc),
                        }
                    )

        return written_files, skipped, errors

    def _save_stack_artifacts(
        self,
        *,
        batch_result,
        output_dir: str,
        cfg: dict[str, Any],
        artifacts: set[str],
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Save stack sources as stack-level files.
        Image artifacts are written as multi-page TIFF stacks.
        """
        written_files: list[str] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        frames = list(batch_result.frame_results)
        if not frames:
            return written_files, skipped, errors

        base_name = self._stack_base_name(frames, cfg)

        for artifact in self._ALL_ARTIFACTS:
            if artifact not in artifacts:
                continue
            try:
                if artifact == "dt":
                    arrays = self._collect_stack_arrays(frames, artifact)
                    if arrays is None:
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, "_dt", ".tiff", cfg)
                    iio.imwrite(path, np.stack(arrays, axis=0))
                    written_files.append(path)
                elif artifact == "labels":
                    arrays = self._collect_stack_arrays(frames, artifact)
                    if arrays is None:
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    if not any(arr.size and int(np.max(arr)) > 0 for arr in arrays):
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, "_labels", ".tiff", cfg)
                    iio.imwrite(path, np.stack(arrays, axis=0))
                    written_files.append(path)
                elif artifact == "overlay":
                    arrays = self._collect_stack_arrays(frames, artifact)
                    if arrays is None:
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, "_overlay", ".tiff", cfg)
                    iio.imwrite(path, np.stack(arrays, axis=0))
                    written_files.append(path)
                elif artifact == "diameters":
                    arrays = self._collect_stack_arrays(frames, artifact)
                    if arrays is None:
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, "_diameters", ".tiff", cfg)
                    iio.imwrite(path, np.stack(arrays, axis=0))
                    written_files.append(path)
                elif artifact == "orientations":
                    arrays = self._collect_stack_arrays(frames, artifact)
                    if arrays is None:
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, "_orientations", ".tiff", cfg)
                    iio.imwrite(path, np.stack(arrays, axis=0))
                    written_files.append(path)
                elif artifact == "outline":
                    arrays = self._collect_stack_arrays(frames, artifact)
                    if arrays is None:
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, "_outline", ".tiff", cfg)
                    iio.imwrite(path, np.stack(arrays, axis=0))
                    written_files.append(path)
                elif artifact == "tables":
                    if not any(
                        frame.samples_table is not None and len(frame.samples_table) > 0
                        for frame in frames
                    ):
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, "_samples", ".csv", cfg)
                    self._write_stack_samples_csv(frames, path)
                    written_files.append(path)
                elif artifact == "style_vector":
                    if not all(frame.style_vec is not None for frame in frames):
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, "_style", ".npy", cfg)
                    stack = np.stack(
                        [np.asarray(frame.style_vec, dtype=np.float32) for frame in frames],
                        axis=0,
                    )
                    np.save(path, stack)
                    written_files.append(path)
            except Exception as exc:
                errors.append(
                    {
                        "frame_index": -1,
                        "frame_id": str(base_name),
                        "artifact": artifact,
                        "error": str(exc),
                    }
                )

        return written_files, skipped, errors

    def _collect_stack_arrays(self, frames: list[Any], artifact: str) -> list[np.ndarray] | None:
        """Provide an internal helper for collect stack arrays.
        
        Args:
            frames (list[Any]): Array-like data carrying frames values for analysis.
            artifact (str): Value for artifact interpreted by this routine.
        
        Returns:
            list[np.ndarray] | None: Array containing the collect stack arrays output aligned with downstream expectations."""
        arrays: list[np.ndarray] = []
        for frame in frames:
            if artifact == "dt":
                src = frame.dt_u8
                if src is None:
                    return None
                arrays.append(np.asarray(src, dtype=np.uint8))
            elif artifact == "labels":
                src = frame.labels_2d
                if src is None:
                    return None
                arr = np.asarray(src)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                if arr.ndim != 2:
                    return None
                arrays.append(np.asarray(arr, dtype=np.uint16))
            elif artifact == "overlay":
                src = frame.overlay_img
                if src is None:
                    return None
                arrays.append(np.asarray(src, dtype=np.uint8))
            elif artifact == "diameters":
                src = frame.diameters_overlay
                if src is None:
                    return None
                arrays.append(np.asarray(src, dtype=np.uint8))
            elif artifact == "orientations":
                src = frame.orientations_overlay
                if src is None:
                    return None
                arrays.append(np.asarray(src, dtype=np.uint8))
            elif artifact == "outline":
                src = self._cached_outline_image(frame)
                if src is None:
                    return None
                arrays.append(np.asarray(src, dtype=np.uint8))
            else:
                return None
        return arrays

    @staticmethod
    def _cached_outline_image(frame: Any) -> np.ndarray | None:
        """
        Return the cached outline visualization image.
        Saving must not recompute outlines.
        """
        cache = frame.segmentation_outline_cache
        if not isinstance(cache, tuple) or len(cache) != 2:
            return None
        cached_img = cache[1]
        if cached_img is None:
            return None
        return np.asarray(cached_img, dtype=np.uint8)

    def _stack_base_name(self, frames: list[Any], cfg: dict[str, Any]) -> str:
        """Provide an internal helper for stack base name.
        
        Args:
            frames (list[Any]): Array-like data carrying frames values for analysis.
            cfg (dict[str, Any]): Value for cfg interpreted by this routine.
        
        Returns:
            str: String value representing the normalized stack base name outcome."""
        explicit_base = cfg.get("base_name")
        if explicit_base:
            return self._sanitize_name(str(explicit_base))

        source_base_name = cfg.get("source_base_name")
        if source_base_name:
            return self._sanitize_name(str(source_base_name))

        first_id = str(frames[0].frame_id or "").strip()
        if first_id:
            return self._sanitize_name(os.path.splitext(os.path.basename(first_id))[0])
        return "stack"

    @staticmethod
    def _write_stack_samples_csv(frames: list[Any], path: str) -> None:
        """Provide an internal helper for write stack samples csv.
        
        Args:
            frames (list[Any]): Array-like data carrying frames values for analysis.
            path (str): Filesystem location describing where path is read or written.
        
        Returns:
            None: Result object produced by the write stack samples csv routine."""
        with open(path, "w", newline="") as handle:
            handle.write("z_index,frame_id,x,y,diameter,angle_deg\n")
            for frame in frames:
                if frame.samples_table is None or len(frame.samples_table) == 0:
                    continue
                z_index = int(frame.z_index)
                frame_id = str(frame.frame_id)
                thickness_units = frame.thickness_units
                for sample_index, row in enumerate(frame.samples_table, start=1):
                    x, y, t_px = row[:3]
                    angle_val = float(row[3]) if len(row) > 3 else float("nan")
                    if thickness_units is not None and len(thickness_units) >= sample_index:
                        diameter_val = float(thickness_units[sample_index - 1])
                    else:
                        diameter_val = float(t_px)
                    handle.write(
                        f"{z_index},{frame_id},{int(x)},{int(y)},{diameter_val:.6f},{angle_val:.6f}\n"
                    )

    def _normalize_export_config(self, export_config: dict[str, Any]) -> dict[str, Any]:
        """Normalize export config into a canonical representation.
        
        Args:
            export_config (dict[str, Any]): Configuration values controlling runtime behavior.
        
        Returns:
            dict[str, Any]: Mapping of computed keys and values.
        
        Raises:
            ValueError: If provided values fail domain validation checks."""
        cfg = {
            "preset": "full",
            "image_format": "png",
            "naming": "frame_id_z",
            "collision": "suffix",
            "layout": "frames",
            "include": [],
            "exclude": [],
            "base_name": None,
            "source_base_name": None,
            "write_manifest": True,
        }
        cfg.update(export_config)

        preset = str(cfg.get("preset", "full"))
        if preset not in self._PRESET_ARTIFACTS:
            raise ValueError(f"Unsupported preset: {preset}")
        cfg["preset"] = preset

        image_format = str(cfg.get("image_format", "png")).lower()
        if image_format not in {"png", "tiff"}:
            raise ValueError(f"Unsupported image_format: {image_format}")
        cfg["image_format"] = image_format

        naming = str(cfg.get("naming", "frame_id_z"))
        if naming not in {"frame_id", "frame_id_z", "index"}:
            raise ValueError(f"Unsupported naming: {naming}")
        cfg["naming"] = naming

        layout = str(cfg.get("layout", "frames")).lower()
        if layout not in {"single", "frames", "stack"}:
            raise ValueError(f"Unsupported layout: {layout}")
        cfg["layout"] = layout

        collision = str(cfg.get("collision", "suffix"))
        if collision not in {"suffix", "overwrite", "error"}:
            raise ValueError(f"Unsupported collision: {collision}")
        cfg["collision"] = collision

        include = cfg.get("include", [])
        exclude = cfg.get("exclude", [])
        if include is None:
            include = []
        if exclude is None:
            exclude = []
        cfg["include"] = [str(item).strip().lower() for item in include]
        cfg["exclude"] = [str(item).strip().lower() for item in exclude]
        cfg["source_base_name"] = (
            str(cfg.get("source_base_name")).strip() if cfg.get("source_base_name") is not None else None
        )
        cfg["write_manifest"] = bool(cfg.get("write_manifest", True))
        return cfg

    def _resolve_artifacts(self, cfg: dict[str, Any]) -> set[str]:
        """Provide an internal helper for resolve artifacts.
        
        Args:
            cfg (dict[str, Any]): Value for cfg interpreted by this routine.
        
        Returns:
            set[str]: Result object produced by the resolve artifacts routine."""
        artifacts = set(self._PRESET_ARTIFACTS[cfg["preset"]])
        for artifact in cfg.get("include", []):
            if artifact in self._ALL_ARTIFACTS:
                artifacts.add(artifact)
        for artifact in cfg.get("exclude", []):
            artifacts.discard(artifact)
        return artifacts

    def _make_base_name(self, frame, index: int, cfg: dict[str, Any]) -> str:
        """Provide an internal helper for make base name.
        
        Args:
            frame (Any): Array-like data carrying frame values for analysis.
            index (int): Zero-based index selecting which index to operate on.
            cfg (dict[str, Any]): Value for cfg interpreted by this routine.
        
        Returns:
            str: String value representing the normalized make base name outcome."""
        explicit_base = cfg.get("base_name")
        if explicit_base:
            return self._sanitize_name(str(explicit_base))

        naming = cfg.get("naming", "frame_id_z")
        frame_id = self._sanitize_name(os.path.splitext(os.path.basename(str(frame.frame_id)))[0])

        if naming == "index":
            return f"frame_{index:04d}"
        if not frame_id:
            frame_id = f"frame_{index:04d}"
        if naming == "frame_id":
            return frame_id

        z_index_raw = frame.z_index
        try:
            z_index = int(z_index_raw)
        except Exception:
            z_index = int(index)
        return f"{frame_id}_z{z_index:03d}"

    @staticmethod
    def _sanitize_name(value: str) -> str:
        """Provide an internal helper for sanitize name.
        
        Args:
            value (str): Value for value interpreted by this routine.
        
        Returns:
            str: String value representing the normalized sanitize name outcome."""
        clean = os.path.basename(value).strip()
        return clean if clean else "frame"

    def _resolve_output_path(
        self,
        output_dir: str,
        base_name: str,
        suffix: str,
        extension: str,
        cfg: dict[str, Any],
    ) -> str:
        """Provide an internal helper for resolve output path.
        
        Args:
            output_dir (str): Filesystem location describing where output dir is read or written.
            base_name (str): Value for base name interpreted by this routine.
            suffix (str): Value for suffix interpreted by this routine.
            extension (str): Value for extension interpreted by this routine.
            cfg (dict[str, Any]): Value for cfg interpreted by this routine.
        
        Returns:
            str: String value representing the normalized resolve output path outcome.
        
        Raises:
            FileExistsError: If the routine encounters an unrecoverable error condition."""
        candidate = os.path.join(output_dir, f"{base_name}{suffix}{extension}")
        policy = cfg.get("collision", "suffix")
        if policy == "overwrite" or not os.path.exists(candidate):
            return candidate
        if policy == "error":
            raise FileExistsError(f"Output file already exists: {candidate}")

        # suffix mode
        i = 2
        while True:
            next_candidate = os.path.join(output_dir, f"{base_name}{suffix}_{i}{extension}")
            if not os.path.exists(next_candidate):
                return next_candidate
            i += 1

    def _write_samples_csv(self, frame, path: str) -> None:
        """Provide an internal helper for write samples csv.
        
        Args:
            frame (Any): Array-like data carrying frame values for analysis.
            path (str): Filesystem location describing where path is read or written.
        
        Returns:
            None: Result object produced by the write samples csv routine."""
        mean_val, std_val, orientation_index, mean_angle_deg, std_angle_deg, porosity, sample_count = (
            self._measurement_summary_from_frame(frame)
        )

        with open(path, "w", newline="") as handle:
            handle.write("metric,value\n")
            handle.write(f"mean,{mean_val:.6f}\n")
            handle.write(f"std,{std_val:.6f}\n")
            handle.write(f"orientation_index,{orientation_index:.6f}\n")
            handle.write(f"mean_angle_deg,{mean_angle_deg:.6f}\n")
            handle.write(f"std_angle_deg,{std_angle_deg:.6f}\n")
            handle.write(f"porosity,{porosity:.6f}\n")
            handle.write(f"sample_number,{sample_count}\n")
            handle.write("\n\n")
            handle.write("x,y,diameter,angle_deg\n")

            thickness_units = frame.thickness_units
            for idx, row in enumerate(frame.samples_table, start=1):
                x, y, t_px = row[:3]
                angle_val = float(row[3]) if len(row) > 3 else float("nan")
                if thickness_units is not None and len(thickness_units) >= idx:
                    diameter_val = float(thickness_units[idx - 1])
                else:
                    diameter_val = float(t_px)
                handle.write(f"{int(x)},{int(y)},{diameter_val:.6f},{angle_val:.6f}\n")

    def _measurement_summary_from_frame(
        self, frame
    ) -> tuple[float, float, float, float, float, float, int]:
        """Provide an internal helper for measurement summary from frame.
        
        Args:
            frame (Any): Array-like data carrying frame values for analysis.
        
        Returns:
            tuple[float, float, float, float, float, float, int]: Tuple grouping the measurement summary from frame outputs in a stable positional order."""
        summary = frame.fiber_summary
        if summary:
            return (
                float(summary.get("mean_diameter", float("nan"))),
                float(summary.get("std_diameter", float("nan"))),
                float(summary.get("orientation_index", float("nan"))),
                float(summary.get("mean_angle_deg", float("nan"))),
                float(summary.get("std_angle_deg", float("nan"))),
                float(summary.get("porosity", float("nan"))),
                int(summary.get("sample_count", 0)),
            )

        diameters = (
            np.asarray(frame.thickness_units)
            if frame.thickness_units is not None
            else np.array([])
        )
        angles = (
            np.asarray([sample[5] for sample in (frame.measurement_samples or [])], dtype=np.float32)
            if frame.measurement_samples is not None
            else np.array([])
        )

        mean_val = float(np.mean(diameters)) if diameters.size else float("nan")
        std_val = float(np.std(diameters)) if diameters.size else float("nan")
        sample_count = int(frame.samples_table.shape[0]) if frame.samples_table is not None else int(diameters.size)
        orientation_index, mean_angle_deg, std_angle_deg = self._compute_orientation_stats(angles)
        porosity = float(frame.porosity) if frame.porosity is not None else float("nan")
        return mean_val, std_val, orientation_index, mean_angle_deg, std_angle_deg, porosity, sample_count

    @staticmethod
    def _compute_orientation_stats(angles_samples_deg: np.ndarray) -> tuple[float, float, float]:
        """Provide an internal helper for compute orientation stats.
        
        Args:
            angles_samples_deg (np.ndarray): Numeric control parameter for angles samples deg during computation.
        
        Returns:
            tuple[float, float, float]: Tuple grouping the compute orientation stats outputs in a stable positional order."""
        if angles_samples_deg is None or angles_samples_deg.size == 0:
            return float("nan"), float("nan"), float("nan")

        angles_rad = np.deg2rad(angles_samples_deg)
        mean_cos = float(np.mean(np.cos(angles_rad)))
        mean_sin = float(np.mean(np.sin(angles_rad)))

        orientation_index = float(np.clip(np.hypot(mean_cos, mean_sin), 0.0, 1.0))
        mean_angle_rad = float(np.arctan2(mean_sin, mean_cos))
        mean_angle_deg = float(np.rad2deg(mean_angle_rad))
        if mean_angle_deg > 90.0:
            mean_angle_deg -= 180.0
        elif mean_angle_deg <= -90.0:
            mean_angle_deg += 180.0

        if orientation_index > 0 and np.isfinite(orientation_index):
            r = float(np.clip(orientation_index, 1e-8, 1.0))
            std_angle_rad = float(np.sqrt(max(0.0, -2.0 * np.log(r))))
            std_angle_deg = float(np.rad2deg(std_angle_rad))
        else:
            std_angle_deg = float("nan")
        return orientation_index, mean_angle_deg, std_angle_deg
