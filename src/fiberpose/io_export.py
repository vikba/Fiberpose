"""Export service and batch result exporter."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np

from .domain import AppConfig, BatchResult, Frame, ImageBatch
from .render import render


def frame_at_original_scale(frame: Frame) -> Frame:
    """Copy of `frame` with per-pixel outputs mapped from the processing scale back to `orig_img`'s size.

    Labels use nearest neighbour (ids never mixed); the DT is resampled and divided by the scale
    so its values are distances in original pixels.
    """
    scale = float(frame.scale_factor or 1.0)
    if scale == 1.0 or frame.orig_img is None or frame.dt is None:
        return frame
    (h, w), (sh, sw) = frame.orig_img.shape[:2], frame.dt.shape[:2]
    interp = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LINEAR
    resize = lambda a: None if a is None else cv2.resize(np.asarray(a, np.float32), (w, h), interpolation=interp)
    rows = np.minimum(((np.arange(h) + 0.5) * sh / h).astype(np.intp), sh - 1)
    cols = np.minimum(((np.arange(w) + 0.5) * sw / w).astype(np.intp), sw - 1)
    labels = frame.labels_2d
    return dataclasses.replace(
        frame,
        orig_img_scaled=None,
        scale_factor=1.0,
        dt=resize(frame.dt) / scale,
        foreground_prob=resize(frame.foreground_prob),
        labels_2d=None if labels is None else labels[rows[:, None], cols],
        # measurement_samples already hold the same tuples in original coordinates.
        drawing_samples=frame.measurement_samples,
        views={},
    )


class ExportService:
    """Pure export helpers for predictions/results tables."""

    @staticmethod
    def default_save_options() -> dict[str, Any]:
        return dict(AppConfig().save_options)

    @staticmethod
    def save_artifact_keys() -> tuple[str, ...]:
        return ("dt", "foreground", "labels", "overlay", "diameters", "orientations", "outline", "tables")

    @classmethod
    def normalize_save_options(cls, raw_options: Any) -> dict[str, Any]:
        defaults = cls.default_save_options()
        if not isinstance(raw_options, dict):
            return defaults

        normalized = dict(defaults)
        for key in cls.save_artifact_keys():
            if key in raw_options:
                normalized[key] = bool(raw_options.get(key))

        image_format = str(raw_options.get("image_format", defaults["image_format"])).strip().lower()
        normalized["image_format"] = image_format if image_format in {"png", "tiff"} else defaults["image_format"]

        collision = str(raw_options.get("collision", defaults["collision"])).strip().lower()
        normalized["collision"] = collision if collision in {"suffix", "overwrite", "error"} else defaults["collision"]

        normalized["write_manifest"] = bool(raw_options.get("write_manifest", defaults["write_manifest"]))
        return normalized

    @staticmethod
    def current_save_layout(image_batch: ImageBatch | None) -> str:
        if image_batch is None:
            return "single"
        source_type = str((image_batch.metadata or {}).get("source_type", "")).strip().lower()
        if source_type == "folder":
            return "frames"
        if source_type == "stack":
            return "stack"
        return "single"

    @staticmethod
    def current_source_base_name(image_batch: ImageBatch | None, current_image_name: str | None) -> str:
        if image_batch is not None:
            source_path = str((image_batch.metadata or {}).get("source_path", "")).strip()
            if source_path:
                return os.path.splitext(os.path.basename(os.path.normpath(source_path)))[0] or "fiberpose"
        return current_image_name or "fiberpose"

    @classmethod
    def artifact_exclude_from_save_options(cls, options: dict[str, Any]) -> list[str]:
        return [name for name in cls.save_artifact_keys() if not bool(options.get(name, False))]

    @staticmethod
    def effective_batch_result_for_save(
        *,
        layout: str,
        image_batch: ImageBatch | None,
        batch_result: BatchResult | None,
        current_frame: Frame,
    ) -> BatchResult | None:
        if layout != "single":
            if image_batch is None or batch_result is None:
                return None
            if len(batch_result.frame_results) != len(image_batch):
                return None
            return batch_result

        if batch_result is not None and len(batch_result.frame_results) == 1:
            return batch_result

        if current_frame.orig_img is None:
            return None

        frame_result = current_frame.snapshot(
            frame_id=str(current_frame.frame_id),
            z_index=int(current_frame.z_index),
        )
        return BatchResult(frame_results=[frame_result])

    @classmethod
    def build_export_config(
        cls,
        *,
        kind: str,
        layout: str,
        save_options: dict[str, Any],
        source_base_name: str,
        base_name: str | None = None,
        image_format_override: str | None = None,
    ) -> dict[str, Any]:
        if kind not in {"results", "csv"}:
            raise ValueError(f"Unsupported save kind: {kind}")

        image_format = image_format_override or str(save_options.get("image_format", "png"))
        export_config: dict[str, Any] = {
            "preset": "full" if kind == "results" else "measurements",
            "collision": str(save_options.get("collision", "suffix")),
            "write_manifest": bool(save_options.get("write_manifest", True)),
            "layout": layout,
            "source_base_name": source_base_name,
        }
        if kind == "results":
            export_config["exclude"] = cls.artifact_exclude_from_save_options(save_options)

        if layout == "single":
            export_config.update(
                {
                    "naming": "frame_id",
                    "collision": "overwrite",
                    "base_name": base_name or source_base_name,
                    "write_manifest": False,
                }
            )
            if kind == "results":
                export_config["image_format"] = "tiff" if image_format == "tiff" else "png"
        elif layout == "frames":
            export_config["naming"] = "frame_id"
            if kind == "results":
                export_config["image_format"] = "tiff" if image_format == "tiff" else "png"
        elif layout == "stack":
            if base_name:
                export_config["base_name"] = str(base_name)
        else:
            raise ValueError(f"Unsupported save layout: {layout}")

        return export_config

    @staticmethod
    def has_export_data(kind: str, batch_result: BatchResult | None) -> bool:
        if batch_result is None:
            return False
        if kind == "results":
            return any(fr.has_prediction or fr.has_segmentation for fr in batch_result.frame_results)
        if kind == "csv":
            return any(
                fr.has_measurements and fr.samples_table is not None and len(fr.samples_table) > 0
                for fr in batch_result.frame_results
            )
        return False


class BatchResultExporter:
    """Persist batch/frame outputs using a small dict-based export config."""

    _ALL_ARTIFACTS = ("dt", "foreground", "labels", "overlay", "diameters", "orientations", "outline", "tables")
    _PRESET_ARTIFACTS = {
        "predictions": {"dt", "foreground", "labels", "overlay", "diameters", "orientations", "outline"},
        "measurements": {"tables"},
        "full": {"dt", "foreground", "labels", "overlay", "diameters", "orientations", "outline", "tables"},
    }
    # Array written for each non-table artifact; None means the frame lacks it.
    _ARRAYS = {
        "dt": lambda frame: frame.dt,
        "foreground": lambda frame: frame.foreground_prob,
        "labels": lambda frame: render(frame, "Segmentation") if frame.labels_2d is not None and frame.labels_2d.any() else None,
        "overlay": lambda frame: render(frame, "Overlay"),
        "diameters": lambda frame: render(frame, "Diameters"),
        "orientations": lambda frame: render(frame, "Orientations"),
        "outline": lambda frame: render(frame, "Segmentation outline"),
    }

    def __init__(self, export_config: dict[str, Any] | None = None):
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

        frames = list(batch_result.frame_results)
        if "tables" in artifacts and len(frames) > 1:
            try:
                summary_base = self._stack_base_name(frames, cfg)
                summary_path = self._resolve_output_path(output_dir, summary_base, "_summary", ".csv", cfg)
                self._write_batch_summary_csv(frames, summary_path)
                written_files.append(summary_path)
            except Exception as exc:
                errors.append(
                    {
                        "frame_index": -1,
                        "frame_id": str(self._stack_base_name(frames, cfg)),
                        "artifact": "summary",
                        "error": str(exc),
                    }
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
        written_files: list[str] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        image_ext = ".tiff" if cfg["image_format"] == "tiff" else ".png"

        for frame_index, frame in enumerate(batch_result.frame_results):
            frame = frame_at_original_scale(frame)
            base_name = self._make_base_name(frame, frame_index, cfg)
            for artifact in self._ALL_ARTIFACTS:
                if artifact not in artifacts:
                    continue
                try:
                    if artifact == "tables":
                        if frame.samples_table is None or len(frame.samples_table) == 0:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_samples", ".csv", cfg)
                        self._write_samples_csv(frame, path)
                        written_files.append(path)
                    else:
                        array = self._ARRAYS[artifact](frame)
                        if array is None:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        ext = ".tiff" if artifact in ("dt", "foreground") else image_ext
                        path = self._resolve_output_path(output_dir, base_name, f"_{artifact}", ext, cfg)
                        iio.imwrite(path, array)
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
        frames = [frame_at_original_scale(frame) for frame in batch_result.frame_results]
        if not frames:
            return written_files, skipped, errors

        base_name = self._stack_base_name(frames, cfg)

        for artifact in self._ALL_ARTIFACTS:
            if artifact not in artifacts:
                continue
            try:
                if artifact == "tables":
                    if not any(
                        frame.samples_table is not None and len(frame.samples_table) > 0
                        for frame in frames
                    ):
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, "_samples", ".csv", cfg)
                    self._write_stack_samples_csv(frames, path)
                    written_files.append(path)
                else:
                    arrays = [self._ARRAYS[artifact](frame) for frame in frames]
                    if any(array is None for array in arrays):
                        skipped.append(self._skip_item(-1, artifact, "missing_data"))
                        continue
                    path = self._resolve_output_path(output_dir, base_name, f"_{artifact}", ".tiff", cfg)
                    iio.imwrite(path, np.stack(arrays, axis=0))
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

    def _stack_base_name(self, frames: list[Any], cfg: dict[str, Any]) -> str:
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
                        f"{z_index},{frame_id},{int(x)},{int(y)},{diameter_val:.2f},{angle_val:.2f}\n"
                    )

    def _write_batch_summary_csv(self, frames: list[Any], path: str) -> None:
        """Write a one-row-per-frame summary table for multi-image batches."""
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame_index",
                    "z_index",
                    "frame_id",
                    "diam_mean",
                    "diam_sd",
                    "diam_mode",
                    "diam_median",
                    "orientation_index",
                    "mean_angle_deg",
                    "std_angle_deg",
                    "porosity",
                    "sample_count",
                    "fragments_with_centerlines",
                    "units",
                    "has_segmentation",
                    "has_measurements",
                ]
            )
            for frame_index, frame in enumerate(frames):
                diam_mean, diam_sd, diam_mode, diam_median, orientation_index, mean_angle_deg, std_angle_deg, porosity, sample_count = (
                    self._measurement_summary_from_frame(frame)
                )
                summary = frame.fiber_summary or {}
                fragments_with_centerlines = int(summary.get("fragments_with_centerlines", 0))
                units = str(summary.get("units", "px"))
                writer.writerow(
                    [
                        int(frame_index),
                        int(frame.z_index),
                        str(frame.frame_id),
                        f"{diam_mean:.2f}",
                        f"{diam_sd:.2f}",
                        f"{diam_mode:.2f}",
                        f"{diam_median:.2f}",
                        f"{orientation_index:.2f}",
                        f"{mean_angle_deg:.2f}",
                        f"{std_angle_deg:.2f}",
                        f"{porosity:.2f}",
                        int(sample_count),
                        fragments_with_centerlines,
                        units,
                        int(frame.has_segmentation),
                        int(frame.has_measurements),
                    ]
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
            "naming": "frame_id",
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

        naming = str(cfg.get("naming", "frame_id"))
        if naming not in {"frame_id", "index"}:
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
        artifacts = set(self._PRESET_ARTIFACTS[cfg["preset"]])
        for artifact in cfg.get("include", []):
            if artifact in self._ALL_ARTIFACTS:
                artifacts.add(artifact)
        for artifact in cfg.get("exclude", []):
            artifacts.discard(artifact)
        return artifacts

    def _make_base_name(self, frame, index: int, cfg: dict[str, Any]) -> str:
        explicit_base = cfg.get("base_name")
        if explicit_base:
            return self._sanitize_name(str(explicit_base))

        naming = cfg.get("naming", "frame_id")
        frame_id = self._sanitize_name(os.path.splitext(os.path.basename(str(frame.frame_id)))[0])

        if naming == "index":
            return f"frame_{index:04d}"
        if not frame_id:
            frame_id = f"frame_{index:04d}"
        return frame_id

    @staticmethod
    def _sanitize_name(value: str) -> str:
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
        diam_mean, diam_sd, diam_mode, diam_median, orientation_index, mean_angle_deg, std_angle_deg, porosity, sample_count = (
            self._measurement_summary_from_frame(frame)
        )

        with open(path, "w", newline="") as handle:
            summary = frame.fiber_summary or {}
            fragments_with_centerlines = int(summary.get("fragments_with_centerlines", 0))
            handle.write("metric,value\n")
            handle.write(f"diam_mean,{diam_mean:.2f}\n")
            handle.write(f"diam_sd,{diam_sd:.2f}\n")
            handle.write(f"diam_mode,{diam_mode:.2f}\n")
            handle.write(f"diam_median,{diam_median:.2f}\n")
            handle.write(f"orientation_index,{orientation_index:.2f}\n")
            handle.write(f"mean_angle_deg,{mean_angle_deg:.2f}\n")
            handle.write(f"std_angle_deg,{std_angle_deg:.2f}\n")
            handle.write(f"porosity,{porosity:.2f}\n")
            handle.write(f"sample_number,{sample_count}\n")
            handle.write(f"fragments_with_centerlines,{fragments_with_centerlines}\n")
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
                handle.write(f"{int(x)},{int(y)},{diameter_val:.2f},{angle_val:.2f}\n")

    def _measurement_summary_from_frame(
        self, frame
    ) -> tuple[float, float, float, float, float, float, float, float, int]:
        summary = frame.fiber_summary or {}
        return (
            float(summary.get("diam_mean", float("nan"))),
            float(summary.get("diam_sd", float("nan"))),
            float(summary.get("diam_mode", float("nan"))),
            float(summary.get("diam_median", float("nan"))),
            float(summary.get("orientation_index", float("nan"))),
            float(summary.get("mean_angle_deg", float("nan"))),
            float(summary.get("std_angle_deg", float("nan"))),
            float(summary.get("porosity", float("nan"))),
            int(summary.get("sample_count", 0)),
        )
