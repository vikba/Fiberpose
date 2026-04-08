from __future__ import annotations

import json
import os
from typing import Any

import imageio.v3 as iio
import numpy as np

from .utils import label_to_rgb


class BatchResultExporter:
    """Persist batch/frame outputs using a small dict-based export config."""

    _ALL_ARTIFACTS = ("dt", "labels", "overlay", "debug", "samples", "style_vector")
    _PRESET_ARTIFACTS = {
        "predictions": {"dt", "labels", "overlay", "debug"},
        "measurements": {"samples"},
        "full": {"dt", "labels", "overlay", "debug", "samples", "style_vector"},
    }

    def __init__(self, export_config: dict[str, Any] | None = None):
        self.export_config = dict(export_config or {})

    def save_all(
        self,
        batch_result,
        output_dir: str,
        export_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self._normalize_export_config(export_config or self.export_config)
        artifacts = self._resolve_artifacts(cfg)

        os.makedirs(output_dir, exist_ok=True)

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
                        iio.imwrite(path, self._labels_to_color(frame.labels_2d))
                        written_files.append(path)
                    elif artifact == "overlay":
                        if frame.overlay_img is None:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_overlay", image_ext, cfg)
                        iio.imwrite(path, np.asarray(frame.overlay_img, dtype=np.uint8))
                        written_files.append(path)
                    elif artifact == "debug":
                        if frame.measurements_overlay is None:
                            skipped.append(self._skip_item(frame_index, artifact, "missing_data"))
                            continue
                        path = self._resolve_output_path(output_dir, base_name, "_debug", image_ext, cfg)
                        iio.imwrite(path, np.asarray(frame.measurements_overlay, dtype=np.uint8))
                        written_files.append(path)
                    elif artifact == "samples":
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
                            "frame_id": str(getattr(frame, "frame_id", "")),
                            "artifact": artifact,
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

    def _normalize_export_config(self, export_config: dict[str, Any]) -> dict[str, Any]:
        cfg = {
            "preset": "full",
            "image_format": "png",
            "naming": "frame_id_z",
            "collision": "suffix",
            "include": [],
            "exclude": [],
            "base_name": None,
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
        cfg["include"] = [str(item) for item in include]
        cfg["exclude"] = [str(item) for item in exclude]
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

        naming = cfg.get("naming", "frame_id_z")
        frame_id = self._sanitize_name(os.path.splitext(os.path.basename(str(getattr(frame, "frame_id", ""))))[0])

        if naming == "index":
            return f"frame_{index:04d}"
        if not frame_id:
            frame_id = f"frame_{index:04d}"
        if naming == "frame_id":
            return frame_id

        z_index_raw = getattr(frame, "z_index", index)
        try:
            z_index = int(z_index_raw)
        except Exception:
            z_index = int(index)
        return f"{frame_id}_z{z_index:03d}"

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

    @staticmethod
    def _labels_to_color(labels: np.ndarray) -> np.ndarray:
        arr = np.asarray(labels, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError("labels must be a 2D array")
        h, w = arr.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        if arr.size == 0:
            return out
        for value in np.unique(arr):
            if int(value) == 0:
                continue
            mask = arr == value
            if not mask.any():
                continue
            out[mask] = label_to_rgb(int(value))
        return out

    def _write_samples_csv(self, frame, path: str) -> None:
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
            handle.write("x,y,diameter,angle_deg,fiber_id\n")

            thickness_units = getattr(frame, "thickness_units", None)
            for idx, row in enumerate(frame.samples_table, start=1):
                x, y, t_px = row[:3]
                angle_val = float(row[3]) if len(row) > 3 else float("nan")
                if thickness_units is not None and len(thickness_units) >= idx:
                    diameter_val = float(thickness_units[idx - 1])
                else:
                    diameter_val = float(t_px)
                handle.write(f"{int(x)},{int(y)},{diameter_val:.6f},{angle_val:.6f},0\n")

    def _measurement_summary_from_frame(
        self, frame
    ) -> tuple[float, float, float, float, float, float, int]:
        summary = getattr(frame, "fiber_summary", None)
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
            if getattr(frame, "thickness_units", None) is not None
            else np.array([])
        )
        angles = (
            np.asarray(frame.angles_samples_deg)
            if getattr(frame, "angles_samples_deg", None) is not None
            else np.array([])
        )

        mean_val = float(np.mean(diameters)) if diameters.size else float("nan")
        std_val = float(np.std(diameters)) if diameters.size else float("nan")
        sample_count = int(frame.samples_table.shape[0]) if frame.samples_table is not None else int(diameters.size)
        orientation_index, mean_angle_deg, std_angle_deg = self._compute_orientation_stats(angles)
        porosity = float(frame.porosity) if getattr(frame, "porosity", None) is not None else float("nan")
        return mean_val, std_val, orientation_index, mean_angle_deg, std_angle_deg, porosity, sample_count

    @staticmethod
    def _compute_orientation_stats(angles_samples_deg: np.ndarray) -> tuple[float, float, float]:
        if angles_samples_deg is None or getattr(angles_samples_deg, "size", 0) == 0:
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
