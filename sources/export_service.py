"""Export/save logic shared by presenter workflows."""

from __future__ import annotations

import os
from typing import Any

from .datastructures import AppConfig, BatchResult, Frame, ImageBatch


class ExportService:
    """Pure export helpers for predictions/results tables."""

    @staticmethod
    def default_save_options() -> dict[str, Any]:
        return dict(AppConfig().save_options)

    @staticmethod
    def save_artifact_keys() -> tuple[str, ...]:
        return ("dt", "labels", "overlay", "diameters", "orientations", "outline", "tables")

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
            include_measurements=True,
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
            export_config["naming"] = "frame_id_z"
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
