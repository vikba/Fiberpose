"""CLI runner for batch-oriented Fiberpose processing."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Iterable

import imageio.v3 as iio
from .domain import AppConfig, BatchResult, Frame, ImageBatch, SampleProperties
from .models import resolve_model_path
from .paths import default_config_path

from tqdm import tqdm


class CliRunner:
    """Run segmentation/measurement batch processing from the command line."""

    def __init__(
        self,
        *,
        input_path: str,
        config_path: str | None = None,
        output_dir: str | None = None,
        compute_measurements: bool = False,
        invert_input: bool | None = None,
        save_kind: str = "results",
    ) -> None:
        self.input_path = str(input_path)
        self.config_path = os.path.abspath(config_path or default_config_path())
        self.output_dir = output_dir
        self.compute_measurements = bool(compute_measurements)
        self.invert_input = invert_input
        self.save_kind = str(save_kind)

    @staticmethod
    def _resolve_model_path(config: AppConfig) -> str:
        return resolve_model_path(config.model_path)

    @staticmethod
    def _resolve_device(config: AppConfig) -> str:
        from .ml import onnx_cuda_available

        if bool(config.use_gpu) and onnx_cuda_available():
            return "cuda"
        return "cpu"

    def _effective_output_dir(self) -> str:
        if self.output_dir:
            return os.path.abspath(self.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(os.getcwd(), f"fiberpose_cli_{timestamp}")

    @staticmethod
    def _supported_image_extensions() -> set[str]:
        return {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    @staticmethod
    def _frame_identity(image_batch: ImageBatch, index: int) -> tuple[str, int]:
        metadata = image_batch.metadata or {}
        filenames = metadata.get("filenames", [])
        z_indices = metadata.get("z_indices", [])

        frame_id = str(filenames[index]) if index < len(filenames) else str(index)
        z_index = int(z_indices[index]) if index < len(z_indices) else int(index)
        return frame_id, z_index

    @staticmethod
    def _outcome_message(outcome: Any) -> str:
        if outcome is None:
            return "unknown pipeline failure"
        if getattr(outcome, "error_message", None):
            return str(outcome.error_message)
        if getattr(outcome, "info_message", None):
            return str(outcome.info_message)
        return "pipeline stage failed"

    @staticmethod
    def _is_frame_success(frame: Frame, compute_measurements: bool) -> bool:
        return bool(frame.has_measurements) if compute_measurements else bool(frame.has_segmentation)

    @staticmethod
    def _progress(total: int, desc: str):
        return tqdm(total=int(total), desc=desc, unit="frame")

    @classmethod
    def _folder_frame_iterator(
        cls,
        folder_path: str,
    ) -> tuple[Iterable[tuple[Any, str, int]], dict[str, Any], int]:
        entries = sorted(os.listdir(folder_path))
        ext_set = cls._supported_image_extensions()
        jobs: list[tuple[str, str, int]] = []
        total = 0
        first_valid_path: str | None = None

        for name in entries:
            path = os.path.join(folder_path, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in ext_set:
                continue
            try:
                probe = iio.imread(path)
            except Exception:
                continue
            if first_valid_path is None:
                first_valid_path = path
            frames_count = len(ImageBatch._normalize_stack(probe))
            del probe
            if frames_count <= 0:
                continue
            jobs.append((path, name, frames_count))
            total += int(frames_count)

        if first_valid_path is None or total == 0:
            raise ValueError("Input folder contains no readable image frames.")

        metadata: dict[str, Any] = {
            "source_type": "Folder",
            "source_path": folder_path,
            "z_indices": list(range(total)),
        }
        metadata.update(ImageBatch._extract_voxel_sizes(first_valid_path))

        def _iter() -> Iterable[tuple[Any, str, int]]:
            z_counter = 0
            for path, name, _frames_count in jobs:
                data = iio.imread(path)
                frames = ImageBatch._normalize_stack(data)
                for frame in frames:
                    frame_id = name
                    yield frame, frame_id, z_counter
                    z_counter += 1

        return _iter(), metadata, total

    def _process_one_frame(
        self,
        *,
        frame_img: Any,
        frame_id: str,
        z_index: int,
        config: AppConfig,
        sample_props: SampleProperties,
        controller: Any,
        invert_input: bool,
    ) -> Frame:
        frame = Frame.empty(frame_id=frame_id, z_index=z_index, orig_img=frame_img)
        if self.compute_measurements:
            outcome = controller.ensure_measurements(
                frame,
                config,
                sample_props,
                invert_input=invert_input,
            )
        else:
            outcome = controller.ensure_segmentation(frame, config, invert_input=invert_input)

        if not outcome.ok and not frame.error_message:
            frame.fail(self._outcome_message(outcome))
        return frame

    def _process_batch(
        self,
        *,
        image_batch: ImageBatch,
        config: AppConfig,
        controller: Any,
        invert_input: bool,
    ) -> BatchResult:
        frame_results: list[Frame] = []

        progress = self._progress(len(image_batch.frames), "Processing")
        for idx, image in enumerate(image_batch.frames):
            frame_id, z_index = self._frame_identity(image_batch, idx)
            frame = Frame.empty(frame_id=frame_id, z_index=z_index, orig_img=image)

            if self.compute_measurements:
                outcome = controller.ensure_measurements(
                    frame,
                    config,
                    SampleProperties.from_image_metadata(
                        image_batch.metadata,
                        source_path=str((image_batch.metadata or {}).get("source_path", "")),
                    ),
                    invert_input=invert_input,
                )
            else:
                outcome = controller.ensure_segmentation(frame, config, invert_input=invert_input)

            if not outcome.ok and not frame.error_message:
                frame.fail(self._outcome_message(outcome))

            frame_results.append(
                frame.snapshot(
                    frame_id=frame_id,
                    z_index=z_index,
                )
            )
            progress.update(1)
        progress.close()
        return BatchResult(frame_results=frame_results)

    def _run_streaming_folder(
        self,
        *,
        folder_path: str,
        config: AppConfig,
        controller: Any,
        invert_input: bool,
        output_dir: str,
    ) -> int:
        from .io_export import BatchResultExporter, ExportService

        frame_iter, metadata, total_frames = self._folder_frame_iterator(folder_path)
        sample_props = SampleProperties.from_image_metadata(metadata, source_path=folder_path)
        image_batch_stub = ImageBatch(frames=[], metadata=metadata)

        save_options = ExportService.normalize_save_options(config.save_options)
        export_config = ExportService.build_export_config(
            kind=self.save_kind,
            layout="frames",
            save_options=save_options,
            source_base_name=ExportService.current_source_base_name(image_batch_stub, current_image_name=None),
            base_name=None,
            image_format_override=None,
        )
        export_config["write_manifest"] = False

        exporter = BatchResultExporter(export_config=export_config)
        failures: list[tuple[str, str]] = []
        processed = 0
        written_files: list[str] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        progress = self._progress(total_frames, "Processing")
        for frame_img, frame_id, z_index in frame_iter:
            frame = self._process_one_frame(
                frame_img=frame_img,
                frame_id=frame_id,
                z_index=z_index,
                config=config,
                sample_props=sample_props,
                controller=controller,
                invert_input=invert_input,
            )
            if self._is_frame_success(frame, self.compute_measurements):
                processed += 1
            else:
                failures.append((str(frame.frame_id), str(frame.error_message or "Processing failed")))

            batch = BatchResult(frame_results=[frame])
            export_result = exporter.save_all(batch, output_dir, export_config=export_config)
            written_files.extend(export_result.get("written_files", []))
            skipped.extend(export_result.get("skipped", []))
            errors.extend(export_result.get("errors", []))
            for err in export_result.get("errors", []):
                failures.append(
                    (
                        str(err.get("frame_id", f"frame_{err.get('frame_index', '?')}")),
                        f"Export {err.get('artifact', 'artifact')} failed: {err.get('error', 'unknown error')}",
                    )
                )
            del batch
            del frame
            del frame_img
            progress.update(1)
        progress.close()

        manifest = {
            "export_config": export_config,
            "written_files": [os.path.relpath(p, output_dir) for p in written_files],
            "written_count": len(written_files),
            "skipped": skipped,
            "errors": errors,
        }
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        print(f"Processed {processed} of {total_frames} frames.")
        print(f"Saved {len(written_files)} file(s) in: {output_dir}")
        if failures:
            print("Failures:")
            for frame_name, reason in failures[:10]:
                print(f"  - {os.path.basename(frame_name)}: {reason}")
            if len(failures) > 10:
                print(f"  ... ({len(failures) - 10} more)")
        return 0 if not failures else 1

    def _print_summary(self, *, batch_result: BatchResult, export_result: dict[str, Any]) -> list[tuple[str, str]]:
        failures: list[tuple[str, str]] = []
        processed = 0
        for fr in batch_result.frame_results:
            if self.compute_measurements:
                ok = bool(fr.has_measurements)
            else:
                ok = bool(fr.has_segmentation)
            if ok:
                processed += 1
            else:
                failures.append((str(fr.frame_id), str(fr.error_message or "Processing failed")))

        export_errors = export_result.get("errors", [])
        for err in export_errors:
            failures.append(
                (
                    str(err.get("frame_id", f"frame_{err.get('frame_index', '?')}")),
                    f"Export {err.get('artifact', 'artifact')} failed: {err.get('error', 'unknown error')}",
                )
            )

        print(f"Processed {processed} of {len(batch_result.frame_results)} frames.")
        print(f"Saved {int(export_result.get('written_count', 0))} file(s) in: {export_result.get('output_dir')}")
        if failures:
            print("Failures:")
            for frame_id, reason in failures[:10]:
                print(f"  - {os.path.basename(frame_id)}: {reason}")
            if len(failures) > 10:
                print(f"  ... ({len(failures) - 10} more)")
        return failures

    def run(self) -> int:
        if not self.input_path:
            print("Error: --input is required in CLI mode.")
            return 2
        if self.save_kind not in {"results", "csv"}:
            print("Error: --save-kind must be one of: results, csv.")
            return 2

        source_path = os.path.abspath(self.input_path)
        if not os.path.exists(source_path):
            print(f"Error: input path does not exist: {source_path}")
            return 2

        try:
            config = AppConfig.load_from_json(self.config_path)
        except Exception as exc:
            print(f"Error: failed to load config '{self.config_path}': {exc}")
            return 2

        invert_input = bool(self.invert_input) if self.invert_input is not None else False
        try:
            model_path = self._resolve_model_path(config)
        except Exception as exc:
            print(f"Error: failed to locate or download the model: {exc}")
            return 2
        try:
            device = self._resolve_device(config)
        except Exception as exc:
            print(f"Error: failed to detect runtime providers: {exc}")
            return 2

        try:
            from .fiber_analyzer_2D import FiberAnalyzer_2D
            from .pipeline import PipelineController

            analyzer = FiberAnalyzer_2D(
                model_path=model_path,
                device=device,
                strict=config.strict_centerlines,
                keep_segments_without_measurement_centerlines=bool(
                    config.keep_segments_without_measurement_centerlines
                ),
            )
            analyzer.set_strict_centerline_filter(bool(config.strict_centerlines))
            analyzer.set_keep_segments_without_measurement_centerlines(
                bool(config.keep_segments_without_measurement_centerlines)
            )
            controller = PipelineController(analyzer)
        except Exception as exc:
            print(f"Error: failed to initialize analyzer: {exc}")
            return 2

        output_dir = self._effective_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        if os.path.isdir(source_path):
            return self._run_streaming_folder(
                folder_path=source_path,
                config=config,
                controller=controller,
                invert_input=invert_input,
                output_dir=output_dir,
            )

        try:
            image_batch = ImageBatch.from_source(source_path)
        except Exception as exc:
            print(f"Error: failed to load input '{source_path}': {exc}")
            return 2

        if len(image_batch) == 0:
            print("Error: input source has no readable frames.")
            return 2

        batch_result = self._process_batch(
            image_batch=image_batch,
            config=config,
            controller=controller,
            invert_input=invert_input,
        )

        from .io_export import ExportService

        save_options = ExportService.normalize_save_options(config.save_options)
        layout = ExportService.current_save_layout(image_batch)
        source_base_name = ExportService.current_source_base_name(image_batch, current_image_name=None)
        export_config = ExportService.build_export_config(
            kind=self.save_kind,
            layout=layout,
            save_options=save_options,
            source_base_name=source_base_name,
            base_name=None,
            image_format_override=None,
        )

        export_result = batch_result.save_all(output_dir, export_config)
        failures = self._print_summary(batch_result=batch_result, export_result=export_result)
        return 0 if not failures else 1
