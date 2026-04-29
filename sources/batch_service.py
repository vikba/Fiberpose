"""Batch orchestration service for frame-wise processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .datastructures import BatchResult, Frame, ImageBatch


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
        batch_result.stack3d_meta = {
            "info": "3D post-processing not run yet. Click 'Analyze 3D' to merge labels across slices."
        }
        callbacks.on_finished(batch_result)
        return batch_result
