from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .datastructures import BatchResult, Frame, ImageBatch

if TYPE_CHECKING:
    from .fiber_analyzer_2D import FiberAnalyzer_2D

class BatchAnalyzer:
    """Dimension-agnostic orchestrator that executes per-frame analysis."""

    def __init__(self, analyzer: "FiberAnalyzer_2D"):
        self._analyzer = analyzer

    def analyze(
        self,
        batch: ImageBatch,
        frame_processor: Callable[[np.ndarray, str, int], Frame],
    ) -> BatchResult:
        results: List[Frame] = []
        z_indices = batch.metadata.get("z_indices", list(range(len(batch))))
        filenames = batch.metadata.get("filenames", [str(i) for i in range(len(batch))])
        for idx, frame in enumerate(batch.frames):
            self._analyzer.reset_prediction_state()
            frame_id = str(filenames[idx]) if idx < len(filenames) else str(idx)
            z_index = int(z_indices[idx]) if idx < len(z_indices) else idx
            results.append(frame_processor(frame, frame_id, z_index))
        return BatchResult(frame_results=results)


class StackAnalyzer_3D:
    """3D stack analyzer for semantic volume merge and native-grid skeletonization."""

    def __init__(
        self,
        voxel_spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    ):
        self.voxel_spacing_zyx = self._parse_voxel_spacing(voxel_spacing_zyx)

    def process(
        self,
        batch_result: BatchResult,
        *,
        force: bool = False,
    ) -> BatchResult:
        if batch_result.has_3d_outputs and not force:
            return batch_result

        t0 = perf_counter()
        meta: Dict[str, Any] = {
            "pipeline": "semantic_skeleton_native",
            "voxel_spacing_zyx": [float(v) for v in self.voxel_spacing_zyx],
        }

        try:
            semantic_volume = batch_result.semantic_volume
            if semantic_volume is None or force:
                semantic_volume = self._build_semantic_volume(batch_result.frame_results)
            batch_result.semantic_volume = semantic_volume

            if semantic_volume is None:
                batch_result.semantic_skeleton_volume = None
                meta["error"] = "No per-slice labels available to build a 3D semantic volume."
                batch_result.stack3d_meta = meta
                return batch_result

            semantic_volume = np.asarray(semantic_volume, dtype=np.int32)
            batch_result.semantic_volume = semantic_volume
            meta["volume_shape"] = [int(v) for v in semantic_volume.shape]
            meta["num_labels_raw"] = int(np.max(semantic_volume)) if semantic_volume.size else 0
            meta["semantic_num_labels"] = int(np.max(semantic_volume)) if semantic_volume.size else 0
            binary_mask = semantic_volume > 0
            meta["num_foreground_voxels"] = int(np.count_nonzero(binary_mask))
            meta["semantic_num_voxels"] = int(np.count_nonzero(binary_mask))
            if not np.any(binary_mask):
                batch_result.semantic_skeleton_volume = np.zeros_like(semantic_volume, dtype=np.int32)
                meta["num_skeleton_voxels"] = 0
                batch_result.stack3d_meta = meta
                return batch_result

            skeletonize_3d, skeleton_method = self._resolve_3d_skeletonizer()
            skeleton_mask = np.asarray(skeletonize_3d(binary_mask), dtype=bool)
            semantic_skeleton = np.zeros_like(semantic_volume, dtype=np.int32)
            semantic_skeleton[skeleton_mask] = 1
            batch_result.semantic_skeleton_volume = semantic_skeleton
            meta["skeleton_method"] = skeleton_method
            meta["num_skeleton_voxels"] = int(np.count_nonzero(skeleton_mask))

            if semantic_volume.shape[0] < 2:
                meta["warning"] = "Single-slice batch: semantic 3D flow ran on one slice."
        except Exception as exc:
            batch_result.semantic_skeleton_volume = None
            meta["error"] = str(exc)
        finally:
            meta["elapsed_ms"] = round((perf_counter() - t0) * 1000.0, 3)

        batch_result.stack3d_meta = meta
        return batch_result

    @staticmethod
    def _parse_voxel_spacing(voxel_spacing_zyx: Sequence[float]) -> tuple[float, float, float]:
        if len(voxel_spacing_zyx) != 3:
            raise ValueError("voxel_spacing_zyx must contain exactly 3 values.")
        spacing = tuple(float(v) for v in voxel_spacing_zyx)
        if any(v <= 0 for v in spacing):
            raise ValueError("voxel_spacing_zyx values must be positive.")
        return spacing

    @staticmethod
    def _build_semantic_volume(frame_results: List[Frame]) -> Optional[np.ndarray]:
        if not frame_results:
            return None

        labels_slices: List[Optional[np.ndarray]] = []
        available: List[np.ndarray] = []
        for frame in frame_results:
            labels = frame.labels_2d
            if labels is None:
                labels_slices.append(None)
                continue
            arr = np.asarray(labels)
            if arr.ndim == 3:
                arr = arr[..., 0]
            if arr.ndim != 2:
                labels_slices.append(None)
                continue
            arr = arr.astype(np.int32, copy=False)
            labels_slices.append(arr)
            available.append(arr)

        if not available:
            return None

        min_h = min(a.shape[0] for a in available)
        min_w = min(a.shape[1] for a in available)
        stack: List[np.ndarray] = []
        for arr in labels_slices:
            if arr is None:
                stack.append(np.zeros((min_h, min_w), dtype=np.int32))
            else:
                stack.append(arr[:min_h, :min_w].astype(np.int32, copy=False))
        return np.stack(stack, axis=0)

    @staticmethod
    def _resolve_3d_skeletonizer() -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        try:
            from skimage.morphology import skeletonize_3d as _skeletonize_3d

            return _skeletonize_3d, "skimage.morphology.skeletonize_3d"
        except Exception:
            pass

        try:
            from skimage.morphology import skeletonize as _skeletonize
        except Exception as exc:
            raise RuntimeError(
                "True 3D skeletonization requires scikit-image with a 3D skeletonizer."
            ) from exc

        probe = np.zeros((3, 3, 3), dtype=bool)
        try:
            probe_result = np.asarray(_skeletonize(probe, method="lee"), dtype=bool)

            def _skeletonize_lee(mask: np.ndarray) -> np.ndarray:
                return np.asarray(_skeletonize(mask, method="lee"), dtype=bool)

            method = "skimage.morphology.skeletonize(method='lee')"
        except Exception:
            probe_result = np.asarray(_skeletonize(probe), dtype=bool)

            def _skeletonize_lee(mask: np.ndarray) -> np.ndarray:
                return np.asarray(_skeletonize(mask), dtype=bool)

            method = "skimage.morphology.skeletonize"

        if probe_result.ndim != 3:
            raise RuntimeError(
                "Detected a 2D-only skeletonize implementation. Install a scikit-image build with 3D skeletonization support."
            )

        return _skeletonize_lee, method
