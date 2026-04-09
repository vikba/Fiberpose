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
        *,
        preprocess_binary_mask: bool = True,
        opening_iters: int = 0,
        closing_iters: int = 0,
        min_component_voxels: int = 8,
        component_connectivity: int = 1,
    ):
        self.voxel_spacing_zyx = self._parse_voxel_spacing(voxel_spacing_zyx)
        self.preprocess_binary_mask = bool(preprocess_binary_mask)
        self.opening_iters = self._parse_nonnegative_int(opening_iters, "opening_iters")
        self.closing_iters = self._parse_nonnegative_int(closing_iters, "closing_iters")
        self.min_component_voxels = self._parse_nonnegative_int(
            min_component_voxels,
            "min_component_voxels",
        )
        self.component_connectivity = self._parse_connectivity(component_connectivity)

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
            binary_mask_raw = semantic_volume > 0
            raw_voxels = int(np.count_nonzero(binary_mask_raw))
            meta["num_foreground_voxels_raw"] = raw_voxels

            if self.preprocess_binary_mask:
                binary_mask, preprocess_meta = self._preprocess_binary_mask(binary_mask_raw)
                meta["mask_preprocess"] = preprocess_meta
            else:
                binary_mask = np.asarray(binary_mask_raw, dtype=bool)
                meta["mask_preprocess"] = {"enabled": False}

            cleaned_voxels = int(np.count_nonzero(binary_mask))
            meta["num_foreground_voxels"] = cleaned_voxels
            meta["semantic_num_voxels"] = cleaned_voxels
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
    def _parse_nonnegative_int(value: int, name: str) -> int:
        try:
            out = int(value)
        except Exception as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if out < 0:
            raise ValueError(f"{name} must be >= 0.")
        return out

    @staticmethod
    def _parse_connectivity(value: int) -> int:
        out = StackAnalyzer_3D._parse_nonnegative_int(value, "component_connectivity")
        if out < 1 or out > 3:
            raise ValueError("component_connectivity must be in [1, 3] for 3D masks.")
        return out

    def _preprocess_binary_mask(self, binary_mask: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        from scipy.ndimage import (
            binary_closing,
            binary_opening,
            generate_binary_structure,
            label,
        )

        mask = np.asarray(binary_mask, dtype=bool)
        structure = generate_binary_structure(3, int(self.component_connectivity))
        before_voxels = int(np.count_nonzero(mask))

        if self.opening_iters > 0:
            for _ in range(self.opening_iters):
                mask = np.asarray(binary_opening(mask, structure=structure), dtype=bool)

        if self.closing_iters > 0:
            for _ in range(self.closing_iters):
                mask = np.asarray(binary_closing(mask, structure=structure), dtype=bool)

        removed_components = 0
        removed_voxels = 0
        num_components_before = 0
        if self.min_component_voxels > 1 and np.any(mask):
            labeled, num_components_before = label(mask, structure=structure)
            component_sizes = np.bincount(labeled.ravel())
            keep = component_sizes >= int(self.min_component_voxels)
            keep[0] = False
            removed_components = int(np.count_nonzero((~keep[1:]) if keep.size > 1 else np.array([], dtype=bool)))
            if removed_components > 0:
                keep_mask = keep[labeled]
                removed_voxels = int(np.count_nonzero(mask & (~keep_mask)))
                mask = np.asarray(keep_mask, dtype=bool)

        after_voxels = int(np.count_nonzero(mask))
        meta: Dict[str, Any] = {
            "enabled": True,
            "opening_iters": int(self.opening_iters),
            "closing_iters": int(self.closing_iters),
            "min_component_voxels": int(self.min_component_voxels),
            "connectivity": int(self.component_connectivity),
            "voxels_before": before_voxels,
            "voxels_after": after_voxels,
            "components_before_filter": int(num_components_before),
            "components_removed": int(removed_components),
            "voxels_removed_by_component_filter": int(removed_voxels),
        }
        return mask, meta

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
