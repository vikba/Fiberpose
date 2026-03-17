from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_opening,
    convolve,
    distance_transform_edt,
    label as ndi_label,
    zoom,
)

from .datastructures import BatchResult, Frame, ImageBatch
from .fiber_analyzer_2D import FiberAnalyzer_2D

class BatchAnalyzer:
    """Dimension-agnostic orchestrator that executes per-frame analysis."""

    def __init__(self, analyzer: FiberAnalyzer_2D):
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
    """3D stack analyzer for semantic mask skeletonization and fragment splitting."""

    def __init__(
        self,
        voxel_spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
        histogram_bins: int = 40,
        junction_cut_scale: float = 1.6,
        junction_cut_min_radius: float = 1.25,
        junction_cut_radius_bias: float = 0.5,
        junction_dilation_iters: int = 1,
        opening_kernel_zyx: Sequence[int] = (1, 3, 3),
        junction_cut_max_radius: Optional[float] = None,
        skeleton_anisotropy_threshold: float = 1.3,
    ):
        self.voxel_spacing_zyx = self._parse_voxel_spacing(voxel_spacing_zyx)
        self.histogram_bins = max(int(histogram_bins), 1)
        self.junction_cut_scale = float(junction_cut_scale)
        self.junction_cut_min_radius = max(0.0, float(junction_cut_min_radius))
        self.junction_cut_radius_bias = max(0.0, float(junction_cut_radius_bias))
        self.junction_dilation_iters = max(0, int(junction_dilation_iters))
        if len(opening_kernel_zyx) != 3:
            raise ValueError("opening_kernel_zyx must contain exactly 3 values.")
        kz, ky, kx = (int(v) for v in opening_kernel_zyx)
        if kz < 1 or ky < 1 or kx < 1:
            raise ValueError("opening_kernel_zyx values must be >= 1.")
        self.opening_kernel_zyx = (kz, ky, kx)
        self.opening_kernel = np.ones(self.opening_kernel_zyx, dtype=bool)
        self.junction_cut_max_radius = (
            float(junction_cut_max_radius) if junction_cut_max_radius is not None else None
        )
        if self.junction_cut_max_radius is not None and self.junction_cut_max_radius <= 0.0:
            raise ValueError("junction_cut_max_radius must be positive when provided.")
        self.skeleton_anisotropy_threshold = max(1.0, float(skeleton_anisotropy_threshold))

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
            "pipeline": "semantic_skeleton",
            "voxel_spacing_zyx": [float(v) for v in self.voxel_spacing_zyx],
            "histogram_bins": int(self.histogram_bins),
            "junction_cut_scale": float(self.junction_cut_scale),
            "junction_cut_min_radius": float(self.junction_cut_min_radius),
            "junction_cut_radius_bias": float(self.junction_cut_radius_bias),
            "junction_dilation_iters": int(self.junction_dilation_iters),
            "opening_kernel_zyx": [int(v) for v in self.opening_kernel_zyx],
            "junction_cut_max_radius": (
                float(self.junction_cut_max_radius)
                if self.junction_cut_max_radius is not None
                else None
            ),
            "skeleton_anisotropy_threshold": float(self.skeleton_anisotropy_threshold),
        }

        try:
            semantic_volume = batch_result.semantic_volume
            if semantic_volume is None or force:
                semantic_volume = self._build_semantic_volume(batch_result.frame_results)
            batch_result.semantic_volume = semantic_volume

            if semantic_volume is None:
                batch_result.semantic_skeleton_volume = None
                batch_result.instance_volume = None
                batch_result.instance_skeleton_volume = None
                batch_result.instance_radius_samples = None
                meta["error"] = "No per-slice labels available to build a 3D semantic volume."
                batch_result.merge_meta = meta
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
                zero_labels = np.zeros_like(semantic_volume, dtype=np.int32)
                batch_result.semantic_skeleton_volume = zero_labels
                batch_result.instance_volume = zero_labels
                batch_result.instance_skeleton_volume = zero_labels
                batch_result.instance_radius_samples = np.zeros((0, 5), dtype=np.float32)
                meta["num_labels_merged"] = 0
                meta["instance_num_labels"] = 0
                meta["num_skeleton_voxels"] = 0
                meta["num_junction_voxels"] = 0
                meta["instance_num_voxels"] = 0
                meta["num_radius_samples"] = 0
                meta["radius_hist_counts"] = []
                meta["radius_hist_edges"] = []
                meta["radius_hist_kind"] = "radius"
                batch_result.merge_meta = meta
                return batch_result

            skeletonize_3d, skeleton_method = self._resolve_3d_skeletonizer()
            anisotropy_ratio = float(max(self.voxel_spacing_zyx) / min(self.voxel_spacing_zyx))
            meta["voxel_anisotropy_ratio"] = anisotropy_ratio
            dt3d = np.asarray(
                distance_transform_edt(binary_mask, sampling=self.voxel_spacing_zyx),
                dtype=np.float32,
            )
            skeleton_raw, spacing_mode, zoom_factors_zyx = self._skeletonize_with_spacing(
                binary_mask,
                skeletonize_3d,
                self.voxel_spacing_zyx,
                self.skeleton_anisotropy_threshold,
            )
            raw_skeleton = skeleton_raw.astype(np.int32, copy=False)
            batch_result.semantic_skeleton_volume = raw_skeleton
            junction_mask = self._detect_junctions(
                skeleton_raw,
                dilation_iters=self.junction_dilation_iters,
            )
            cut_mask = self._cut_at_junctions(
                binary_mask,
                dt3d,
                junction_mask,
                self.voxel_spacing_zyx,
                self.junction_cut_scale,
                self.junction_cut_min_radius,
                self.junction_cut_radius_bias,
                self.junction_cut_max_radius,
            )
            if np.any(cut_mask):
                cut_mask = np.asarray(
                    binary_opening(cut_mask, structure=self.opening_kernel),
                    dtype=bool,
                )
            fragment_labels, n_fragments = ndi_label(
                cut_mask.astype(np.uint8),
                structure=np.ones((3, 3, 3), dtype=np.uint8),
            )
            fragment_labels = np.asarray(fragment_labels, dtype=np.int32)
            skeleton_frag, _, _ = self._skeletonize_with_spacing(
                cut_mask,
                skeletonize_3d,
                self.voxel_spacing_zyx,
                self.skeleton_anisotropy_threshold,
            )
            fragment_skeleton = np.zeros_like(fragment_labels, dtype=np.int32)
            fragment_skeleton[skeleton_frag] = fragment_labels[skeleton_frag]

            batch_result.instance_volume = fragment_labels
            batch_result.instance_skeleton_volume = fragment_skeleton
            batch_result.instance_radius_samples = self._recover_radius_samples(
                fragment_labels,
                fragment_skeleton,
                self.voxel_spacing_zyx,
            )

            meta["skeleton_method"] = skeleton_method
            meta["skeleton_spacing_mode"] = spacing_mode
            meta["skeleton_zoom_factors_zyx"] = [float(v) for v in zoom_factors_zyx]
            meta["num_skeleton_voxels_raw"] = int(np.count_nonzero(skeleton_raw))
            meta["num_junction_voxels"] = int(np.count_nonzero(junction_mask))
            meta["num_foreground_voxels_cut"] = int(np.count_nonzero(cut_mask))
            meta["num_labels_merged"] = int(n_fragments)
            meta["instance_num_labels"] = int(n_fragments)
            meta["instance_num_voxels"] = int(np.count_nonzero(cut_mask))
            meta["num_skeleton_voxels"] = int(np.count_nonzero(fragment_skeleton))
            samples = batch_result.instance_radius_samples
            num_samples = int(samples.shape[0]) if samples is not None else 0
            meta["num_radius_samples"] = num_samples
            if num_samples > 0:
                radii = np.asarray(samples[:, 4], dtype=np.float32)
                meta["radius_min"] = float(np.min(radii))
                meta["radius_max"] = float(np.max(radii))
                meta["radius_mean"] = float(np.mean(radii))
                counts, edges = np.histogram(radii, bins=self.histogram_bins)
                meta["radius_hist_counts"] = counts.astype(int).tolist()
                meta["radius_hist_edges"] = edges.astype(float).tolist()
                meta["radius_hist_kind"] = "radius"
            else:
                meta["radius_hist_counts"] = []
                meta["radius_hist_edges"] = []
                meta["radius_hist_kind"] = "radius"

            if semantic_volume.shape[0] < 2:
                meta["warning"] = "Single-slice batch: semantic 3D flow ran on one slice."
        except Exception as exc:
            batch_result.semantic_skeleton_volume = None
            batch_result.instance_skeleton_volume = None
            batch_result.instance_radius_samples = None
            meta["error"] = str(exc)
        finally:
            meta["elapsed_ms"] = round((perf_counter() - t0) * 1000.0, 3)

        batch_result.merge_meta = meta
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

    @staticmethod
    def _skeletonize_with_spacing(
        mask: np.ndarray,
        skeletonize_3d: Callable[[np.ndarray], np.ndarray],
        voxel_spacing_zyx: Sequence[float],
        anisotropy_threshold: float,
    ) -> tuple[np.ndarray, str, tuple[float, float, float]]:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim != 3:
            raise ValueError(f"Expected a 3D mask, got shape {mask_bool.shape}.")
        if mask_bool.size == 0 or not np.any(mask_bool):
            return (
                np.zeros_like(mask_bool, dtype=bool),
                "native",
                (1.0, 1.0, 1.0),
            )

        spacing = StackAnalyzer_3D._parse_voxel_spacing(voxel_spacing_zyx)
        ratio = float(max(spacing) / min(spacing))
        if ratio <= float(anisotropy_threshold) or min(mask_bool.shape) < 2:
            return (
                np.asarray(skeletonize_3d(mask_bool), dtype=bool),
                "native",
                (1.0, 1.0, 1.0),
            )

        target_spacing = float(min(spacing))
        zoom_factors = (
            float(spacing[0] / target_spacing),
            float(spacing[1] / target_spacing),
            float(spacing[2] / target_spacing),
        )
        mask_iso = np.asarray(
            zoom(
                mask_bool.astype(np.uint8, copy=False),
                zoom_factors,
                order=0,
                mode="nearest",
                prefilter=False,
            ),
            dtype=bool,
        )
        if mask_iso.size == 0 or not np.any(mask_iso):
            return (
                np.zeros_like(mask_bool, dtype=bool),
                "isotropic_resample",
                zoom_factors,
            )

        skeleton_iso = np.asarray(skeletonize_3d(mask_iso), dtype=bool)
        if not np.any(skeleton_iso):
            return (
                np.zeros_like(mask_bool, dtype=bool),
                "isotropic_resample",
                zoom_factors,
            )

        coords_iso = np.argwhere(skeleton_iso).astype(np.float64, copy=False)
        inv_factors = np.asarray([1.0 / zf for zf in zoom_factors], dtype=np.float64)
        coords_orig = np.rint(coords_iso * inv_factors).astype(np.int64, copy=False)
        coords_orig[:, 0] = np.clip(coords_orig[:, 0], 0, mask_bool.shape[0] - 1)
        coords_orig[:, 1] = np.clip(coords_orig[:, 1], 0, mask_bool.shape[1] - 1)
        coords_orig[:, 2] = np.clip(coords_orig[:, 2], 0, mask_bool.shape[2] - 1)

        skeleton_native = np.zeros_like(mask_bool, dtype=bool)
        skeleton_native[coords_orig[:, 0], coords_orig[:, 1], coords_orig[:, 2]] = True
        skeleton_native &= mask_bool
        return skeleton_native, "isotropic_resample", zoom_factors

    @staticmethod
    def _detect_junctions(skeleton_mask: np.ndarray, dilation_iters: int = 0) -> np.ndarray:
        skeleton = np.asarray(skeleton_mask, dtype=bool)
        if skeleton.ndim != 3:
            raise ValueError(f"Expected a 3D skeleton mask, got shape {skeleton.shape}.")
        if not np.any(skeleton):
            return np.zeros_like(skeleton, dtype=bool)

        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        neighbor_count = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
        neighbor_count = neighbor_count - skeleton.astype(np.uint8)
        junctions = skeleton & (neighbor_count >= 3)
        if dilation_iters > 0 and np.any(junctions):
            junctions = binary_dilation(
                junctions,
                structure=np.ones((3, 3, 3), dtype=bool),
                iterations=int(dilation_iters),
            )
        return junctions

    @staticmethod
    def _cut_at_junctions(
        binary_mask: np.ndarray,
        dt_volume: np.ndarray,
        junction_mask: np.ndarray,
        voxel_spacing_zyx: Sequence[float],
        radius_scale: float,
        min_radius: float,
        radius_bias: float,
        max_radius: Optional[float],
    ) -> np.ndarray:
        mask = np.asarray(binary_mask, dtype=bool)
        dt = np.asarray(dt_volume, dtype=np.float32)
        junctions = np.asarray(junction_mask, dtype=bool)
        if mask.shape != dt.shape or mask.shape != junctions.shape:
            raise ValueError("binary_mask, dt_volume, and junction_mask must have the same shape.")
        if mask.ndim != 3:
            raise ValueError(f"Expected a 3D mask, got shape {mask.shape}.")

        spacing = StackAnalyzer_3D._parse_voxel_spacing(voxel_spacing_zyx)
        sz, sy, sx = spacing
        out = mask.copy()
        coords = np.argwhere(junctions)
        if coords.size == 0:
            return out

        for z, y, x in coords:
            local_r = float(dt[z, y, x]) * float(radius_scale) + float(radius_bias)
            if not np.isfinite(local_r) or local_r <= 0.0:
                continue
            radius = max(float(min_radius), local_r)
            if max_radius is not None:
                radius = min(radius, float(max_radius))
            if radius <= 0.0:
                continue

            rz = max(1, int(np.ceil(radius / sz)))
            ry = max(1, int(np.ceil(radius / sy)))
            rx = max(1, int(np.ceil(radius / sx)))

            z0 = max(0, int(z) - rz)
            y0 = max(0, int(y) - ry)
            x0 = max(0, int(x) - rx)
            z1 = min(mask.shape[0], int(z) + rz + 1)
            y1 = min(mask.shape[1], int(y) + ry + 1)
            x1 = min(mask.shape[2], int(x) + rx + 1)

            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            dist2 = (
                ((zz - int(z)) * sz) ** 2
                + ((yy - int(y)) * sy) ** 2
                + ((xx - int(x)) * sx) ** 2
            )
            sphere = dist2 <= (radius * radius)
            region = out[z0:z1, y0:y1, x0:x1]
            region[sphere] = False
            out[z0:z1, y0:y1, x0:x1] = region

        return out

    @staticmethod
    def _recover_radius_samples(
        label_volume: np.ndarray,
        skeleton_volume: np.ndarray,
        voxel_spacing_zyx: Sequence[float],
    ) -> np.ndarray:
        label_volume = np.asarray(label_volume, dtype=np.int32)
        skeleton_volume = np.asarray(skeleton_volume, dtype=np.int32)
        if label_volume.shape != skeleton_volume.shape:
            raise ValueError("label_volume and skeleton_volume must share the same shape.")
        if label_volume.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {label_volume.shape}.")

        spacing = StackAnalyzer_3D._parse_voxel_spacing(voxel_spacing_zyx)
        rows: List[np.ndarray] = []

        object_ids = np.unique(skeleton_volume)
        for object_id in object_ids:
            if object_id == 0:
                continue

            object_id_int = int(object_id)
            mask = label_volume == object_id_int
            skel = skeleton_volume == object_id_int
            if not np.any(mask) or not np.any(skel):
                continue

            coords = np.argwhere(mask)
            z0, y0, x0 = coords.min(axis=0)
            z1, y1, x1 = coords.max(axis=0) + 1
            mask_crop = mask[z0:z1, y0:y1, x0:x1]
            skel_crop = skel[z0:z1, y0:y1, x0:x1]

            dt = np.asarray(distance_transform_edt(mask_crop, sampling=spacing), dtype=np.float32)
            zz, yy, xx = np.nonzero(skel_crop)
            if zz.size == 0:
                continue

            radii = dt[zz, yy, xx]
            row = np.column_stack(
                [
                    np.full(radii.shape, float(object_id_int), dtype=np.float32),
                    (zz + z0).astype(np.float32, copy=False),
                    (yy + y0).astype(np.float32, copy=False),
                    (xx + x0).astype(np.float32, copy=False),
                    radii.astype(np.float32, copy=False),
                ]
            )
            rows.append(row.astype(np.float32, copy=False))

        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)
