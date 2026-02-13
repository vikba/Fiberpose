from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    from numba import njit
    from numba.typed import List as NumbaList

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator

    NumbaList = None  # type: ignore

from .datastructures import BatchResult, Frame, ImageBatch
from .fiber_analyzer_2D import FiberAnalyzer_2D

_NUMBA_PAIR_CODE_LIMIT = 8_000_000


@njit(cache=True)
def _uf_find_numba(parent: np.ndarray, item: int) -> int:
    root = int(item)
    while parent[root] != root:
        root = int(parent[root])

    while parent[item] != item:
        nxt = int(parent[item])
        parent[item] = root
        item = nxt
    return root


@njit(cache=True)
def _uf_union_numba(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra = _uf_find_numba(parent, int(a))
    rb = _uf_find_numba(parent, int(b))
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


@njit(cache=True)
def _link_adjacent_labels_numba(
    labels_list,
    max_labels: np.ndarray,
    offsets: np.ndarray,
    min_overlap: int,
    iou_threshold: float,
    parent: np.ndarray,
    rank: np.ndarray,
) -> None:
    n_slices = len(labels_list)
    for z in range(n_slices - 1):
        a = labels_list[z]
        b = labels_list[z + 1]
        if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
            continue

        max_a = int(max_labels[z]) + 1
        max_b = int(max_labels[z + 1]) + 1
        area_a = np.zeros(max_a, dtype=np.int64)
        area_b = np.zeros(max_b, dtype=np.int64)
        overlap = np.zeros(max_a * max_b, dtype=np.int32)
        touched = np.empty(a.shape[0] * a.shape[1], dtype=np.int64)
        touched_n = 0

        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                la = int(a[y, x])
                lb = int(b[y, x])

                if 0 <= la < max_a:
                    area_a[la] += 1
                else:
                    la = 0

                if 0 <= lb < max_b:
                    area_b[lb] += 1
                else:
                    lb = 0

                if la > 0 and lb > 0:
                    code = la * max_b + lb
                    if overlap[code] == 0:
                        touched[touched_n] = code
                        touched_n += 1
                    overlap[code] += 1

        for idx in range(touched_n):
            code = int(touched[idx])
            ov = int(overlap[code])
            if ov < min_overlap:
                continue
            la = int(code // max_b)
            lb = int(code - (la * max_b))
            denom = int(area_a[la] + area_b[lb] - ov)
            if denom <= 0:
                continue
            if (ov / float(denom)) < iou_threshold:
                continue

            ua = int(offsets[z]) + la
            ub = int(offsets[z + 1]) + lb
            _uf_union_numba(parent, rank, ua, ub)


@njit(cache=True)
def _assign_object_ids_numba(
    parent: np.ndarray,
    max_labels: np.ndarray,
    offsets: np.ndarray,
    present_labels_list,
) -> np.ndarray:
    root_to_object = np.zeros(parent.shape[0], dtype=np.int32)
    node_to_object = np.zeros(parent.shape[0], dtype=np.int32)
    next_id = 1

    for z in range(max_labels.shape[0]):
        max_label = int(max_labels[z])
        if max_label <= 0:
            continue

        present = present_labels_list[z]
        base = int(offsets[z])
        for lbl in range(1, max_label + 1):
            if lbl >= present.shape[0] or present[lbl] == 0:
                continue
            gid = base + lbl
            root = _uf_find_numba(parent, gid)
            oid = int(root_to_object[root])
            if oid == 0:
                oid = next_id
                root_to_object[root] = oid
                next_id += 1
            node_to_object[gid] = oid
    return node_to_object


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
    """3D stack analyzer that builds volumes, merges labels, and skeletonizes objects."""

    class _UnionFind:
        def __init__(self):
            self.parent: Dict[int, int] = {}
            self.rank: Dict[int, int] = {}

        def add(self, item: int) -> None:
            if item in self.parent:
                return
            self.parent[item] = item
            self.rank[item] = 0

        def find(self, item: int) -> int:
            parent = self.parent.get(item, item)
            if parent != item:
                parent = self.find(parent)
                self.parent[item] = parent
            return parent

        def union(self, a: int, b: int) -> None:
            ra = self.find(a)
            rb = self.find(b)
            if ra == rb:
                return
            if self.rank[ra] < self.rank[rb]:
                self.parent[ra] = rb
            elif self.rank[ra] > self.rank[rb]:
                self.parent[rb] = ra
            else:
                self.parent[rb] = ra
                self.rank[ra] += 1

    def __init__(
        self,
        iou_threshold: float = 0.1,
        min_overlap: int = 1,
        voxel_spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
        histogram_bins: int = 40,
    ):
        self.iou_threshold = float(iou_threshold)
        self.min_overlap = int(min_overlap)
        self.voxel_spacing_zyx = self._parse_voxel_spacing(voxel_spacing_zyx)
        self.histogram_bins = max(int(histogram_bins), 1)

    def process(
        self,
        batch_result: BatchResult,
        *,
        do_merge: bool = True,
        do_skeleton: bool = True,
        force: bool = False,
    ) -> BatchResult:
        if batch_result.has_3d_outputs and not force:
            return batch_result

        t0 = perf_counter()
        meta: Dict[str, Any] = {
            "iou_threshold": float(self.iou_threshold),
            "min_overlap": int(self.min_overlap),
            "do_merge": bool(do_merge),
            "do_skeleton": bool(do_skeleton),
            "voxel_spacing_zyx": [float(v) for v in self.voxel_spacing_zyx],
            "histogram_bins": int(self.histogram_bins),
        }

        try:
            labels_volume = batch_result.labels_volume
            if labels_volume is None or force:
                labels_volume = self._build_labels_volume(batch_result.frame_results)
            batch_result.labels_volume = labels_volume

            if labels_volume is None:
                batch_result.merged_labels_volume = None
                batch_result.merge_object_id_map = None
                batch_result.merged_skeleton_volume = None
                batch_result.skeleton_radius_samples = None
                meta["error"] = "No per-slice labels available to build a 3D labels volume."
                batch_result.merge_meta = meta
                return batch_result

            labels_volume = np.asarray(labels_volume, dtype=np.int32)
            batch_result.labels_volume = labels_volume
            meta["volume_shape"] = [int(v) for v in labels_volume.shape]
            meta["num_labels_raw"] = int(np.max(labels_volume)) if labels_volume.size else 0

            merged_volume: Optional[np.ndarray] = None
            object_id_map: Optional[Dict[tuple[int, int], int]] = None
            if do_merge:
                merged_slices, object_id_map = self.merge_labels(
                    labels_volume[z] for z in range(labels_volume.shape[0])
                )
                if merged_slices:
                    merged_volume = np.stack(merged_slices, axis=0).astype(np.int32, copy=False)
                else:
                    merged_volume = np.zeros_like(labels_volume, dtype=np.int32)
                batch_result.merged_labels_volume = merged_volume
                batch_result.merge_object_id_map = object_id_map
            else:
                merged_volume = labels_volume.copy()
                batch_result.merged_labels_volume = merged_volume
                batch_result.merge_object_id_map = None

            meta["num_labels_merged"] = int(np.max(merged_volume)) if merged_volume.size else 0
            meta["object_id_map_size"] = len(object_id_map) if object_id_map is not None else 0

            if do_skeleton:
                skeleton_source = merged_volume if merged_volume is not None else labels_volume
                merged_skeleton, skeleton_method = self._skeletonize_per_label(skeleton_source)
                batch_result.merged_skeleton_volume = merged_skeleton
                batch_result.skeleton_radius_samples = self._recover_radius_samples(
                    skeleton_source,
                    merged_skeleton,
                    self.voxel_spacing_zyx,
                )
                meta["num_skeleton_voxels"] = int(np.count_nonzero(merged_skeleton))
                meta["skeleton_method"] = skeleton_method
                samples = batch_result.skeleton_radius_samples
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
            else:
                batch_result.merged_skeleton_volume = None
                batch_result.skeleton_radius_samples = None
                meta["num_skeleton_voxels"] = 0
                meta["num_radius_samples"] = 0

            if labels_volume.shape[0] < 2:
                meta["warning"] = "Single-slice batch: merge ran without cross-slice linking."
        except Exception as exc:
            batch_result.skeleton_radius_samples = None
            meta["error"] = str(exc)
        finally:
            meta["elapsed_ms"] = round((perf_counter() - t0) * 1000.0, 3)

        batch_result.merge_meta = meta
        return batch_result

    def merge_labels(
        self, labels_slices: Iterable[np.ndarray]
    ) -> tuple[List[np.ndarray], Dict[tuple[int, int], int]]:
        labels_list = [np.asarray(lbl, dtype=np.int32) for lbl in labels_slices]
        if not labels_list:
            return [], {}

        if not _HAVE_NUMBA:
            return self._merge_labels_python(labels_list)

        try:
            return self._merge_labels_numba(labels_list)
        except Exception:
            return self._merge_labels_python(labels_list)

    def _merge_labels_numba(
        self, labels_list: List[np.ndarray]
    ) -> tuple[List[np.ndarray], Dict[tuple[int, int], int]]:
        n_slices = len(labels_list)
        max_labels = np.zeros(n_slices, dtype=np.int32)
        offsets = np.zeros(n_slices, dtype=np.int64)

        next_offset = 0
        for z, labels in enumerate(labels_list):
            arr = np.ascontiguousarray(labels, dtype=np.int32)
            labels_list[z] = arr
            max_label = int(np.max(arr)) if arr.size else 0
            max_labels[z] = max_label
            offsets[z] = next_offset
            next_offset += max_label + 1

        for z in range(n_slices - 1):
            a = labels_list[z]
            b = labels_list[z + 1]
            if a.shape != b.shape:
                continue
            pair_codes = (int(max_labels[z]) + 1) * (int(max_labels[z + 1]) + 1)
            if pair_codes > _NUMBA_PAIR_CODE_LIMIT:
                return self._merge_labels_python(labels_list)

        labels_numba = NumbaList()  # type: ignore[operator]
        present_numba = NumbaList()  # type: ignore[operator]
        for z, labels in enumerate(labels_list):
            labels_numba.append(labels)
            max_label = int(max_labels[z])
            if max_label <= 0:
                present_numba.append(np.zeros(1, dtype=np.uint8))
                continue
            counts = np.bincount(labels.ravel().astype(np.int64), minlength=max_label + 1)
            present = (counts > 0).astype(np.uint8)
            present[0] = 0
            present_numba.append(present)

        parent = np.arange(next_offset, dtype=np.int64)
        rank = np.zeros(next_offset, dtype=np.int16)

        _link_adjacent_labels_numba(
            labels_numba,
            max_labels,
            offsets,
            int(self.min_overlap),
            float(self.iou_threshold),
            parent,
            rank,
        )
        node_to_object = _assign_object_ids_numba(parent, max_labels, offsets, present_numba)

        labels_3d: List[np.ndarray] = []
        object_id_map: Dict[tuple[int, int], int] = {}
        for z, labels in enumerate(labels_list):
            max_label = int(max_labels[z])
            if max_label <= 0:
                labels_3d.append(np.zeros_like(labels, dtype=np.int32))
                continue

            base = int(offsets[z])
            lut = np.zeros(max_label + 1, dtype=np.int32)
            lut[1:] = node_to_object[base + 1 : base + max_label + 1]
            labels_3d.append(lut[labels])

            for lbl in range(1, max_label + 1):
                oid = int(lut[lbl])
                if oid > 0:
                    object_id_map[(z, lbl)] = oid

        return labels_3d, object_id_map

    def _merge_labels_python(
        self, labels_list: List[np.ndarray]
    ) -> tuple[List[np.ndarray], Dict[tuple[int, int], int]]:
        uf = self._UnionFind()
        label_offsets: List[int] = []
        next_offset = 0
        for z, labels in enumerate(labels_list):
            max_label = int(np.max(labels)) if labels.size else 0
            label_offsets.append(next_offset)
            for lbl in range(1, max_label + 1):
                uf.add(next_offset + lbl)
            next_offset += max_label + 1

        for z in range(len(labels_list) - 1):
            a = labels_list[z]
            b = labels_list[z + 1]
            if a.shape != b.shape:
                continue
            mask = (a > 0) & (b > 0)
            if not mask.any():
                continue
            pairs = np.stack([a[mask], b[mask]], axis=1)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            area_a = np.bincount(a.ravel().astype(np.int64))
            area_b = np.bincount(b.ravel().astype(np.int64))
            for (la, lb), overlap in zip(unique_pairs, counts):
                if overlap < self.min_overlap:
                    continue
                denom = area_a[int(la)] + area_b[int(lb)] - overlap
                if denom <= 0:
                    continue
                iou = overlap / float(denom)
                if iou < self.iou_threshold:
                    continue
                ua = label_offsets[z] + int(la)
                ub = label_offsets[z + 1] + int(lb)
                uf.union(ua, ub)

        root_to_id: Dict[int, int] = {}
        object_id_map: Dict[tuple[int, int], int] = {}
        next_id = 1
        for z, labels in enumerate(labels_list):
            unique_labels = np.unique(labels)
            for lbl in unique_labels:
                if lbl == 0:
                    continue
                key = label_offsets[z] + int(lbl)
                root = uf.find(key)
                if root not in root_to_id:
                    root_to_id[root] = next_id
                    next_id += 1
                object_id_map[(z, int(lbl))] = root_to_id[root]

        labels_3d: List[np.ndarray] = []
        for z, labels in enumerate(labels_list):
            if labels.size == 0:
                labels_3d.append(labels.copy())
                continue
            mapped = np.zeros_like(labels, dtype=np.int32)
            for lbl in np.unique(labels):
                if lbl == 0:
                    continue
                mapped[labels == lbl] = object_id_map.get((z, int(lbl)), 0)
            labels_3d.append(mapped)

        return labels_3d, object_id_map

    @staticmethod
    def _parse_voxel_spacing(voxel_spacing_zyx: Sequence[float]) -> tuple[float, float, float]:
        if len(voxel_spacing_zyx) != 3:
            raise ValueError("voxel_spacing_zyx must contain exactly 3 values.")
        spacing = tuple(float(v) for v in voxel_spacing_zyx)
        if any(v <= 0 for v in spacing):
            raise ValueError("voxel_spacing_zyx values must be positive.")
        return spacing

    @staticmethod
    def _build_labels_volume(frame_results: List[Frame]) -> Optional[np.ndarray]:
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
    def _skeletonize_per_label(labels_volume: np.ndarray) -> tuple[np.ndarray, str]:
        labels_volume = np.asarray(labels_volume, dtype=np.int32)
        if labels_volume.size == 0:
            return np.zeros_like(labels_volume, dtype=np.int32), "none"

        if labels_volume.ndim != 3:
            raise ValueError(f"Expected a 3D labels volume, got shape {labels_volume.shape}.")

        skeletonize_3d, method = StackAnalyzer_3D._resolve_3d_skeletonizer()

        out = np.zeros_like(labels_volume, dtype=np.int32)
        for object_id in np.unique(labels_volume):
            if object_id == 0:
                continue
            mask = labels_volume == int(object_id)
            skel = np.asarray(skeletonize_3d(mask), dtype=bool)
            out[skel] = int(object_id)
        return out, method

    @staticmethod
    def _recover_radius_samples(
        labels_volume: np.ndarray,
        skeleton_volume: np.ndarray,
        voxel_spacing_zyx: Sequence[float],
    ) -> np.ndarray:
        labels_volume = np.asarray(labels_volume, dtype=np.int32)
        skeleton_volume = np.asarray(skeleton_volume, dtype=np.int32)
        if labels_volume.shape != skeleton_volume.shape:
            raise ValueError("labels_volume and skeleton_volume must share the same shape.")
        if labels_volume.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {labels_volume.shape}.")

        spacing = StackAnalyzer_3D._parse_voxel_spacing(voxel_spacing_zyx)
        rows: List[np.ndarray] = []

        object_ids = np.unique(skeleton_volume)
        for object_id in object_ids:
            if object_id == 0:
                continue

            object_id_int = int(object_id)
            mask = labels_volume == object_id_int
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
