from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np

from .datastructures import BatchResult, Frame, ImageBatch
from .fiber_analyzer import FiberAnalyzer


class BatchAnalyzer:
    """Dimension-agnostic orchestrator that executes per-frame analysis."""

    def __init__(self, analyzer: FiberAnalyzer):
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


class StackAnalyzer3D:
    """Post-processing label merger across adjacent z slices."""

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

    def __init__(self, iou_threshold: float = 0.1, min_overlap: int = 1):
        self.iou_threshold = float(iou_threshold)
        self.min_overlap = int(min_overlap)

    def merge_labels(
        self, labels_slices: Iterable[np.ndarray]
    ) -> tuple[List[np.ndarray], Dict[tuple[int, int], int]]:
        labels_list = [np.asarray(lbl) for lbl in labels_slices]
        if not labels_list:
            return [], {}

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
