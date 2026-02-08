from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional
import os

import imageio.v3 as iio
import numpy as np

from .fiber_analyzer import FiberAnalyzer


@dataclass
class ImageBatch:
    """Container for an ordered batch of frames plus lightweight metadata."""
    frames: List[np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @staticmethod
    def _normalize_stack(data: np.ndarray) -> List[np.ndarray]:
        if data.ndim == 2:
            return [data]
        if data.ndim == 3:
            # Treat HxWxC as a single frame when C looks like color channels.
            if data.shape[2] in (3, 4):
                return [data]
            # Otherwise assume ZxHxW grayscale stack.
            return [data[z] for z in range(data.shape[0])]
        if data.ndim == 4:
            # ZxHxWxC color stack.
            return [data[z] for z in range(data.shape[0])]
        raise ValueError("Unsupported image dimensions for batch loading.")

    @classmethod
    def from_path(cls, path: str) -> ImageBatch:
        data = iio.imread(path)
        frames = cls._normalize_stack(data)
        source_type = "single" if len(frames) == 1 else "stack"
        filenames = [os.path.basename(path)] * len(frames)
        metadata = {
            "source_type": source_type,
            "source_path": path,
            "filenames": filenames,
            "z_indices": list(range(len(frames))),
        }
        return cls(frames=frames, metadata=metadata)

    @classmethod
    def from_folder(
        cls, folder: str, extensions: Optional[Iterable[str]] = None
    ) -> ImageBatch:
        entries = sorted(os.listdir(folder))
        ext_set = {ext.lower() for ext in extensions} if extensions else None
        frames: List[np.ndarray] = []
        filenames: List[str] = []
        for name in entries:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if ext_set is not None:
                if os.path.splitext(name)[1].lower() not in ext_set:
                    continue
            try:
                data = iio.imread(path)
            except Exception:
                continue
            image_frames = cls._normalize_stack(data)
            for idx, frame in enumerate(image_frames):
                frames.append(frame)
                if len(image_frames) == 1:
                    filenames.append(name)
                else:
                    filenames.append(f"{os.path.splitext(name)[0]}_z{idx:03d}")
        metadata = {
            "source_type": "folder",
            "source_path": folder,
            "filenames": filenames,
            "z_indices": list(range(len(frames))),
        }
        return cls(frames=frames, metadata=metadata)

    @classmethod
    def from_source(
        cls, path: str, extensions: Optional[Iterable[str]] = None
    ) -> ImageBatch:
        if os.path.isdir(path):
            return cls.from_folder(path, extensions=extensions)
        if os.path.isfile(path):
            return cls.from_path(path)
        raise FileNotFoundError(f"Source path does not exist: {path}")


@dataclass
class FrameResult:
    labels_2d: Optional[np.ndarray]
    centerlines: Optional[np.ndarray]
    measurements: Optional[Dict[str, Any]]
    z_index: int
    frame_id: str
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    frame_results: List[FrameResult]
    labels_3d: Optional[List[np.ndarray]] = None
    object_id_map: Optional[Dict[tuple[int, int], int]] = None
    adjacency_graph: Optional[Any] = None

    def frame_at(self, index: int) -> FrameResult:
        return self.frame_results[index]


class BatchAnalyzer:
    """Dimension-agnostic orchestrator that executes per-frame analysis."""

    def __init__(self, analyzer: FiberAnalyzer):
        self._analyzer = analyzer

    def analyze(
        self,
        batch: ImageBatch,
        frame_processor: Callable[[np.ndarray, str, int], FrameResult],
    ) -> BatchResult:
        results: List[FrameResult] = []
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
