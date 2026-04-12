"""Fastest-ascend segmentation over distance maps with optional numba acceleration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from numba import njit, prange


@dataclass(frozen=True)
class FastestAscendConfig:
    """Store configuration values for fastest ascend."""
    sigma: float = 1.0
    dist_thresh: float = 0.2
    flux_merge_thresh: float = 5.0

    def normalized(self) -> "FastestAscendConfig":
        """Orchestrate normalized within the current module workflow.
        
        Returns:
            'FastestAscendConfig': Result object produced by the normalized routine."""
        sigma = max(0.0, float(self.sigma))
        dist_thresh = float(np.clip(float(self.dist_thresh), 0.0, 1.0))
        flux_merge_thresh = max(0.0, float(self.flux_merge_thresh))
        return FastestAscendConfig(
            sigma=sigma,
            dist_thresh=dist_thresh,
            flux_merge_thresh=flux_merge_thresh,
        )


@njit(fastmath=True, parallel=True)
def _compute_steepest_neighbor_map(dist_map_flat: np.ndarray, h: int, w: int, dist_thresh: float) -> np.ndarray:
    """Return 1D flow targets: -1 for background/attractor, else downstream index."""
    n_pixels = h * w
    next_indices = np.full(n_pixels, -1, dtype=np.int32)

    for i in prange(n_pixels):
        if dist_map_flat[i] <= dist_thresh:
            continue

        r = i // w
        c = i - r * w
        best_val = dist_map_flat[i]
        best_idx = i

        for dr in (-1, 0, 1):
            nr = r + dr
            if nr < 0 or nr >= h:
                continue
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nc = c + dc
                if nc < 0 or nc >= w:
                    continue
                nbr_idx = nr * w + nc
                nbr_val = dist_map_flat[nbr_idx]
                if nbr_val > best_val:
                    best_val = nbr_val
                    best_idx = nbr_idx

        if best_idx != i:
            next_indices[i] = best_idx

    return next_indices


@njit(fastmath=True)
def _compute_flux_topology(next_indices: np.ndarray, n_pixels: int) -> np.ndarray:
    """Accumulate unit mass along the directed flow graph with Kahn traversal."""
    flux = np.zeros(n_pixels, dtype=np.float32)
    in_degree = np.zeros(n_pixels, dtype=np.int32)

    for i in range(n_pixels):
        target = next_indices[i]
        if target != -1:
            flux[i] = 1.0
            in_degree[target] += 1

    queue = np.zeros(n_pixels, dtype=np.int32)
    head = 0
    tail = 0
    for i in range(n_pixels):
        if next_indices[i] != -1 and in_degree[i] == 0:
            queue[tail] = i
            tail += 1

    while head < tail:
        curr = queue[head]
        head += 1

        downstream = next_indices[curr]
        if downstream != -1:
            flux[downstream] += flux[curr]
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                queue[tail] = downstream
                tail += 1

    return flux


@njit(fastmath=True)
def _resolve_paths_to_labels(next_indices: np.ndarray) -> np.ndarray:
    """Resolve each pixel to its final attractor index with path compression."""
    n_pixels = next_indices.shape[0]
    dest_map = np.full(n_pixels, -1, dtype=np.int32)

    for i in range(n_pixels):
        if dest_map[i] != -1:
            continue

        curr = i
        step_count = 0
        while True:
            if dest_map[curr] != -1:
                sink = dest_map[curr]
                break

            nxt = next_indices[curr]
            if nxt == -1:
                sink = curr
                break

            curr = nxt
            step_count += 1
            if step_count > n_pixels:
                sink = curr
                break

        curr = i
        step_count = 0
        while dest_map[curr] == -1:
            dest_map[curr] = sink
            nxt = next_indices[curr]
            if nxt == -1 or nxt == curr:
                break
            curr = nxt
            step_count += 1
            if step_count > n_pixels:
                break

    return dest_map


@njit(fastmath=True)
def _merge_regions_numba(
    labels_flat: np.ndarray,
    flux_flat: np.ndarray,
    h: int,
    w: int,
    flux_threshold: float,
) -> np.ndarray:
    """Merge neighboring regions if either side has flux above threshold."""
    n_pixels = h * w
    parent = np.arange(n_pixels + 1, dtype=np.int32)

    def find(i):
        """Orchestrate find within the current module workflow.
        
        Args:
            i (Any): Value for i interpreted by this routine.
        
        Returns:
            Any: Object containing the find outcome for downstream use."""
        root = i
        while root != parent[root]:
            root = parent[root]

        curr = i
        while curr != root:
            nxt = parent[curr]
            parent[curr] = root
            curr = nxt
        return root

    def union(i, j):
        """Orchestrate union within the current module workflow.
        
        Args:
            i (Any): Value for i interpreted by this routine.
            j (Any): Value for j interpreted by this routine."""
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for r in range(h):
        for c in range(w - 1):
            idx1 = r * w + c
            idx2 = idx1 + 1

            l1 = labels_flat[idx1]
            l2 = labels_flat[idx2]
            if l1 != l2 and l1 != -1 and l2 != -1:
                f1 = flux_flat[idx1]
                f2 = flux_flat[idx2]
                if (f1 if f1 > f2 else f2) > flux_threshold:
                    union(l1, l2)

    for r in range(h - 1):
        for c in range(w):
            idx1 = r * w + c
            idx2 = (r + 1) * w + c

            l1 = labels_flat[idx1]
            l2 = labels_flat[idx2]
            if l1 != l2 and l1 != -1 and l2 != -1:
                f1 = flux_flat[idx1]
                f2 = flux_flat[idx2]
                if (f1 if f1 > f2 else f2) > flux_threshold:
                    union(l1, l2)

    merged_labels = np.zeros_like(labels_flat)
    for i in range(n_pixels):
        old_label = labels_flat[i]
        if old_label != -1:
            merged_labels[i] = find(old_label)
        else:
            merged_labels[i] = -1

    return merged_labels


class FluxSegmentor:
    """Distance-transform segmentation via steepest-ascent flow and flux-based merging."""

    def __init__(self, dist_map: np.ndarray, sigma: float = 1.0):
        """Initialize flux segmentor and establish required runtime state.
        
        Args:
            dist_map (np.ndarray): Value for dist map interpreted by this routine.
            sigma (float): Numeric control parameter for sigma during computation.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        dist_arr = np.asarray(dist_map, dtype=np.float32)
        if dist_arr.ndim != 2:
            raise ValueError("dist_map must be a 2D array")

        self.h, self.w = dist_arr.shape
        self.H, self.W = self.h, self.w
        self.dist_map = self._prepare_distance_map(dist_arr, sigma=float(sigma))
        self.dist_flat = np.ascontiguousarray(self.dist_map.ravel())

        self.next_indices: np.ndarray | None = None
        self.flux_map: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.attractors: np.ndarray | None = None

    @staticmethod
    def _prepare_distance_map(dist_map: np.ndarray, sigma: float) -> np.ndarray:
        """Provide an internal helper for prepare distance map.
        
        Args:
            dist_map (np.ndarray): Value for dist map interpreted by this routine.
            sigma (float): Numeric control parameter for sigma during computation.
        
        Returns:
            np.ndarray: Array containing the prepare distance map output aligned with downstream expectations."""
        safe_map = np.nan_to_num(dist_map, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = float(np.min(safe_map))
        max_val = float(np.max(safe_map))
        if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
            normalized = np.zeros_like(safe_map, dtype=np.float32)
        else:
            normalized = (safe_map - min_val) / (max_val - min_val)

        if sigma > 0.0:
            return gaussian_filter(normalized, sigma=float(sigma)).astype(np.float32, copy=False)
        return normalized.astype(np.float32, copy=False)

    def process(self, dist_thresh: float = 0.2, flux_merge_thresh: float = 5.0) -> np.ndarray:
        """Run the full segmentation pipeline and return integer labels."""
        dist_thresh = float(np.clip(float(dist_thresh), 0.0, 1.0))
        flux_merge_thresh = max(0.0, float(flux_merge_thresh))
        n_pixels = self.h * self.w

        self.next_indices = _compute_steepest_neighbor_map(
            self.dist_flat,
            self.h,
            self.w,
            dist_thresh,
        )

        self.flux_map = _compute_flux_topology(self.next_indices, n_pixels)

        raw_dests = _resolve_paths_to_labels(self.next_indices)
        is_bg = self.dist_flat <= dist_thresh
        raw_dests[is_bg] = -1

        if flux_merge_thresh > 0.0:
            merged_dests = _merge_regions_numba(
                raw_dests,
                self.flux_map,
                self.h,
                self.w,
                flux_merge_thresh,
            )
        else:
            merged_dests = raw_dests

        self.labels, self.attractors = self._post_process_labels(merged_dests, is_bg)
        return self.labels

    def _post_process_labels(self, dests_flat: np.ndarray, is_bg_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Provide an internal helper for post process labels.
        
        Args:
            dests_flat (np.ndarray): Value for dests flat interpreted by this routine.
            is_bg_flat (np.ndarray): Boolean flag that toggles is bg flat behavior.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: Array containing the post process labels output aligned with downstream expectations."""
        labels_2d = np.zeros((self.h, self.w), dtype=np.int32)
        attractors_2d = np.zeros((self.h, self.w), dtype=np.int32)

        valid_mask = ~is_bg_flat & (dests_flat != -1)
        valid_dests = dests_flat[valid_mask]
        if valid_dests.size == 0:
            return labels_2d, attractors_2d

        unique_ids, inverse, counts = np.unique(valid_dests, return_inverse=True, return_counts=True)

        temp_labels = np.zeros(self.h * self.w, dtype=np.int32)
        temp_labels[valid_mask] = inverse + 1
        labels_2d = temp_labels.reshape(self.h, self.w)

        for idx, count in zip(unique_ids, counts):
            r, c = divmod(int(idx), self.w)
            attractors_2d[r, c] = int(count)

        return labels_2d, attractors_2d

    def get_results(self) -> dict[str, np.ndarray | None]:
        """Return results in the format expected by downstream consumers.
        
        Returns:
            dict[str, np.ndarray | None]: Array containing the transformed or derived data."""
        flux_2d = None
        if self.flux_map is not None:
            flux_2d = self.flux_map.reshape(self.h, self.w)
        return {
            "dist_map": self.dist_map,
            "labels": self.labels,
            "flux": flux_2d,
            "attractors": self.attractors,
        }


def segment_fastest_ascend(
    dist_map: np.ndarray,
    *,
    sigma: float = 1.0,
    dist_thresh: float = 0.2,
    flux_merge_thresh: float = 5.0,
) -> np.ndarray:
    """Convenience API used by FiberAnalyzer."""
    config = FastestAscendConfig(
        sigma=float(sigma),
        dist_thresh=float(dist_thresh),
        flux_merge_thresh=float(flux_merge_thresh),
    ).normalized()

    segmentor = FluxSegmentor(dist_map=dist_map, sigma=config.sigma)
    return segmentor.process(
        dist_thresh=config.dist_thresh,
        flux_merge_thresh=config.flux_merge_thresh,
    )
