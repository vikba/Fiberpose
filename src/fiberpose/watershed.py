"""Alignment-merge watershed: DT image -> one label per fiber piece.

DT maxima are flooded into basins, which over-segments every fiber into a chain. Two
touching basins are merged when the vector joining their seed points runs along the fiber
axis (perpendicular to the DT gradient) on their collision line and at both seeds, and the
line really spans a fiber.
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.feature import peak_local_max

_SEED_SIGMA = 1.0  # smoothing of the DT before picking seed maxima
_FILL_MIN_DT = 1.0  # lowest DT maximum that still seeds an unreached area
_SCALES = (1.5, 3.0, 6.0)  # structure-tensor integration scales; ~dt/2 is picked per pixel
_MIN_COHERENCE = 0.3  # mean coherence a collision line needs for its orientation to count
_MIN_CONTACT = 2  # shortest collision line (pixel pairs) that can merge


@njit(cache=True)
def _flood(dt, labels, mask, climb):
    """Grow `labels` in place over `mask`, highest DT first, 8-connected.

    A front never steps up by more than `climb` in DT. No compactness term: it makes
    basins cross the valley and climb the neighbouring fiber's flank.
    """
    h, w = dt.shape
    heap = [(-dt[r, c], i, r, c) for i, (r, c) in enumerate(zip(*np.nonzero(labels)))]
    age = len(heap)
    heapq.heapify(heap)
    while heap:
        priority, _, r, c = heapq.heappop(heap)
        for rr in range(max(r - 1, 0), min(r + 2, h)):
            for cc in range(max(c - 1, 0), min(c + 2, w)):
                if mask[rr, cc] and labels[rr, cc] == 0 and dt[rr, cc] <= dt[r, c] + climb:
                    labels[rr, cc] = labels[r, c]
                    heapq.heappush(heap, (max(-dt[rr, cc], priority), age, rr, cc))
                    age += 1


def _basins(dt, mask, seed_distance, climb_limit):
    """Flood the mask from the DT maxima -> int32 labels 1..n covering the mask."""
    peaks = peak_local_max(
        ndi.gaussian_filter(dt, _SEED_SIGMA), min_distance=seed_distance, labels=mask.astype(np.int32),
        exclude_border=False,
    )
    labels = np.zeros(dt.shape, np.int32)
    labels[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    dt64 = dt.astype(np.float64)
    _flood(dt64, labels, mask, climb_limit)
    # What the climb limit left unreached and has a DT maximum of its own becomes a basin ...
    rest = mask & (labels == 0)
    extra, _ = ndi.label(rest & (dt >= ndi.maximum_filter(dt, 7)) & (dt >= _FILL_MIN_DT), np.ones((3, 3)))
    extra = np.where(extra, extra + labels.max(), 0).astype(np.int32)
    _flood(dt64, extra, rest, np.inf)
    labels += extra
    # ... and the remaining slivers go to whatever basin touches them.
    _flood(dt64, labels, mask, np.inf)
    return labels


def _orientation(dt):
    """Gradient orientation (mod pi, x = column) and coherence from the DT structure tensor."""
    gy, gx = (ndi.gaussian_filter(dt, 1.0, order=order) for order in ((1, 0), (0, 1)))
    scale = np.clip(0.5 * ndi.maximum_filter(dt, 5), _SCALES[0], _SCALES[-1])
    pick = np.abs(np.log(scale) - np.log(_SCALES)[:, None, None]).argmin(0)[None]
    jxx, jyy, jxy = (
        np.take_along_axis(np.stack([ndi.gaussian_filter(j, s) for s in _SCALES]), pick, 0)[0]
        for j in (gx * gx, gy * gy, gx * gy)
    )
    return 0.5 * np.arctan2(2 * jxy, jxx - jyy), np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-9)


def _merges(dt, labels, align_tol_deg, span_factor, tangent_tol_deg):
    """Pairs (a, b) of touching basins that belong to the same fiber, as a (2, k) array."""
    # Collision lines: 4-connected pixel pairs (p, q) with different labels, grouped by label pair.
    index = np.arange(labels.size).reshape(labels.shape)
    p, q = (np.concatenate([index[:, :-1].ravel(), index[:-1].ravel()]),
            np.concatenate([index[:, 1:].ravel(), index[1:].ravel()]))
    flat = labels.ravel()
    touching = (flat[p] != flat[q]) & (flat[p] > 0) & (flat[q] > 0)
    p, q = p[touching], q[touching]
    pairs, line, length = np.unique(np.sort([flat[p], flat[q]], axis=0), axis=1, return_inverse=True, return_counts=True)
    a, b = pairs
    pixels, line, n_lines = np.concatenate([p, q]), np.tile(line.ravel(), 2), len(length)

    ids = np.arange(1, labels.max() + 1)
    seeds = np.array(ndi.maximum_position(dt, labels, ids), int).reshape(-1, 2)  # each basin's highest pixel
    radius = dt[tuple(seeds.T)].astype(float)
    theta, coherence = _orientation(dt)

    # The crossing direction is the seed-to-seed vector, not the collision line itself (a
    # plain flood draws that along grid directions). It runs along the fiber when it is
    # perpendicular to the DT gradient: sin^2 of the angle between them close to 1.
    (ya, xa), (yb, xb) = seeds[a - 1].T, seeds[b - 1].T
    phi = np.arctan2(yb - ya, xb - xa)
    along = lambda angle, tol_deg: np.sin(phi - angle) ** 2 >= np.cos(np.deg2rad(tol_deg)) ** 2  # noqa: E731

    weight = coherence.ravel()[pixels]
    total = np.bincount(line, weight, n_lines)
    score = np.bincount(line, weight * np.sin(phi[line] - theta.ravel()[pixels]) ** 2, n_lines) / (total + 1e-9)
    aligned = (score >= np.cos(np.deg2rad(align_tol_deg)) ** 2) & (total / (2 * length) >= _MIN_COHERENCE)

    # max, not min, of the two radii: it also stops a small basin merging into a big one.
    saddle = np.zeros(n_lines)
    np.maximum.at(saddle, line, dt.ravel()[pixels])
    spans = (saddle >= span_factor * np.maximum(radius[a - 1], radius[b - 1])) \
        & (length >= saddle) & (length >= _MIN_CONTACT)

    tangent = along(theta[ya, xa], tangent_tol_deg) & along(theta[yb, xb], tangent_tol_deg)
    return pairs[:, aligned & spans & tangent]


def segment_fibers(
    dt: np.ndarray,
    *,
    mask_threshold: float = 0.5,
    seed_distance: int = 3,
    climb_limit: float = 0.3,
    align_tol_deg: float = 30.0,
    span_factor: float = 0.5,
    tangent_tol_deg: float = 35.0,
    min_area: int = 40,
) -> np.ndarray:
    """Label the fiber pieces of a distance-transform image: uint16, 1..n, 0 = background."""
    dt = np.asarray(dt, dtype=np.float32)
    labels = _basins(dt, dt > mask_threshold, seed_distance, climb_limit)
    n = int(labels.max()) + 1
    a, b = _merges(dt, labels, align_tol_deg, span_factor, tangent_tol_deg)
    fiber = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)[1]
    keep = np.bincount(fiber[labels].ravel(), minlength=n) >= min_area
    keep[fiber[0]] = False  # background
    return (np.cumsum(keep) * keep)[fiber][labels].astype(np.uint16)
