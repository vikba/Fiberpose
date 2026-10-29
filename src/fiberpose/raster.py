"""Vectorized disk rasterization that reproduces ``cv2.circle`` exactly.

Several pipeline stages need "the pixels covered by filled circles drawn along a
polyline". Drawing each circle into a full-size mask makes those stages cost
O(objects x image area). The helpers here return pixel indices instead, so the
cost is proportional to the covered area only.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

# Upper bound on (points x stencil pixels) materialized at once by stamp_disks.
_STAMP_CHUNK = 1 << 22


@lru_cache(maxsize=None)
def disk_mask(radius: int) -> np.ndarray:
    """Boolean (2r+1, 2r+1) mask of the pixels ``cv2.circle(..., thickness=-1)`` fills."""
    radius = int(radius)
    canvas = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    cv2.circle(canvas, (radius, radius), radius, 1, thickness=-1)
    mask = canvas.astype(bool)
    mask.flags.writeable = False
    return mask


@lru_cache(maxsize=None)
def disk_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """(dy, dx) offsets of the pixels in ``disk_mask(radius)`` relative to its center."""
    dy, dx = np.nonzero(disk_mask(radius))
    dy = dy - int(radius)
    dx = dx - int(radius)
    dy.flags.writeable = False
    dx.flags.writeable = False
    return dy, dx


def disk_radii(radii: np.ndarray, expansion: float) -> np.ndarray:
    """Integer disk radii ``max(1, round(radius * expansion))`` (round half to even, like ``round``)."""
    scaled = np.asarray(radii, dtype=np.float64) * float(expansion)
    return np.maximum(1, np.rint(scaled).astype(np.int64))


def stamp_disks(
    points_xy: np.ndarray,
    radii_px: np.ndarray,
    owners: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels covered by filled disks, as unique ``(pixel, owner)`` pairs.

    Each point ``i`` contributes a disk of integer radius ``radii_px[i]`` centered at
    ``round(points_xy[i])`` and attributed to ``owners[i]`` (a non-negative id). Centers outside the image
    are skipped and disks are clipped at the image border, matching a loop of
    ``cv2.circle`` calls on an image of ``shape``.

    Returns flat pixel indices (``y * width + x``) and owner ids, sorted by owner then pixel.
    """
    h, w = int(shape[0]), int(shape[1])
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    cx = np.rint(pts[:, 0]).astype(np.int64)
    cy = np.rint(pts[:, 1]).astype(np.int64)
    radii_px = np.asarray(radii_px, dtype=np.int64).reshape(-1)
    owners = np.asarray(owners, dtype=np.int64).reshape(-1)

    inside = (cx >= 0) & (cy >= 0) & (cx < w) & (cy < h)
    cx, cy, radii_px, owners = cx[inside], cy[inside], radii_px[inside], owners[inside]

    keys: list[np.ndarray] = []
    for radius in np.unique(radii_px):
        sel = np.flatnonzero(radii_px == radius)
        dy, dx = disk_offsets(int(radius))
        step = max(1, _STAMP_CHUNK // dy.size)
        for start in range(0, sel.size, step):
            idx = sel[start:start + step]
            ys = cy[idx, None] + dy[None, :]
            xs = cx[idx, None] + dx[None, :]
            valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            pixel = ys[valid] * w + xs[valid]
            owner = np.broadcast_to(owners[idx, None], valid.shape)[valid]
            keys.append(np.unique(owner * (h * w) + pixel))

    if not keys:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    key = np.unique(np.concatenate(keys))
    return key % (h * w), key // (h * w)
