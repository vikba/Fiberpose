from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

INF = float("inf")


@dataclass
class FiberFragment:
    """Polyline fragment with two endpoints used for stitching and segmentation."""

    fragment_id: int
    centerline_xy: np.ndarray
    radii: Optional[np.ndarray] = None
    parent_fiber_id: int = -1
    _endpoints: Tuple["FragmentEndpoint", "FragmentEndpoint"] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.centerline_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("centerline_xy must be an array of shape (N, 2).")
        if pts.shape[0] < 2:
            raise ValueError("Each fragment must contain at least two points.")
        self.centerline_xy = pts

        if self.radii is None:
            radii = np.zeros(len(pts), dtype=np.float32)
        else:
            radii = np.asarray(self.radii, dtype=np.float32).reshape(-1)
            if radii.shape[0] != pts.shape[0]:
                raise ValueError("radii length must match centerline length.")
        self.radii = radii
        self.parent_fiber_id = int(self.parent_fiber_id)

        self._endpoints = (
            FragmentEndpoint(self, "a"),
            FragmentEndpoint(self, "b"),
        )

    def endpoints(self) -> Tuple["FragmentEndpoint", "FragmentEndpoint"]:
        return self._endpoints

    @property
    def points_xy(self) -> np.ndarray:
        return self.centerline_xy

    def length(self) -> float:
        if len(self.centerline_xy) < 2:
            return 0.0
        diffs = np.diff(self.centerline_xy, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())


class FragmentEndpoint:
    """Endpoint with local direction estimation suitable for stitching heuristics."""

    __slots__ = ("fragment", "end", "connected_to", "dir", "_fit_N")

    def __init__(self, fragment: FiberFragment, end: str, fit_N: int = 100) -> None:
        self.fragment = fragment
        if end not in ("a", "b"):
            raise ValueError("Endpoint end must be either 'a' or 'b'.")
        self.end = end
        self.connected_to: Optional["FragmentEndpoint"] = None
        self._fit_N = int(max(2, fit_N))
        self.dir = self._estimate_direction()

    def _points_for_fit(self) -> np.ndarray:
        pts = self.fragment.centerline_xy
        n = min(len(pts), self._fit_N)
        if self.end == "a":
            return pts[:n].copy()
        return pts[-n:][::-1].copy()

    def _estimate_direction(self) -> np.ndarray:
        pts = self._points_for_fit()
        mu = pts.mean(axis=0)
        centered = pts - mu
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            direction = Vt[0]
        except np.linalg.LinAlgError:
            direction = pts[-1] - pts[0]
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        seed = pts[0] - pts[1] if len(pts) >= 2 else np.array([1.0, 0.0], np.float32)
        if np.dot(direction, seed) < 0:
            direction = -direction
        return direction.astype(np.float32)

    def index(self) -> int:
        return 0 if self.end == "a" else len(self.fragment.centerline_xy) - 1

    def pos(self) -> np.ndarray:
        return self.fragment.centerline_xy[self.index()]

    def radius(self) -> float:
        return float(self.fragment.radii[self.index()])

    def last_n_points(self, n: int) -> np.ndarray:
        pts = self.fragment.centerline_xy
        if self.end == "a":
            return pts[: min(n, len(pts))].copy()
        return pts[-min(n, len(pts)) :][::-1].copy()

    def same_fragment(self, other: "FragmentEndpoint") -> bool:
        return self.fragment.fragment_id == other.fragment.fragment_id

    def is_facing(
        self,
        other: "FragmentEndpoint",
        *,
        cos_min: float = 0.95,
    ) -> bool:
        p1 = self.pos().astype(np.float32)
        p2 = other.pos().astype(np.float32)
        v = p2 - p1
        d = float(np.linalg.norm(v))
        if d < 1e-9:
            return False
        u12 = v / d
        u21 = -u12
        d1 = self.dir / (np.linalg.norm(self.dir) + 1e-12)
        d2 = other.dir / (np.linalg.norm(other.dir) + 1e-12)
        return (float(np.dot(d1, u12)) >= cos_min) and (float(np.dot(d2, u21)) >= cos_min)

    def disconnect(self) -> None:
        if self.connected_to is not None:
            other = self.connected_to
            self.connected_to = None
            if other.connected_to is self:
                other.connected_to = None

    def connect(self, other: "FragmentEndpoint") -> None:
        if other is self or self.same_fragment(other):
            return
        if self.connected_to is other and other.connected_to is self:
            return
        self.disconnect()
        other.disconnect()
        self.connected_to = other
        other.connected_to = self


__all__ = ["FiberFragment", "FragmentEndpoint", "INF"]
