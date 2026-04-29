"""Interaction helpers for scalebar and profile tools."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.measure import profile_line


@dataclass
class ScalebarTool:
    """State machine for selecting a two-point scalebar."""

    length_value: float = 100.0
    unit: str = "um"
    pending: bool = False
    points: list[tuple[float, float]] = field(default_factory=list)

    def start(self, length_value: float, unit: str) -> None:
        self.length_value = float(length_value)
        self.unit = str(unit)
        self.pending = True
        self.points = []

    def cancel(self) -> None:
        self.pending = False
        self.points = []

    def add_point(self, x: float, y: float) -> float | None:
        if not self.pending:
            return None
        self.points.append((float(x), float(y)))
        if len(self.points) < 2:
            return None
        (x0, y0), (x1, y1) = self.points[:2]
        pixel_length = float(np.hypot(x1 - x0, y1 - y0))
        self.pending = False
        self.points = []
        if pixel_length <= 0:
            return None
        return float(self.length_value / pixel_length)


@dataclass
class ProfileTool:
    """Two-point line profile interaction helper."""

    active: bool = False
    points_scene: list[tuple[float, float]] = field(default_factory=list)
    points_img: list[tuple[float, float]] = field(default_factory=list)

    def start(self) -> None:
        self.active = True
        self.points_scene = []
        self.points_img = []

    def stop(self) -> None:
        self.active = False
        self.points_scene = []
        self.points_img = []

    def reset(self) -> None:
        self.points_scene = []
        self.points_img = []

    def add_point(self, scene_xy: tuple[float, float], img_xy: tuple[float, float]) -> bool:
        if len(self.points_scene) >= 2:
            self.reset()
        self.points_scene.append((float(scene_xy[0]), float(scene_xy[1])))
        self.points_img.append((float(img_xy[0]), float(img_xy[1])))
        return len(self.points_scene) == 2

    @staticmethod
    def compute_profile(
        dt: np.ndarray,
        p0_xy: tuple[float, float],
        p1_xy: tuple[float, float],
        *,
        scale_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        x0, y0 = p0_xy
        x1, y1 = p1_xy
        length_px = float(np.hypot(x1 - x0, y1 - y0))
        if length_px <= 0:
            return None

        profile_vals = profile_line(dt, (y0, x0), (y1, x1), mode="reflect", order=1)
        if profile_vals is None or profile_vals.size == 0:
            return None

        profile_vals = np.asarray(profile_vals, dtype=np.float32, copy=False)
        scale = float(scale_factor)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        true_length_px = length_px / scale
        profile_vals_true = profile_vals / scale
        distances = np.linspace(0.0, true_length_px, profile_vals.size, dtype=np.float32)
        delta = float(profile_vals_true[-1] - profile_vals_true[0])
        summary = (
            f"Length: {true_length_px:.2f} px · ΔDT: {delta:+.2f} · "
            f"Min/Max: {float(profile_vals_true.min()):.2f} / {float(profile_vals_true.max()):.2f}"
        )
        return distances, profile_vals_true, summary
