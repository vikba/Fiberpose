#!/usr/bin/env python3
"""
Pipeline:
1) Load PNG -> grayscale float32
2) Gaussian blur (sigma=1 px by default)
3) Compute gradient: dy, dx
4) Detect centerlines (FiberAnalyzer)
5) Assess each ridge/centerline point using DT-selected precomputed disk:
   - choose R = clamp(round(dt*0.5), 1..15)
   - inside disk: ignore low-mag gradients
   - compute orientation concentration using doubled-angle stats (antiparallel matches)
   - keep point if circular std <= sigma_max and enough samples
6) Write DT overlay: kept points in green, removed in red
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

from numba import njit, prange

# ----------------------------
# Disk assessment (Numba core)
# ----------------------------


def precompute_disks(R_max: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute offsets for filled disks with radii 1..R_max.
    Returns:
      starts[R] / ends[R]  (int32) into flat arrays dys/dxs
      dys, dxs  (int16) flat arrays concatenating all offsets
    Note: index 0 is unused.
    """
    starts = np.zeros(R_max + 1, dtype=np.int32)
    ends = np.zeros(R_max + 1, dtype=np.int32)

    all_dy = []
    all_dx = []
    cursor = 0
    for R in range(1, R_max + 1):
        starts[R] = cursor
        rr = R * R
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dy * dy + dx * dx <= rr:
                    all_dy.append(dy)
                    all_dx.append(dx)
                    cursor += 1
        ends[R] = cursor

    dys = np.asarray(all_dy, dtype=np.int16)
    dxs = np.asarray(all_dx, dtype=np.int16)
    return starts, ends, dys, dxs


@njit(cache=True)
def _clamp_int(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=True)
def _sigma_from_resultant(Rbar: float, eps: float) -> float:
    # sigma for orientation (period pi) using doubled angles:
    # sigma = 0.5 * sqrt(-2 ln(Rbar))
    if Rbar < eps:
        return 1e9
    return 0.5 * np.sqrt(-2.0 * np.log(Rbar + eps))


@njit(cache=True)
def assess_point_disk(
    y: int,
    x: int,
    dt: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    dys: np.ndarray,
    dxs: np.ndarray,
    R_max: int = 15,
    mag_min: float = 0.0,
    sigma_max: float = 0.35,   # radians (~20°)
    min_count: int = 20,
    wmax: float = 1e9,
    eps: float = 1e-8,
) -> tuple[np.bool_, np.float32, np.int32]:
    """
    Assess a single point using a DT-selected disk.

    - R = clamp(round(0.5*dt[y,x]), 1..R_max)
    - For offsets in disk: ignore gradient magnitudes < mag_min
    - Use doubled-angle stats without atan2:
        cos(2θ) = (gx^2 - gy^2)/(gx^2 + gy^2)
        sin(2θ) = 2 gx gy /(gx^2 + gy^2)
      So antiparallel gradients map together.
    Returns: (keep, sigma_rad, used_count)
    """
    H, W = dt.shape
    if y < 0 or y >= H or x < 0 or x >= W:
        return False, np.float32(1e9), np.int32(0)

    R = int(np.round(0.5 * float(dt[y, x])))
    R = _clamp_int(R, 1, R_max)

    s0 = starts[R]
    s1 = ends[R]

    C = 0.0
    S = 0.0
    Wsum = 0.0
    used = 0

    for i in range(s0, s1):
        yy = y + int(dys[i])
        xx = x + int(dxs[i])
        if yy < 0 or yy >= H or xx < 0 or xx >= W:
            continue

        gxx = float(gx[yy, xx])
        gyy = float(gy[yy, xx])
        mag2 = gxx * gxx + gyy * gyy
        mag = np.sqrt(mag2)
        if mag < mag_min:
            continue

        denom = mag2 + eps
        c = (gxx * gxx - gyy * gyy) / denom
        s = (2.0 * gxx * gyy) / denom

        w = mag
        if w > wmax:
            w = wmax

        C += w * c
        S += w * s
        Wsum += w
        used += 1

    if used < min_count or Wsum <= eps:
        return False, np.float32(1e9), np.int32(used)

    Rbar = np.sqrt(C * C + S * S) / (Wsum + eps)
    sigma = _sigma_from_resultant(Rbar, eps)
    keep = sigma <= sigma_max
    return keep, np.float32(sigma), np.int32(used)


@njit(parallel=True, cache=True)
def assess_points_disk(
    ys: np.ndarray,
    xs: np.ndarray,
    dt: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    dys: np.ndarray,
    dxs: np.ndarray,
    R_max: int,
    mag_min: float,
    sigma_max: float,
    min_count: int,
    wmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch assess many points (parallel).
    Returns keep_mask (bool), sigma (float32), used_count (int32).
    """
    n = ys.size
    keep = np.zeros(n, dtype=np.bool_)
    sigma = np.empty(n, dtype=np.float32)
    used = np.empty(n, dtype=np.int32)

    for i in prange(n):
        k, s, u = assess_point_disk(
            int(ys[i]), int(xs[i]),
            dt, gx, gy,
            starts, ends, dys, dxs,
            R_max, mag_min, sigma_max, min_count, wmax,
        )
        keep[i] = k
        sigma[i] = s
        used[i] = u
    return keep, sigma, used


# ---------------------------------
# Your original code starts here
# ---------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from sources.fiber_analyzer import FiberAnalyzer
from sources.ridge_detector import RidgeDetector


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur using scipy if available, otherwise skimage."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=sigma)
    except Exception:
        try:
            from skimage.filters import gaussian
            return gaussian(img, sigma=sigma, preserve_range=True)
        except Exception as e:
            raise RuntimeError(
                "Need scipy or scikit-image for Gaussian blur.\n"
                "Install one of:\n  pip install scipy\n  pip install scikit-image"
            ) from e


def to_gray_float32(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale float32."""
    img = np.asarray(img)
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        rgb = img[..., :3].astype(np.float32)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported image shape {img.shape}")
    return gray.astype(np.float32, copy=False)


def normalize99(x: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.0)
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _draw_disk_rgb(img: np.ndarray, y: int, x: int, radius: int, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    r0 = max(0, y - radius)
    r1 = min(h, y + radius + 1)
    c0 = max(0, x - radius)
    c1 = min(w, x + radius + 1)
    if r0 >= r1 or c0 >= c1:
        return
    yy, xx = np.ogrid[r0:r1, c0:c1]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius * radius
    region = img[r0:r1, c0:c1]
    region[mask] = color


class _SimpleLine:
    """Minimal line wrapper matching RidgeDetector output."""
    def __init__(self, pts: np.ndarray):
        self.pts = np.asarray(pts, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_png", help="Input PNG (grayscale/RGB/RGBA)")
    ap.add_argument("-o", "--overlay-output", default=None, help="Output PNG for DT + kept/removed overlay")
    ap.add_argument("--used-output", default=None, help="Output PNG for DT + pixels used in disk assessment")
    ap.add_argument("--mag-output", default=None, help="Output PNG for gradient magnitude")
    ap.add_argument("--point-radius", type=int, default=0.5, help="Radius in pixels for kept/removed points")
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian blur sigma in pixels")

    ap.add_argument("--ridge-sigma", type=float, default=1, help="Sigma for ridge detection")
    ap.add_argument("--ridge-low", type=float, default=0.05, help="Low threshold for ridge response")
    ap.add_argument("--ridge-high", type=float, default=0.15, help="High threshold for ridge response")
    ap.add_argument("--ridge-polarity", choices=["dark", "light"], default="light", help="Ridge polarity")

    # NEW: assessment knobs
    ap.add_argument("--assess-Rmax", type=int, default=15, help="Max disk radius for assessment")
    ap.add_argument("--assess-mag-min", type=float, default=0.1, help="Min gradient magnitude inside disk")
    ap.add_argument("--assess-sigma-max-deg", type=float, default=40.0, help="Max angular std (deg) to keep point")
    ap.add_argument("--assess-min-count", type=int, default=4, help="Min valid disk samples to keep point")
    ap.add_argument("--assess-wmax", type=float, default=50.0, help="Cap magnitude weighting inside disk")

    ap.add_argument("--fiber-max-angle", type=float, default=90.0, help="Max angle (deg) for centerline filtering")
    args = ap.parse_args()

    base = args.input_png.rsplit(".", 1)[0]
    if args.overlay_output is None:
        args.overlay_output = base + "_dt_overlay.png"
    if args.used_output is None:
        args.used_output = base + "_dt_used.png"
    if args.mag_output is None:
        args.mag_output = base + "_grad_mag.png"

    img = imageio.imread(args.input_png)
    gray = to_gray_float32(img)
    gray_blur = gaussian_blur(gray, sigma=args.sigma).astype(np.float32, copy=False)

    # gradients after blur
    dy, dx = np.gradient(gray_blur)
    mag = np.sqrt(dy * dy + dx * dx)
    # Gradient magnitude visualization: ~1 -> green, below -> blue, above -> red
    max_mag = float(mag.max())
    hi = max(1.0, max_mag)
    denom = hi - 1.0
    if denom < 1e-8:
        denom = 1.0

    mag_rgb = np.zeros(mag.shape + (3,), dtype=np.float32)
    low_mask = mag <= 1.0
    high_mask = ~low_mask

    # Below 1: blue -> green
    t_low = np.clip(mag[low_mask], 0.0, 1.0)
    mag_rgb[low_mask, 1] = 255.0 * t_low
    mag_rgb[low_mask, 2] = 255.0 * (1.0 - t_low)

    # Above 1: green -> red
    t_high = np.clip((mag[high_mask] - 1.0) / denom, 0.0, 1.0)
    mag_rgb[high_mask, 0] = 255.0 * t_high
    mag_rgb[high_mask, 1] = 255.0 * (1.0 - t_high)

    imageio.imwrite(args.mag_output, mag_rgb.astype(np.uint8))
    print(f"Wrote {args.mag_output}  shape={mag_rgb.shape}")

    # FiberAnalyzer expects dt_u8 + float dt + gradients gx/gy
    dt_u8 = np.clip(gray_blur, 0, 255).astype(np.uint8)
    ridge_sigma = args.ridge_sigma if args.ridge_sigma is not None else args.sigma

    # fragments, markers = FiberAnalyzer.compute_centerlines(
    #     dt_u8=dt_u8,
    #     dt=gray_blur,
    #     gx=dx,
    #     gy=dy,
    #     max_angle_deg=args.fiber_max_angle,
    #     sigma=ridge_sigma,
    #     low=float(args.ridge_low),
    #     high=float(args.ridge_high),
    #     polarity=args.ridge_polarity,
    #     markers_from_raw_ridges=False,
    # )

    ridge_lines, ridge_junctions = RidgeDetector.detect_ridges(
        gray_blur,
        sigma=ridge_sigma,
        low=float(args.ridge_low),
        high=float(args.ridge_high),
        polarity=args.ridge_polarity,
    )

    lines = [_SimpleLine(line.pts) for line in ridge_lines]
    print(f"Ridge lines: {len(lines)}; junctions: {len(ridge_junctions)}")

    # -------------------------
    # NEW: batch point assessment
    # -------------------------
    Rmax = int(args.assess_Rmax)
    starts, ends, dys, dxs = precompute_disks(R_max=Rmax)

    ys_list = []
    xs_list = []
    line_offsets = []
    cursor = 0
    for line in lines:
        pts = np.asarray(line.pts, dtype=np.float32)
        if pts.size == 0:
            line_offsets.append((cursor, cursor))
            continue
        yy = np.round(pts[:, 0]).astype(np.int32)
        xx = np.round(pts[:, 1]).astype(np.int32)
        ys_list.append(yy)
        xs_list.append(xx)
        line_offsets.append((cursor, cursor + yy.size))
        cursor += yy.size

    if ys_list:
        ys = np.concatenate(ys_list)
        xs = np.concatenate(xs_list)
    else:
        ys = np.zeros(0, dtype=np.int32)
        xs = np.zeros(0, dtype=np.int32)

    sigma_max = np.deg2rad(float(args.assess_sigma_max_deg))

    keep, sigma_vals, used_counts = assess_points_disk(
        ys, xs,
        gray_blur.astype(np.float32, copy=False),  # dt
        dx.astype(np.float32, copy=False),         # gx
        dy.astype(np.float32, copy=False),         # gy
        starts, ends, dys, dxs,
        R_max=Rmax,
        mag_min=float(args.assess_mag_min),
        sigma_max=float(sigma_max),
        min_count=int(args.assess_min_count),
        wmax=float(args.assess_wmax),
    )

    print(f"Disk assessment kept {int(keep.sum())}/{keep.size} points")

    # DT overlay with kept (green) vs removed (red) points
    dt_u8 = (255.0 * normalize99(gray_blur)).astype(np.uint8)
    overlay = np.repeat(dt_u8[..., None], 3, axis=-1)

    r = max(0, int(args.point_radius))
    for i in range(ys.size):
        y0 = int(ys[i])
        x0 = int(xs[i])
        if y0 < 0 or y0 >= overlay.shape[0] or x0 < 0 or x0 >= overlay.shape[1]:
            continue
        if keep[i]:
            _draw_disk_rgb(overlay, y0, x0, r, (0, 255, 0))
        else:
            _draw_disk_rgb(overlay, y0, x0, r, (255, 0, 0))

    imageio.imwrite(args.overlay_output, overlay)
    print(f"Wrote {args.overlay_output}  shape={overlay.shape}")

    # Diagnostics: mark all pixels used in disk assessment (mag >= mag_min)
    used_mask = np.zeros(gray_blur.shape, dtype=bool)
    H, W = gray_blur.shape
    for i in range(ys.size):
        y = int(ys[i])
        x = int(xs[i])
        if y < 0 or y >= H or x < 0 or x >= W:
            continue
        R = int(np.round(0.7 * float(gray_blur[y, x])))
        if R < 1:
            R = 1
        elif R > Rmax:
            R = Rmax
        s0 = int(starts[R])
        s1 = int(ends[R])
        for j in range(s0, s1):
            yy = y + int(dys[j])
            xx = x + int(dxs[j])
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                continue
            if mag[yy, xx] < float(args.assess_mag_min):
                continue
            used_mask[yy, xx] = True

    used_overlay = overlay.copy()
    used_overlay[used_mask] = (255, 255, 0)
    imageio.imwrite(args.used_output, used_overlay)
    print(f"Wrote {args.used_output}  shape={used_overlay.shape}")


if __name__ == "__main__":
    main()
