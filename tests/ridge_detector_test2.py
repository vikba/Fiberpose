#python -m tests.ridge_detector_test2 tests/dist_2.png tests/ridges.png

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from imageio.v3 import imread, imwrite
from scipy.ndimage import gaussian_filter, generate_binary_structure, label


def _bresenham_line(r0: int, c0: int, r1: int, c1: int):
    """Yield integer (r,c) points along a Bresenham line."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        yield r, c
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between angles (radians), modulo pi."""
    # Ridge orientation is axial (theta == theta + pi)
    d = (a - b) % np.pi
    if d > (np.pi / 2):
        d = np.pi - d
    return abs(d)


def _bridge_gaps(
    mask: np.ndarray,
    candidate: np.ndarray,
    angle: np.ndarray,
    max_gap: int,
    angle_tol_rad: float,
):
    """Bridge short gaps by connecting compatible endpoints through candidate pixels."""
    if max_gap <= 0:
        return mask

    h, w = mask.shape
    structure = generate_binary_structure(2, 2)

    # Endpoints: pixels with <=1 neighbor in 8-neighborhood
    # Count neighbors by convolution via shifts (fast enough for typical images)
    neigh = np.zeros_like(mask, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ys0 = max(0, -dy)
            ys1 = min(h, h - dy)
            xs0 = max(0, -dx)
            xs1 = min(w, w - dx)
            neigh[ys0:ys1, xs0:xs1] += mask[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx]
    endpoints = np.argwhere(mask & (neigh <= 1))
    if endpoints.shape[0] < 2:
        return mask

    # For each endpoint, try to connect to nearest compatible endpoint within max_gap
    used = np.zeros(endpoints.shape[0], dtype=bool)
    for i, (r0, c0) in enumerate(endpoints):
        if used[i]:
            continue
        a0 = float(angle[r0, c0])
        best_j = -1
        best_d2 = None
        for j in range(i + 1, endpoints.shape[0]):
            if used[j]:
                continue
            r1, c1 = endpoints[j]
            dr = int(r1) - int(r0)
            dc = int(c1) - int(c0)
            if abs(dr) > max_gap or abs(dc) > max_gap:
                continue
            d2 = dr * dr + dc * dc
            if d2 == 0 or d2 > (max_gap * max_gap):
                continue
            a1 = float(angle[r1, c1])
            if _angle_diff(a0, a1) > angle_tol_rad:
                continue
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_j = j

        if best_j >= 0:
            r1, c1 = endpoints[best_j]
            # Only bridge if the path stays within candidate pixels (weak or strong)
            ok = True
            for rr, cc in _bresenham_line(int(r0), int(c0), int(r1), int(c1)):
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    ok = False
                    break
                if not (candidate[rr, cc] or mask[rr, cc]):
                    ok = False
                    break
            if ok:
                for rr, cc in _bresenham_line(int(r0), int(c0), int(r1), int(c1)):
                    mask[rr, cc] = True
                used[i] = True
                used[best_j] = True

    # After bridging, ensure connectivity is still consistent
    lbl, n_lbl = label(mask, structure=structure)
    if n_lbl <= 0:
        return mask
    return mask


def load_grayscale_image(path: Path) -> np.ndarray:
    image = imread(path)
    if image.ndim == 3:
        image = np.dot(image[..., :3], np.array([0.299, 0.587, 0.114]))
    image = image.astype(np.float64)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def smooth_image(image: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(image, sigma=sigma) if sigma > 0.0 else image


def steger_ridge_2d(
    img: np.ndarray,
    sigma: float = 1.0,
    mode: str = "bright",
    hess_thresh: float = 1e-4,
    ratio_thresh: float = 2.0,
    hyst_high: Optional[float] = None,
    boundary_mode: str = "reflect",
    max_gap: int = 2,
    angle_tol_deg: float = 30.0,
    min_length: int = 0,
):
    """
    Steger C. (1998): subpixel ridge (line) detection in 2D.
    Parameters
    ----------
    img : 2D ndarray (float)
        Grayscale image (e.g., a distance transform).
    sigma : float
        Scale (std) for Gaussian derivatives. Choose ~ line half-width.
    mode : {"bright","dark"}
        Detect bright lines on dark background or vice versa.
    hess_thresh : float
        Minimum |n^T H n| to accept (suppresses flat/noisy areas).

    Returns
    -------
    coords : (N, 2) float array
        Subpixel ridge coordinates as (row, col).
    mask : 2D bool array
        Pixel mask of accepted ridge points (pre-subpixel).
    """
    img = img.astype(np.float64)
    sigma = float(sigma)

    # Gaussian derivatives (use reflect/mirror boundaries to reduce edge artifacts)
    Ix = gaussian_filter(img, sigma=sigma, order=(1, 0), mode=boundary_mode)
    Iy = gaussian_filter(img, sigma=sigma, order=(0, 1), mode=boundary_mode)

    Ixx = gaussian_filter(img, sigma=sigma, order=(2, 0), mode=boundary_mode)
    Ixy = gaussian_filter(img, sigma=sigma, order=(1, 1), mode=boundary_mode)
    Iyy = gaussian_filter(img, sigma=sigma, order=(0, 2), mode=boundary_mode)

    # Scale-normalized derivatives (many Steger implementations do this)
    if sigma > 0.0:
        Ix *= sigma
        Iy *= sigma
        s2 = sigma * sigma
        Ixx *= s2
        Ixy *= s2
        Iyy *= s2

    H = np.stack(
        [np.stack([Ixx, Ixy], axis=-1), np.stack([Ixy, Iyy], axis=-1)],
        axis=-2,
    )
    evals, evecs = np.linalg.eigh(H)

    if mode == "bright":
        # bright ridge => most negative curvature along the normal
        idx = 0
        sign_ok = lambda d: d < 0
        ridge_eval = evals[..., 0]
        other_eval = evals[..., 1]
    else:
        # dark ridge => most positive curvature along the normal
        idx = 1
        sign_ok = lambda d: d > 0
        ridge_eval = evals[..., 1]
        other_eval = evals[..., 0]

    n = evecs[..., idx]
    nx, ny = n[..., 0], n[..., 1]

    # Ridge orientation (tangent direction), axial (theta == theta + pi)
    tx, ty = -ny, nx
    angle = np.arctan2(ty, tx) % np.pi

    # Tangent orientation (axial). Used for linking/bridging like the Java/C code.
    tx, ty = -ny, nx
    angle = np.arctan2(ty, tx) % np.pi

    Hn_x = Ixx * nx + Ixy * ny
    Hn_y = Ixy * nx + Iyy * ny
    denom = nx * Hn_x + ny * Hn_y
    numer = Ix * nx + Iy * ny

    with np.errstate(divide="ignore", invalid="ignore"):
        r = -numer / denom

    # Line-likeness: strong curvature along the normal, weak along tangent
    # (suppresses blobs / textured noise)
    line_mask = np.abs(ridge_eval) >= ratio_thresh * (np.abs(other_eval) + 1e-12)

    base_mask = (
        (np.abs(r) <= 0.5)
        & np.isfinite(r)
        & sign_ok(denom)
        & (np.abs(denom) >= hess_thresh)
        & line_mask
    )

    # Hysteresis on ridge strength to reduce gaps.
    # Keep all "strong" points; keep "weak" points only if connected to strong ones.
    strength = np.abs(ridge_eval)
    t_low = float(hess_thresh)
    t_high = float(hyst_high) if hyst_high is not None else (2.0 * t_low)
    strong = base_mask & (strength >= t_high)
    weak = base_mask & (strength >= t_low)

    if np.any(strong):
        structure = generate_binary_structure(2, 2)  # 8-connected
        lbl, n_lbl = label(weak, structure=structure)
        if n_lbl > 0:
            keep = np.zeros(n_lbl + 1, dtype=bool)
            keep_ids = np.unique(lbl[strong])
            keep[keep_ids] = True
            mask = keep[lbl]
        else:
            mask = strong
    else:
        # If nothing is strong, fall back to weak (otherwise output empty)
        mask = weak

    # Gap bridging: the ImageJ/Java implementation links pixels into contours and
    # effectively bridges short breaks. We approximate that here by connecting
    # nearby compatible endpoints through the weak-candidate set.
    angle_tol_rad = np.deg2rad(float(angle_tol_deg))
    mask = _bridge_gaps(mask.copy(), weak, angle, max_gap=int(max_gap), angle_tol_rad=angle_tol_rad)

    # Prune short fragments (Java does this after linking via minLength).
    if min_length and min_length > 1:
        structure = generate_binary_structure(2, 2)
        lbl, n_lbl = label(mask, structure=structure)
        if n_lbl > 0:
            counts = np.bincount(lbl.ravel())
            keep = counts >= int(min_length)
            keep[0] = False
            mask = keep[lbl]

    rr, cc = np.indices(img.shape)
    sub_r = rr + r * nx
    sub_c = cc + r * ny

    coords = np.stack([sub_r[mask], sub_c[mask]], axis=1)
    return coords, mask


def detect_ridge_points(image: np.ndarray, sigma: float, hess_thresh: float, mode: str) -> np.ndarray:
    coords, _ = steger_ridge_2d(image, sigma=sigma, mode=mode, hess_thresh=hess_thresh)
    return coords


def draw_overlay(image: np.ndarray, ridge_points: np.ndarray, radius: int) -> np.ndarray:
    height, width = image.shape
    overlay = np.repeat(image[..., None], 3, axis=2)
    for row, col in ridge_points:
        rr = int(round(row))
        cc = int(round(col))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                r_idx, c_idx = rr + dy, cc + dx
                if 0 <= r_idx < height and 0 <= c_idx < width:
                    overlay[r_idx, c_idx] = np.array([1.0, 0.0, 0.0])
    return (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect ridges using Steger's method and save an overlay.")
    parser.add_argument("image_path", type=Path, help="Input image path.")
    parser.add_argument("output_path", type=Path, help="Output image path.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian derivative scale.")
    parser.add_argument("--hess-thresh", type=float, default=1e-3, help="Minimum |nᵀHn| to accept a ridge point.")
    parser.add_argument(
        "--ratio-thresh",
        type=float,
        default=2.0,
        help="Eigenvalue ratio threshold to suppress blob/noise (higher = cleaner, more gaps).",
    )
    parser.add_argument(
        "--hyst-high",
        type=float,
        default=None,
        help="High threshold for hysteresis on ridge strength (default: 2*hess-thresh).",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=("reflect", "mirror", "nearest", "constant", "wrap"),
        default="reflect",
        help="Boundary mode for Gaussian derivatives.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Bridge endpoint gaps up to this many pixels (0 disables).",
    )
    parser.add_argument(
        "--angle-tol-deg",
        type=float,
        default=30.0,
        help="Max orientation mismatch (degrees) when bridging gaps.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="Remove connected ridge fragments shorter than this (pixels).",
    )
    parser.add_argument(
        "--mode",
        choices=("bright", "dark"),
        default="bright",
        help="Detect bright-on-dark or dark-on-bright ridges.",
    )
    parser.add_argument("--radius", type=int, default=1, help="Overlay radius in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_grayscale_image(args.image_path)
    radius = max(args.radius, 0)
    ridge_points, _ = steger_ridge_2d(
        image,
        sigma=args.sigma,
        mode=args.mode,
        hess_thresh=args.hess_thresh,
        ratio_thresh=args.ratio_thresh,
        hyst_high=args.hyst_high,
        boundary_mode=args.boundary_mode,
        max_gap=args.max_gap,
        angle_tol_deg=args.angle_tol_deg,
        min_length=args.min_length,
    )
    overlay = draw_overlay(image, ridge_points, radius)
    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(output_path.as_posix(), overlay)
    print(f"Saved Steger ridge overlay with {ridge_points.shape[0]} points to {output_path}")


if __name__ == "__main__":
    main()
