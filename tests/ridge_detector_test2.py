#python -m tests.ridge_detector_test2 tests/dist_2.png tests/ridges.png

import argparse
from pathlib import Path

import numpy as np
from imageio.v3 import imread, imwrite
from scipy.ndimage import gaussian_filter


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


def steger_ridge_2d(img, sigma=1.0, mode="bright", hess_thresh=1e-4):
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

    Ix = gaussian_filter(img, sigma=sigma, order=(1, 0), mode="nearest")
    Iy = gaussian_filter(img, sigma=sigma, order=(0, 1), mode="nearest")

    Ixx = gaussian_filter(img, sigma=sigma, order=(2, 0), mode="nearest")
    Ixy = gaussian_filter(img, sigma=sigma, order=(1, 1), mode="nearest")
    Iyy = gaussian_filter(img, sigma=sigma, order=(0, 2), mode="nearest")

    H = np.stack(
        [np.stack([Ixx, Ixy], axis=-1), np.stack([Ixy, Iyy], axis=-1)],
        axis=-2,
    )
    evals, evecs = np.linalg.eigh(H)

    if mode == "bright":
        idx = 0
        sign_ok = lambda d: d < 0
    else:
        idx = 1
        sign_ok = lambda d: d > 0

    n = evecs[..., idx]
    nx, ny = n[..., 0], n[..., 1]

    Hn_x = Ixx * nx + Ixy * ny
    Hn_y = Ixy * nx + Iyy * ny
    denom = nx * Hn_x + ny * Hn_y
    numer = Ix * nx + Iy * ny

    with np.errstate(divide="ignore", invalid="ignore"):
        r = -numer / denom

    mask = (np.abs(r) <= 0.5) & sign_ok(denom) & (np.abs(denom) >= hess_thresh) & np.isfinite(r)

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
    ridge_points = detect_ridge_points(
        image,
        args.sigma,
        args.hess_thresh,
        args.mode,
    )
    overlay = draw_overlay(image, ridge_points, radius)
    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(output_path.as_posix(), overlay)
    print(f"Saved Steger ridge overlay with {ridge_points.shape[0]} points to {output_path}")


if __name__ == "__main__":
    main()
