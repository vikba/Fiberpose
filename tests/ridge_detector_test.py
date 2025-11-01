import argparse
from pathlib import Path

import numpy as np
from imageio.v3 import imread, imwrite
from scipy.ndimage import gaussian_filter

from sources.ridges import ftle_ridge_pts


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


def compute_hessian_components(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fxx = np.zeros_like(field)
    fyy = np.zeros_like(field)
    fxy = np.zeros_like(field)
    fxx[1:-1, :] = field[2:, :] - 2.0 * field[1:-1, :] + field[:-2, :]
    fyy[:, 1:-1] = field[:, 2:] - 2.0 * field[:, 1:-1] + field[:, :-2]
    fxy[1:-1, 1:-1] = (
        field[2:, 2:] - field[2:, :-2] - field[:-2, 2:] + field[:-2, :-2]
    ) * 0.25
    return fxx, fyy, fxy


def estimate_principal_eigvec(field: np.ndarray) -> np.ndarray:
    fxx, fyy, fxy = compute_hessian_components(field)
    hessian = np.empty(field.shape + (2, 2), dtype=np.float64)
    hessian[..., 0, 0] = fxx
    hessian[..., 0, 1] = fxy
    hessian[..., 1, 0] = fxy
    hessian[..., 1, 1] = fyy
    _, eigvecs = np.linalg.eigh(hessian)
    return eigvecs[..., :, 0]


def detect_ridge_points(image: np.ndarray, sigma: float, sdd_thresh: float, percentile: int) -> np.ndarray:
    smoothed = smooth_image(image, sigma)
    field = smoothed.T
    eigvecs = estimate_principal_eigvec(field)
    x = np.arange(field.shape[0], dtype=np.float64)
    y = np.arange(field.shape[1], dtype=np.float64)
    return ftle_ridge_pts(field, eigvecs, x, y, sdd_thresh=sdd_thresh, percentile=percentile)


def draw_overlay(image: np.ndarray, ridge_points: np.ndarray, radius: int) -> np.ndarray:
    height, width = image.shape
    overlay = np.repeat(image[..., None], 3, axis=2)
    for x_coord, y_coord in ridge_points:
        col = int(round(x_coord))
        row = int(round(y_coord))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                rr, cc = row + dy, col + dx
                if 0 <= rr < height and 0 <= cc < width:
                    overlay[rr, cc] = np.array([1.0, 0.0, 0.0])
    return (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect ridges using ftle_ridge_pts and save an overlay.")
    parser.add_argument("image_path", type=Path, help="Input image path.")
    parser.add_argument("output_path", type=Path, help="Output image path.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0, help="Gaussian smoothing sigma.")
    parser.add_argument("--sdd-thresh", type=float, default=0.001, help="Second directional derivative threshold.")
    parser.add_argument("--percentile", type=int, default=0, help="Minimum percentile for ridge selection.")
    parser.add_argument("--radius", type=int, default=1, help="Overlay radius in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_grayscale_image(args.image_path)
    percentile = min(max(args.percentile, 0), 100)
    radius = max(args.radius, 0)
    ridge_points = detect_ridge_points(image, args.gaussian_sigma, args.sdd_thresh, percentile)
    overlay = draw_overlay(image, ridge_points, radius)
    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(output_path.as_posix(), overlay)
    print(f"Saved overlay with {ridge_points.shape[0]} ridge points to {output_path}")


if __name__ == "__main__":
    main()
