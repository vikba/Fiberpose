"""
Draw cross-sectional lines orthogonal to ridge centerlines on a distance-transform (DT) image.

Pipeline:
  1) Load DT image (float). Values are assumed to be radii in pixels.
  2) Detect ridge centerlines using RidgeDetector (ImageJ Steger via JPype).
  3) For each centerline, resample points every --spacing pixels.
  4) Compute normals at those points via fixed-point refinement using DT gradients.
  5) Draw cross-sections of length 2r at each sample (r = DT at point) along the normal.

Notes:
  - Coordinates follow (y, x) for math, but OpenCV draw functions take (x, y).
  - DT gradients are computed once per image and bilinearly sampled.
  - Cross-sections are clipped to image bounds.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from typing import List, Tuple
from sources.ridgedetector import RidgeDetector
# NEW: import shared helpers from utils
from sources.utils import (
    normals_from_dt_2d,
    resample_polyline_xy,
    clamp_to_bounds_xy,
    _bilinear,
)


# -------------------- IO / visualization --------------------
def load_dt_image(path: str) -> np.ndarray:
    """
    Load a DT image (e.g., float TIFF/PNG). Accepts 8/16/32-bit, 1 or 3 channels.
    Returns float32 HxW array (pixel units assumed).
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read DT image: {path}")
    if img.ndim == 3:
        # Convert to grayscale if RGB(A)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dt = img.astype(np.float32)
    return dt


def make_dt_background_bgr(dt: np.ndarray) -> np.ndarray:
    """
    Create a BGR background for overlay. Scales DT to 0..255 grayscale.
    """
    dt_min, dt_max = float(np.min(dt)), float(np.max(dt))
    if dt_max <= dt_min + 1e-9:
        norm = np.zeros_like(dt, dtype=np.uint8)
    else:
        norm = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    return bgr


# -------------------- Main processing --------------------
def detect_centerlines_from_dt(
    dt: np.ndarray,
    sigma: float = 2.0,
    upper: float = 0.17,
    lower: float = 0.0,
    minLength: float = 10.0,
    maxLength: float = 0.0,
    isDarkLine: bool = False,   # DT ridges are typically bright -> detect bright lines
    doCorrectPosition: bool = True,
    doEstimateWidth: bool = False,
    doExtendLine: bool = True,
    overlapOption: str = "NONE",
):
    """
    Use RidgeDetector to extract ridge polylines (centerlines) from DT.
    Returns a list of polylines as (N,2) XY float arrays.
    """
    rd = RidgeDetector()
    contours, _junctions, _params = rd.detect_lines_contours(
        image=dt,
        sigma=sigma,
        upper_thresh=upper,
        lower_thresh=lower,
        min_length=minLength,
        max_length=maxLength,
        is_dark_line=isDarkLine,
        do_correct_position=doCorrectPosition,
        do_estimate_width=doEstimateWidth,
        do_extend_line=doExtendLine,
        overlap_option=overlapOption,
        bechatty=False,
    )
    # Convert OpenCV-contour-style (N,1,2) int arrays into (N,2) XY float arrays
    polylines_xy = []
    for cnt in contours:
        pts_xy = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts_xy) >= 2:
            polylines_xy.append(pts_xy)
    return polylines_xy


def draw_cross_sections(
    dt: np.ndarray,
    polylines_xy: List[np.ndarray],
    spacing: float = 10.0,
    alpha: float = 0.5,
    iters: int = 3,
    tau_g: float = 1e-3,
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 1,
    dt_sigma: float = 1.0,       # NEW: Gaussian sigma for DT smoothing (px). <=0 disables.
    grad_sigma: float = 1.0,     # NEW: Gaussian sigma for gradient smoothing (px). <=0 disables.
) -> np.ndarray:
    """
    For each polyline:
      - resample at 'spacing' pixels,
      - compute normals from DT gradients,
      - draw cross-sections with length 2r where r = DT at sample.
    Returns a BGR overlay image.
    """
    h, w = dt.shape
    overlay = make_dt_background_bgr(dt)

    # Smooth DT before computing gradients (recommended to reduce noise)
    if dt_sigma and dt_sigma > 0:
        dt_for_grad = cv2.GaussianBlur(
            dt, ksize=(0, 0), sigmaX=float(dt_sigma), sigmaY=float(dt_sigma),
            borderType=cv2.BORDER_REPLICATE
        )
    else:
        dt_for_grad = dt

    # Compute DT gradients once (NumPy returns [d/dy, d/dx] order)
    gy, gx = np.gradient(dt.astype(np.float32))  # gy: rows(y), gx: cols(x)

    # Smooth gradients to stabilize normal estimation
    if grad_sigma and grad_sigma > 0:
        gx = cv2.GaussianBlur(
            gx, ksize=(0, 0), sigmaX=float(grad_sigma), sigmaY=float(grad_sigma),
            borderType=cv2.BORDER_REPLICATE
        )
        gy = cv2.GaussianBlur(
            gy, ksize=(0, 0), sigmaX=float(grad_sigma), sigmaY=float(grad_sigma),
            borderType=cv2.BORDER_REPLICATE
        )

    for poly_xy in polylines_xy:
        # 1) Resample to equal arc-length spacing
        samples_xy = resample_polyline_xy(poly_xy, step=spacing)
        if len(samples_xy) < 2:
            continue

        # 2) Convert to (y,x) for the normals routine
        samples_yx = samples_xy[:, [1, 0]].copy()

        # 3) Compute normals at sampled points using DT gradients
        #    NOTE: normals_from_dt_2d expects (gx, gy) in the "x, y" order.
        normals_yx = normals_from_dt_2d(
            centerline=samples_yx,
            dt=dt,
            gx=gx,
            gy=gy,
            alpha=alpha,
            iters=iters,
            tau_g=tau_g,
        )

        # 4) Draw cross-sections of length 2r along each normal
        for k in range(len(samples_yx)):
            y, x = float(samples_yx[k, 0]), float(samples_yx[k, 1])
            n_yx = normals_yx[k]

            # Local radius from DT at the sample location
            r = float(_bilinear(dt, y, x))
            if r <= 0.5:
                continue  # too small to draw visibly

            # Endpoints in (y,x); convert to (x,y) for drawing
            p0_yx = np.array([y, x]) - r * n_yx
            p1_yx = np.array([y, x]) + r * n_yx
            p0_xy = np.array([p0_yx[1], p0_yx[0]])
            p1_xy = np.array([p1_yx[1], p1_yx[0]])

            # Clip to image bounds
            p0_xy = clamp_to_bounds_xy(p0_xy, w, h)
            p1_xy = clamp_to_bounds_xy(p1_xy, w, h)

            # Draw line
            p0 = tuple(np.round(p0_xy).astype(int))
            p1 = tuple(np.round(p1_xy).astype(int))
            cv2.line(overlay, p0, p1, color=line_color, thickness=line_thickness, lineType=cv2.LINE_AA)

    return overlay


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Draw cross-sections orthogonal to ridge centerlines on a distance-transform image."
    )
    ap.add_argument("image", help="Path to distance-transform (DT) image (single-channel).")
    ap.add_argument("--out", default="dt_cross_sections.png", help="Output overlay image path.")
    # RidgeDetector params (sane defaults for DT ridges as bright lines)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--upper", type=float, default=0.17)
    ap.add_argument("--lower", type=float, default=0.0)
    ap.add_argument("--minLength", type=float, default=10.0)
    ap.add_argument("--maxLength", type=float, default=0.0)
    ap.add_argument("--dark", action="store_true", help="Set if your DT ridges are dark (default: False).")
    ap.add_argument("--noCorrect", action="store_true", help="Disable position correction.")
    ap.add_argument("--estWidth", action="store_true", help="Estimate ridge width (unused here).")
    ap.add_argument("--noExtend", action="store_true", help="Disable line extension.")
    ap.add_argument("--overlap", default="NONE", choices=["NONE", "MERGE"], help="Overlap option for detector.")
    # Cross-section sampling/normal params
    ap.add_argument("--spacing", type=float, default=10.0, help="Arc-length spacing along centerlines (pixels).")
    ap.add_argument("--alpha", type=float, default=0.5, help="Probe step as fraction of local radius r (h=alpha*r).")
    ap.add_argument("--iters", type=int, default=3, help="Fixed-point iterations for normal refinement.")
    ap.add_argument("--tauG", type=float, default=1e-3, help="Threshold on |(g+ - g-)| for stable normals.")
    ap.add_argument("--thickness", type=int, default=1, help="Line thickness for cross-sections.")
    # NEW: smoothing controls
    ap.add_argument("--dtSigma", type=float, default=1.0, help="Gaussian sigma to smooth DT before gradients (px).")
    ap.add_argument("--gradSigma", type=float, default=1.0, help="Gaussian sigma to smooth gradients gx, gy (px).")
    args = ap.parse_args(argv)

    # 1) Load DT
    dt = load_dt_image(args.image)

    # 2) Detect ridge centerlines on DT
    polylines_xy = detect_centerlines_from_dt(
        dt=dt,
        sigma=args.sigma,
        upper=args.upper,
        lower=args.lower,
        minLength=args.minLength,
        maxLength=args.maxLength,
        isDarkLine=bool(args.dark),
        doCorrectPosition=not args.noCorrect,
        doEstimateWidth=bool(args.estWidth),
        doExtendLine=not args.noExtend,
        overlapOption=args.overlap,
    )

    # 3) Draw cross-sections with 10px spacing (configurable)
    overlay = draw_cross_sections(
        dt=dt,
        polylines_xy=polylines_xy,
        spacing=args.spacing,
        alpha=args.alpha,
        iters=args.iters,
        tau_g=args.tauG,
        line_color=(0, 255, 0),
        line_thickness=args.thickness,
        dt_sigma=args.dtSigma,        # NEW
        grad_sigma=args.gradSigma,    # NEW
    )

    # 4) Save result
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, overlay)
    print(f"Centerlines: {len(polylines_xy)}")
    print(f"Saved overlay: {args.out}")


if __name__ == "__main__":
    main()
