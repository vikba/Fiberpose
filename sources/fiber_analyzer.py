from __future__ import annotations
import os
import numpy as np
import cv2

from scipy.ndimage import gaussian_filter
from skimage.measure import label as cc_label
from skimage.segmentation import watershed

from .ridgedetector import RidgeDetector
from .utils import normals_from_dt_2d, resample_polyline_xy, _bilinear


# ---------- Paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)



class FiberAnalyzer:
    """
    Non-GUI class encapsulating fiber detection, watershed, and analysis.

    Responsibilities:
      - model init and DT prediction (predict_dt)
      - centerline detection (compute_centerlines)
      - watershed labeling (labels_from_dt_and_markers)
      - analysis (analyze): samples, viz points, gradients, angles
    """

    def __init__(self, high_conf_fib: bool):
        self._rd = RidgeDetector(high_conf_fib)

    def set_high_conf_filter(self, high_conf_fib: bool):
        self._rd.set_high_conf_filter(high_conf_fib)


    # ---------- utils (local) ----------

    @staticmethod
    def _to_uint8_for_view(img: np.ndarray | None) -> np.ndarray | None:
        if img is None:
            return None
        arr = img.astype(np.float32, copy=False)
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        amin, amax = float(arr.min()), float(arr.max())
        if amax <= amin:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = (arr - amin) / (amax - amin)
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _ensure_rgb_u8(img: np.ndarray | None) -> np.ndarray | None:
        if img is None:
            return None
        if img.ndim == 2:
            u8 = FiberAnalyzer._to_uint8_for_view(img)
            return np.stack([u8, u8, u8], axis=-1)
        if img.ndim == 3:
            if img.shape[2] == 1:
                ch = FiberAnalyzer._to_uint8_for_view(img[..., 0])
                return np.stack([ch, ch, ch], axis=-1)
            if img.dtype != np.uint8:
                return FiberAnalyzer._to_uint8_for_view(img)[..., :3]
            return img[..., :3]
        return FiberAnalyzer._ensure_rgb_u8(FiberAnalyzer._to_uint8_for_view(img))

    @staticmethod
    def _labels_to_color(labels: np.ndarray) -> np.ndarray:
        labels = labels.astype(np.int64, copy=False)
        h, w = labels.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        if labels.size == 0:
            return out
        unique_vals = np.unique(labels)
        for v in unique_vals:
            if v == 0:
                continue
            r = (v * 37) % 256
            g = (v * 57) % 256
            b = (v * 97) % 256
            out[labels == v] = (r, g, b)
        return out



    # ---------- centerlines / markers / watershed ----------

    def compute_centerlines(
        self,
        dt_u8: np.ndarray,
        max_angle_deg: float,
        sigma: float = 2.23,
        upperThresh: float = 0.17,
        lowerThresh: float = 0.0,
        minLength: float = 10.0,
        maxLength: float = 0.0,
        isDarkLine: bool = False,
        doCorrectPosition: bool = False,
        doEstimateWidth: bool = False,
        doExtendLine: bool = True,
        overlapOption: str = "NONE",
        trim: int = 3,
        tol: float = 3.0,
        win_len: float = 25.0,
        std_thresh: float = 0.8,
    ):
        """
        Returns:
          - constant_segments: list of (N,1,2) int32 polylines
          - normals: list of (N,2) float normals per segment
          - centerlines_rgb: visualization on DT
          - markers: int32 labeled markers for watershed
        """
        constant_segments_xy, normals_xy = self._rd.detect_constant_segments(
            image=dt_u8,
            sigma=sigma, upperThresh=upperThresh, lowerThresh=lowerThresh,
            minLength=minLength, maxLength=maxLength,
            isDarkLine=isDarkLine, doCorrectPosition=doCorrectPosition,
            doEstimateWidth=doEstimateWidth, doExtendLine=doExtendLine,
            overlapOption=overlapOption, trim=trim, tol=tol,
            win_len=win_len, std_thresh=std_thresh, max_angle=max_angle_deg
        )

        base_dt = self._to_uint8_for_view(dt_u8)
        centerlines_rgb = cv2.cvtColor(base_dt, cv2.COLOR_GRAY2BGR)
        # RidgeDetector delivers polylines in (x, y) order; the `_xy` suffix keeps that explicit.
        for seg_xy in constant_segments_xy:
            cv2.polylines(centerlines_rgb, [seg_xy.astype(np.int32)], isClosed=False,
                          color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        centerlines_binary = np.zeros(dt_u8.shape, dtype=np.uint8)
        for seg_xy in constant_segments_xy:
            cv2.polylines(centerlines_binary, [seg_xy.astype(np.int32)], isClosed=False,
                          color=255, thickness=2, lineType=cv2.LINE_8)
        _, centerlines_binary = cv2.threshold(centerlines_binary, 128, 255, cv2.THRESH_BINARY)
        markers = cc_label(centerlines_binary, connectivity=2).astype(np.int32)
        return constant_segments_xy, normals_xy, centerlines_rgb, markers

    def labels_from_dt_and_markers(self, dt_u8: np.ndarray, markers: np.ndarray, thr_u8: int) -> np.ndarray:
        """
        Watershed on inverted DT with binary mask from threshold and given markers.
        """
        if markers is None or markers.max() == 0:
            return np.zeros_like(dt_u8, dtype=np.uint16)
        t = float(thr_u8) / 255.0
        mask2 = (dt_u8.astype(np.float32) / 255.0) >= t
        elev = 1.0 - (dt_u8.astype(np.float32) / 255.0)
        if not mask2.any():
            return np.zeros_like(dt_u8, dtype=np.uint16)
        conn8 = np.ones((3, 3), dtype=bool)
        labels = watershed(elev, markers=markers, mask=mask2, connectivity=conn8).astype(np.uint16)
        return labels

    def overlay_from_labels(self, orig_img: np.ndarray, labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        base_rgb = self._ensure_rgb_u8(orig_img)
        labels_rgb = self._labels_to_color(labels)
        if base_rgb is None or labels_rgb is None:
            return None
        if base_rgb.shape[:2] != labels_rgb.shape[:2]:
            return None
        base = base_rgb.astype(np.float32)
        over = labels_rgb.astype(np.float32)
        m = (labels > 0)
        out = base.copy()
        if m.any():
            out[m] = base[m] * (1.0 - alpha) + over[m] * alpha
        return out.clip(0, 255).astype(np.uint8)
    
    # --- Pore sampling via DT local maxima -----------------------------------

    def void_mask_and_pore_dt(self, dt_u8: np.ndarray, thr_u8: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            void_mask_u8: uint8, {0,1}, where 1 marks void (pores)
            pore_dt: float32, Euclidean distance (in pixels) inside voids
        """
        if dt_u8 is None:
            return None, None
        t = float(thr_u8) / 255.0
        dt01 = dt_u8.astype(np.float32) / 255.0
        fiber_mask = (dt01 >= t)             # same predicate as labels_from_dt_and_markers
        void_mask = (~fiber_mask).astype(np.uint8)
        if void_mask.any():
            # OpenCV expects 8-bit binary {0,255} or {0,1}; both work for DIST_L2
            pore_dt = cv2.distanceTransform(void_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
        else:
            pore_dt = np.zeros_like(dt01, dtype=np.float32)
        return void_mask, pore_dt


    def _local_maxima(self, arr: np.ndarray, footprint: int = 3, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Return boolean mask of local maxima in arr using grayscale dilation.
        - footprint: odd window size (3,5,7...).
        - mask: optional boolean/uint8 mask to restrict candidates (e.g., void area).
        Notes:
            • Pixels with value 0 are treated as background and ignored
            (set temporarily to 255 so they never appear as maxima).
        """
        k = max(3, int(footprint) | 1)  # ensure odd >= 3
        arr_work = arr.copy()

        # --- NEW: exclude black (0) pixels from maxima search ---------------------
        arr_work[arr_work == 0] = 255

        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dil = cv2.dilate(arr_work, ker)
        lm = (arr_work == dil) & (arr > 0)  # ensure real, nonzero peaks

        if mask is not None:
            lm &= (mask.astype(bool))
        return lm


    def sample_inscribed_circles(
        self,
        pore_dt: np.ndarray,
        void_mask: np.ndarray,
        min_radius_px: float = 1.0,
        min_peak_px: float = 3.0,          # <- NEW: DT peak threshold
        peak_footprint: int = 3,
        max_pores: int | None = None,
    ) -> list[dict]:
        if pore_dt is None or void_mask is None:
            return []

        H, W = pore_dt.shape[:2]
        current_void = (void_mask.astype(np.uint8) > 0).astype(np.uint8)

        lm = self._local_maxima(pore_dt, footprint=peak_footprint, mask=current_void)
        ys, xs = np.nonzero(lm)
        if xs.size == 0:
            return []

        radii = pore_dt[ys, xs]
        # --- NEW: apply peak-height (DT) threshold BEFORE sorting
        keep = radii >= float(min_peak_px)
        if not np.any(keep):
            return []
        xs, ys, radii = xs[keep], ys[keep], radii[keep]

        order = np.argsort(-radii)
        xs, ys, radii = xs[order], ys[order], radii[order]

        circles: list[dict] = []
        for x, y, r in zip(xs, ys, radii):
            if r < float(min_radius_px):
                continue
            r_int = int(np.floor(float(r)))
            if r_int <= 0:
                continue
            if x - r_int < 0 or x + r_int >= W or y - r_int < 0 or y + r_int >= H:
                continue

            cmask = np.zeros_like(current_void, dtype=np.uint8)
            cv2.circle(cmask, (int(x), int(y)), r_int, 1, thickness=-1)
            if np.all(cmask <= current_void):
                circles.append({'x': int(x), 'y': int(y), 'r_px': float(r)})
                current_void[cmask == 1] = 0
                if (max_pores is not None) and (len(circles) >= max_pores):
                    break
        return circles



    # ---------- analysis ----------

    def analyze(self, dt: np.ndarray, constant_segments: list[np.ndarray], spacing_px: float):
        """
        Returns:
          - hist_centers: list of (y, x, r)
          - viz_points: list of (y, x, r, ny, nx, ty, tx)
          - angles_deg: list[float]
          - gx, gy: gradients for debug view (float32)

        Note:
            `_xy` suffix denotes (x, y) ordering, while `_yx` denotes (y, x).
        """
        if dt is None or constant_segments is None:
            return [], [], [], None, None

        constant_segments_xy = constant_segments
        dt_s = gaussian_filter(dt, sigma=1.5)
        gy, gx = np.gradient(dt_s)
        gy = gaussian_filter(gy, sigma=1.0)
        gx = gaussian_filter(gx, sigma=1.0)

        hist_centers = []
        angles_deg = []
        viz_points = []

        def _canon_angle(ty: float, tx: float):
            n = float(np.hypot(ty, tx))
            if n == 0.0 or not np.isfinite(n):
                return None
            ty, tx = ty / n, tx / n
            ang = np.degrees(np.arctan2(ty, tx))
            if ang > 90.0:
                ang -= 180.0; ty, tx = -ty, -tx
            elif ang <= -90.0:
                ang += 180.0; ty, tx = -ty, -tx
            if ang <= -90.0:
                ang = -89.999
            return ty, tx, ang

        for seg_xy in constant_segments_xy:
            seg_xy = seg_xy.reshape(-1, 2).astype(np.float32)
            if len(seg_xy) < 2:
                continue
            samples_xy = resample_polyline_xy(seg_xy, step=max(1.0, float(spacing_px)))
            if len(samples_xy) < 2:
                continue
            # Convert samples from (x, y) to (y, x) for raster-indexed operations.
            samples_yx = samples_xy[:, [1, 0]].astype(np.float32)
            normals_yx = normals_from_dt_2d(
                centerline=samples_yx,
                dt=dt,
                gx=gx,
                gy=gy,
                alpha=0.5,
                iters=3,
                tau_g=1e-3,
            )
            for i, (y, x) in enumerate(samples_yx):
                r = float(_bilinear(dt, float(y), float(x)))
                if r <= 0:
                    continue
                ny, nx = float(normals_yx[i, 0]), float(normals_yx[i, 1])
                # Rotate normal (ny,nx) → tangent (ty,tx) so arrows stay 90° to diameter lines.
                ty, tx = nx, -ny
                canon = _canon_angle(ty, tx)
                if canon is None:
                    angles_deg.append(np.nan)
                else:
                    ty_c, tx_c, ang = canon
                    angles_deg.append(ang)
                    if (i % 10) == 0:
                        viz_points.append((int(round(y)), int(round(x)), r, ny, nx, ty_c, tx_c))
                hist_centers.append((int(round(y)), int(round(x)), r))

        return hist_centers, viz_points, angles_deg, gx, gy
