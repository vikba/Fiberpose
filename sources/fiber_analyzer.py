from __future__ import annotations
import os
import logging
import numpy as np
import cv2

from skimage.measure import label as cc_label
from skimage.segmentation import watershed

from .ridgedetector import RidgeDetector
from .utils import normals_from_dt_2d, resample_polyline_xy, _bilinear


# ---------- Paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

logger = logging.getLogger(__name__)


class FiberAnalyzer:
    """
    Non-GUI class encapsulating fiber detection, watershed, and analysis.

    Responsibilities:
      - centerline detection (compute_centerlines)
      - watershed labeling (labels_from_dt_and_markers)
      - analysis (analyze): samples, viz points, gradients, angles
    """

    def __init__(self, dt: np.ndarray, gx: np.ndarray, gy: np.ndarray, high_conf_fib: bool):
        self._rd = RidgeDetector()
        self._set_dt_and_gradients(dt, gx, gy)
        self._high_conf_fib = bool(high_conf_fib)

    def _set_dt_and_gradients(self, dt: np.ndarray, gx: np.ndarray, gy: np.ndarray):
        self._dt = np.asarray(dt, dtype=np.float32)
        self._dt_u8 = np.clip(self._dt, 0, 255).astype(np.uint8)
        self._gx = np.asarray(gx, dtype=np.float32)
        self._gy = np.asarray(gy, dtype=np.float32)
        if self._gx.shape != self._dt.shape or self._gy.shape != self._dt.shape:
            raise ValueError("Gradient arrays must match DT shape.")

    def set_high_conf_filter(self, high_conf_fib: bool):
        self._high_conf_fib = bool(high_conf_fib)


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

    @staticmethod
    def _split_and_trim_contour(
        contour: np.ndarray,
        junction_points: list[tuple[int, int]],
        trim: int = 3,
        tol: float = 3.0,
    ) -> list[np.ndarray]:
        """
        Split a contour at points close to junctions and trim endpoints.
        """
        pts_xy = contour.reshape(-1, 2)
        split_indices: list[int] = []
        for (jx, jy) in junction_points:
            dists = np.sqrt((pts_xy[:, 0] - jx) ** 2 + (pts_xy[:, 1] - jy) ** 2)
            idxs = np.where(dists < tol)[0]
            if idxs.size > 0:
                min_idx = idxs[np.argmin(dists[idxs])]
                split_indices.append(int(min_idx))
        split_indices = sorted(set(split_indices))

        segments: list[np.ndarray] = []
        boundaries = [0] + split_indices + [len(pts_xy) - 1]
        for i in range(len(boundaries) - 1):
            seg_xy = pts_xy[boundaries[i] : boundaries[i + 1] + 1]
            if len(seg_xy) > 2 * trim:
                seg_xy = seg_xy[trim:-trim]
            else:
                continue
            if len(seg_xy) >= 2:
                segments.append(seg_xy.reshape(-1, 1, 2))
        return segments

    @staticmethod
    def _sliding_window_filter(
        seg_xy: np.ndarray,
        image: np.ndarray,
        win_len: float = 15.0,
        std_thresh: float = 0.6,
        max_angle: float = 20.0,
    ) -> list[np.ndarray]:
        """
        Return sub-segments where intensity variation and slope stay within bounds.
        """
        pts_xy = seg_xy.reshape(-1, 2).astype(np.float32)
        if pts_xy.shape[0] < 2:
            return []

        # cumulative arc length along polyline
        cum_dist = [0.0]
        for i in range(1, len(pts_xy)):
            cum_dist.append(cum_dist[-1] + float(np.linalg.norm(pts_xy[i] - pts_xy[i - 1])))
        cum_dist = np.asarray(cum_dist, dtype=np.float32)
        total_len = float(cum_dist[-1])
        if total_len <= 0:
            return []

        # grayscale image
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        h, w = gray.shape[:2]

        def sample_intensity(p_xy: np.ndarray) -> float:
            x, y = int(round(p_xy[0])), int(round(p_xy[1]))
            x = min(max(x, 0), w - 1)
            y = min(max(y, 0), h - 1)
            return float(gray[y, x])

        valid_windows: list[tuple[int, int]] = []
        start_idx = 0
        half_step = float(win_len) / 2.0
        max_angle = float(max_angle)

        n_pts = len(pts_xy)
        while start_idx < n_pts - 1:
            start_s = float(cum_dist[start_idx])
            target_s = start_s + float(win_len)
            end_idx = int(np.searchsorted(cum_dist, target_s, side="left"))

            if end_idx <= start_idx + 1:
                end_idx = start_idx + 1
            if end_idx >= n_pts:
                end_idx = n_pts - 1

            window_pts_xy = pts_xy[start_idx : end_idx + 1]
            intensities = np.fromiter((sample_intensity(p_xy) for p_xy in window_pts_xy), dtype=np.float32)
            if intensities.size >= 2:
                if np.std(intensities) < std_thresh:
                    ds = float(cum_dist[end_idx] - cum_dist[start_idx])
                    if ds > 0.0:
                        dI = abs(float(intensities[-1]) - float(intensities[0]))
                        angle_deg = np.degrees(np.arctan(dI / ds))
                        if angle_deg < max_angle:
                            valid_windows.append((start_idx, end_idx))

            new_target_s = start_s + half_step
            new_start_idx = int(np.searchsorted(cum_dist, new_target_s, side="left"))
            if new_start_idx <= start_idx:
                break
            start_idx = new_start_idx

        merged: list[tuple[int, int]] = []
        for win in valid_windows:
            if not merged:
                merged.append(win)
            else:
                a0, a1 = merged[-1]
                b0, b1 = win
                if b0 <= a1:
                    merged[-1] = (a0, max(a1, b1))
                else:
                    merged.append(win)

        out_segments: list[np.ndarray] = []
        for s_idx, e_idx in merged:
            if float(cum_dist[e_idx] - cum_dist[s_idx]) >= float(win_len):
                sub_xy = pts_xy[s_idx : e_idx + 1].copy().reshape(-1, 1, 2)
                out_segments.append(sub_xy)
        return out_segments

    @staticmethod
    def _polyline_normals(
        pts_xy: np.ndarray,
        window: int = 8,
        angle_thresh_deg: float = 60.0,
        side: str = "left",
    ) -> np.ndarray:
        """
        Compute per-vertex unit normals for an open polyline.
        """
        pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
        n = len(pts_xy)
        if n < 2:
            raise ValueError("Need at least 2 points for an open polyline.")

        eps = 1e-12
        seg = pts_xy[1:] - pts_xy[:-1]
        seg_norm = np.linalg.norm(seg, axis=1, keepdims=True) + eps
        u = seg / seg_norm

        def rot90(v: np.ndarray) -> np.ndarray:
            return np.stack([-v[:, 1], v[:, 0]], axis=1)

        def unit(v: np.ndarray) -> np.ndarray:
            nrm = np.linalg.norm(v, axis=1, keepdims=True) + eps
            return v / nrm

        angle_thresh = np.deg2rad(angle_thresh_deg)
        normals = np.zeros((n, 2), dtype=np.float32)

        t0 = u[0:1]
        tn = u[-1:]
        normals[0] = rot90(t0)[0]
        normals[-1] = rot90(tn)[0]

        for i in range(1, n - 1):
            d_prev = u[i - 1 : i]
            d_next = u[i : i + 1]
            cosang = np.clip(np.sum(d_prev * d_next, axis=1), -1.0, 1.0)[0]
            ang = np.arccos(cosang)

            if ang >= angle_thresh:
                t = unit(d_prev + d_next)
                if np.linalg.norm(t) < 1e-6:
                    t = d_next
            else:
                seg_lo = max(0, i - window)
                seg_hi = min(n - 2, i + window - 1)
                if seg_lo > seg_hi:
                    seg_lo, seg_hi = i - 1, i
                inds = np.arange(seg_lo, seg_hi + 1)
                center = i - 0.5
                dist = np.abs(inds + 0.5 - center)
                weights = (window + 1) - dist
                weights = np.clip(weights, 1.0, None)
                weights = (weights / (weights.sum() + eps)).reshape(-1, 1)
                t = unit((u[inds] * weights).sum(axis=0, keepdims=True))

            normals[i] = rot90(t)[0]

        if side == "right":
            normals = -normals
        return normals.astype(np.float32)



    # ---------- centerlines / markers / watershed ----------

    def compute_centerlines(
        self,
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
        contours, junction_points, _ = self._rd.detect_lines_contours(
            image=self._dt_u8,
            sigma=sigma,
            upper_thresh=upperThresh,
            lower_thresh=lowerThresh,
            min_length=minLength,
            max_length=maxLength,
            is_dark_line=isDarkLine,
            do_correct_position=doCorrectPosition,
            do_estimate_width=doEstimateWidth,
            do_extend_line=doExtendLine,
            overlap_option=overlapOption,
        )

        processed_segments: list[np.ndarray] = []
        for contour in contours:
            segs_xy = self._split_and_trim_contour(contour, junction_points, trim=trim, tol=tol)
            if segs_xy:
                processed_segments.extend(segs_xy)
            else:
                pts_xy = contour.reshape(-1, 2)
                if len(pts_xy) > 2 * trim:
                    trimmed_xy = pts_xy[trim:-trim].reshape(-1, 1, 2)
                    processed_segments.append(trimmed_xy)

        constant_segments_xy: list[np.ndarray] = []
        for seg_xy in processed_segments:
            constant_segments_xy.extend(
                self._sliding_window_filter(
                    seg_xy,
                    image=self._dt_u8,
                    win_len=win_len,
                    std_thresh=std_thresh,
                    max_angle=max_angle_deg,
                )
            )

        if self._high_conf_fib and constant_segments_xy:
            logger.info(
                "Applying high confidence fiber filter, %d segments before filtering.",
                len(constant_segments_xy),
            )
            min_len = max(2, int(win_len / 2))
            constant_segments_xy = [seg for seg in constant_segments_xy if len(seg) >= min_len]

        normals_xy = [self._polyline_normals(seg_xy) for seg_xy in constant_segments_xy]

        base_dt = self._to_uint8_for_view(self._dt_u8)
        centerlines_rgb = cv2.cvtColor(base_dt, cv2.COLOR_GRAY2BGR)
        # RidgeDetector delivers polylines in (x, y) order; the `_xy` suffix keeps that explicit.
        for seg_xy in constant_segments_xy:
            cv2.polylines(centerlines_rgb, [seg_xy.astype(np.int32)], isClosed=False,
                          color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        #calculation of markers that limit watershed
        centerlines_binary = np.zeros(self._dt_u8.shape, dtype=np.uint8)
        for seg_xy in constant_segments_xy:
            cv2.polylines(centerlines_binary, [seg_xy.astype(np.int32)], isClosed=False,
                          color=255, thickness=2, lineType=cv2.LINE_8)
        _, centerlines_binary = cv2.threshold(centerlines_binary, 128, 255, cv2.THRESH_BINARY)
        markers = cc_label(centerlines_binary, connectivity=2).astype(np.int32)
        
        return constant_segments_xy, normals_xy, centerlines_rgb, markers

    def labels_from_dt_and_markers(self, markers: np.ndarray, thr_u8: int) -> np.ndarray:
        """
        Watershed on inverted DT with binary mask from threshold and given markers.
        """
        if markers is None or markers.max() == 0:
            return np.zeros_like(self._dt_u8, dtype=np.uint16)
        t = float(thr_u8) / 255.0
        dt_norm = self._dt_u8.astype(np.float32) / 255.0
        mask2 = dt_norm >= t
        elev = 1.0 - dt_norm
        if not mask2.any():
            return np.zeros_like(self._dt_u8, dtype=np.uint16)
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

    def void_mask_and_pore_dt(self, thr_u8: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            void_mask_u8: uint8, {0,1}, where 1 marks void (pores)
            pore_dt: float32, Euclidean distance (in pixels) inside voids
        """
        if self._dt_u8 is None:
            return None, None
        t = float(thr_u8) / 255.0
        dt01 = self._dt_u8.astype(np.float32) / 255.0
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

    def analyze(self, constant_segments: list[np.ndarray], spacing_px: float):
        """
        Returns:
          - hist_centers: list of (y, x, r)
          - viz_points: list of (y, x, r, ny, nx, ty, tx)
          - angles_deg: list[float]

        Note:
            `_xy` suffix denotes (x, y) ordering, while `_yx` denotes (y, x).
        """
        if self._dt is None or constant_segments is None:
            return [], [], []

        constant_segments_xy = constant_segments
        dt = self._dt
        gx = self._gx
        gy = self._gy

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

        return hist_centers, viz_points, angles_deg
