"""
Pixel-level ridge (centerline) detector for distance-transform-like images,
with the same public interface as the Steger RidgeDetector.

For every pixel it asks: "is this the highest pixel of its own cross-section?"
  1. optional light separable Gaussian smoothing (sigma, 0 = off)
  2. per pixel, 3x3 stencil only:
       - finite-difference Hessian -> cross-section direction n (eigenvector
         of the most negative eigenvalue) and ridge strength s = -lambda_min
       - non-maximum suppression along n (bilinear samples at p +- n)
  3. hysteresis (low/high on s) through the original Steger linker, which
     orders the pixels into polylines.

No sub-pixel step: every returned point is an integer pixel centre.

Usage (drop-in for the Steger detector):
    lines = RidgeDetector.detect_ridges(dt, sigma=1.0, low=0.1, high=0.25,
                                        polarity="light", min_value=1.0)
    # -> list of (N, 2) float32 arrays, columns [x=col, y=row]

    # pixels only, no ordering (cheaper):
    xy, strength, normal = RidgeDetector.detect_ridge_pixels(dt, ...)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Stage 0: separable Gaussian smoothing (mirror borders, like the original)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _mirror(i: int, n: int) -> int:
    if i < 0:
        return -i
    if i >= n:
        return 2 * n - 2 - i
    return i


@njit(cache=True, parallel=True)
def _smooth_jit(img, k, n):
    h, w = img.shape
    tmp = np.empty((h, w), np.float32)
    out = np.empty((h, w), np.float32)

    # pass 1: along columns index (inside each row, contiguous)
    for ri in prange(h):
        r = np.int64(ri)        # prange index may be uint64; keep index math signed
        for c in range(w):
            s = 0.0
            if c >= n and c < w - n:
                for j in range(-n, n + 1):
                    s += img[r, c + j] * k[j + n]
            else:
                for j in range(-n, n + 1):
                    s += img[r, _mirror(c + j, w)] * k[j + n]
            tmp[r, c] = s

    # pass 2: along rows index; innermost loop runs over contiguous columns
    for ri in prange(h):
        r = np.int64(ri)
        acc = np.zeros(w, np.float64)
        for j in range(-n, n + 1):
            rr = _mirror(r + j, h)
            kj = k[j + n]
            for c in range(w):
                acc[c] += tmp[rr, c] * kj
        for c in range(w):
            out[r, c] = acc[c]
    return out


def _gauss_kernel(sigma: float) -> Tuple[np.ndarray, int]:
    """Pixel-integrated Gaussian, normalised to sum 1."""
    n = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-n, n + 1, dtype=np.float64)
    s2 = sigma * math.sqrt(2.0)
    k = np.array([0.5 * (math.erf((xi + 0.5) / s2) - math.erf((xi - 0.5) / s2)) for xi in x])
    k /= k.sum()
    return k, n


# ---------------------------------------------------------------------------
# Stage 1: 3x3 Hessian + NMS along the cross-section direction
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sample_at(g, r, c, nr, nc, rm, rp, cm, cp):
    """Bilinear value at p + n (|n| = 1), i.e. one pixel along the normal.
    (Sampling where the ray leaves the 3x3 square instead would skip the
    centre diagonal of a 45-degree ridge and leave 3-px-thick lines.)"""
    if nr >= 0.0:
        r0 = r; r1 = rp; ar = nr
    else:
        r0 = rm; r1 = r; ar = 1.0 + nr
    if nc >= 0.0:
        c0 = c; c1 = cp; ac = nc
    else:
        c0 = cm; c1 = c; ac = 1.0 + nc
    r0 = np.int64(r0); r1 = np.int64(r1); c0 = np.int64(c0); c1 = np.int64(c1)
    return ((1.0 - ar) * ((1.0 - ac) * g[r0, c0] + ac * g[r0, c1]) +
            ar * ((1.0 - ac) * g[r1, c0] + ac * g[r1, c1]))


@njit(cache=True, parallel=True)
def _ridge_nms_jit(g, low, high, min_value, max_off2):
    """
    Returns
      ismax : uint8 (h, w)   0 = no ridge, 1 = weak (low <= s < high), 2 = strong
      ev    : float32 (h, w) ridge strength s = -lambda_min (only on ridge pixels)
      nx,ny : float32 (h, w) unit cross-section direction (row, col components)
    """
    h, w = g.shape
    ismax = np.zeros((h, w), np.uint8)
    ev = np.zeros((h, w), np.float32)
    nx = np.zeros((h, w), np.float32)
    ny = np.zeros((h, w), np.float32)

    for ri in prange(h):
        # numba types the prange index as uint64; uint64 mixed with int64 gives
        # float64, which then fails as an array index -> cast once here
        r = np.int64(ri)
        rm = r - 1 if r > 0 else 1
        rp = r + 1 if r < h - 1 else h - 2
        for c in range(w):
            f = g[r, c]
            if f <= min_value:               # background: the main saving
                continue
            cm = c - 1 if c > 0 else 1
            cp = c + 1 if c < w - 1 else w - 2

            # cheap pre-reject: a cross-section max is >= the opposite neighbour
            # pair in at least one of the 4 lattice directions
            if not ((f >= g[rm, c] and f >= g[rp, c]) or
                    (f >= g[r, cm] and f >= g[r, cp]) or
                    (f >= g[rm, cm] and f >= g[rp, cp]) or
                    (f >= g[rm, cp] and f >= g[rp, cm])):
                continue

            hrr = g[rm, c] + g[rp, c] - 2.0 * f
            hcc = g[r, cm] + g[r, cp] - 2.0 * f
            hrc = 0.25 * (g[rp, cp] + g[rm, cm] - g[rp, cm] - g[rm, cp])

            half_tr = 0.5 * (hrr + hcc)
            half_df = 0.5 * (hrr - hcc)
            lam = half_tr - math.sqrt(half_df * half_df + hrc * hrc)   # most negative
            s = -lam
            if s < low:
                continue

            # flank rejection (scale-free): near a true maximum the gradient is
            # small relative to the curvature, |grad| ~ s * (distance to peak).
            # On a flank the gradient ~ slope while noise curvature is small.
            gr = 0.5 * (g[rp, c] - g[rm, c])
            gc = 0.5 * (g[r, cp] - g[r, cm])
            if gr * gr + gc * gc > max_off2 * s * s:
                continue

            # eigenvector of lam; take the better-conditioned of two forms
            ar, ac = hrc, lam - hrr
            br, bc = lam - hcc, hrc
            na = ar * ar + ac * ac
            nb = br * br + bc * bc
            if na >= nb:
                vr, vc, nn = ar, ac, na
            else:
                vr, vc, nn = br, bc, nb
            if nn < 1e-24:                    # isotropic peak: any direction
                vr, vc = 1.0, 0.0
            else:
                inv = 1.0 / math.sqrt(nn)
                vr *= inv
                vc *= inv
            # canonical sign (dominant component positive) so that neighbouring
            # pixels agree -> the asymmetric test below keeps exactly one of two
            # equal centre pixels (even widths)
            if (vr < 0.0) if abs(vr) >= abs(vc) else (vc < 0.0):
                vr = -vr
                vc = -vc

            f_plus = _sample_at(g, r, c, vr, vc, rm, rp, cm, cp)
            f_minus = _sample_at(g, r, c, -vr, -vc, rm, rp, cm, cp)
            if f > f_plus and f >= f_minus:
                ismax[r, c] = 2 if s >= high else 1
                ev[r, c] = s
                nx[r, c] = vr
                ny[r, c] = vc

    return ismax, ev, nx, ny


# ---------------------------------------------------------------------------
# Hysteresis without linking (for detect_ridge_pixels)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _hysteresis_jit(ismax):
    h, w = ismax.shape
    keep = np.zeros((h, w), np.bool_)
    n_cand = 0
    for r in range(h):
        for c in range(w):
            if ismax[r, c] > 0:
                n_cand += 1
    stack = np.empty((n_cand, 2), np.int32)
    for r0 in range(h):
        for c0 in range(w):
            if ismax[r0, c0] != 2 or keep[r0, c0]:
                continue
            keep[r0, c0] = True
            top = 0
            stack[top, 0] = r0
            stack[top, 1] = c0
            top += 1
            while top > 0:
                top -= 1
                r = stack[top, 0]
                c = stack[top, 1]
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        rr = r + dr
                        cc = c + dc
                        if rr < 0 or rr >= h or cc < 0 or cc >= w:
                            continue
                        if ismax[rr, cc] > 0 and not keep[rr, cc]:
                            keep[rr, cc] = True
                            stack[top, 0] = rr
                            stack[top, 1] = cc
                            top += 1
    return keep


# ---------------------------------------------------------------------------
# Linking: the original Steger linker, with three fixes
#   - output buffer sized from ALL ridge pixels (weak ones were not counted)
#   - boundscheck off
#   - int32 seed coordinates (int16 overflowed above 32767 px)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _trace_one_direction(x0, y0, dir_flag, ismax, px, py, alpha_arr, oct4_arr,
                         label, indx, cross_done, w, h, DIRTAB, CLEARTAB,
                         max_angle_diff, line_id, out_x, out_y, out_start):
    x = x0
    y = y0
    p = x * w + y
    last_oct = int(oct4_arr[p])
    if dir_flag == 2:
        last_oct += 4
    out_i = out_start

    while True:
        p = x * w + y
        pxx = px[p]
        pyy = py[p]
        alpha_p = alpha_arr[p]
        octant = int(oct4_arr[p])
        if octant == 0 and 3 <= last_oct <= 5:
            octant = 4
        elif octant == 1 and 4 <= last_oct <= 6:
            octant = 5
        elif octant == 2 and 4 <= last_oct <= 7:
            octant = 6
        elif octant == 3 and (last_oct == 0 or last_oct >= 6):
            octant = 7
        last_oct = octant

        nextismax = 0
        nexti = 1
        mindiff = 1e30
        for i in range(3):
            xx = x + int(DIRTAB[octant, i, 0])
            yy = y + int(DIRTAB[octant, i, 1])
            if xx < 0 or xx >= h or yy < 0 or yy >= w:
                continue
            pp = xx * w + yy
            if ismax[pp] == 0:
                continue
            dx = px[pp] - pxx
            dy = py[pp] - pyy
            dist = math.sqrt(dx * dx + dy * dy)
            diff = abs(alpha_p - alpha_arr[pp])
            if diff >= 0.5 * math.pi:
                diff = math.pi - diff
            score = dist + diff
            if score < mindiff:
                mindiff = score
                nexti = i
            nextismax = 1

        for i in range(2):
            xx = x + int(CLEARTAB[octant, i, 0])
            yy = y + int(CLEARTAB[octant, i, 1])
            if 0 <= xx < h and 0 <= yy < w:
                pp = xx * w + yy
                if ismax[pp] > 0:
                    diff = abs(alpha_p - alpha_arr[pp])
                    if diff >= 0.5 * math.pi:
                        diff = math.pi - diff
                    if diff < max_angle_diff:
                        label[pp] = line_id
                        idx = indx[pp]
                        if idx != 0:
                            cross_done[idx - 1] = True

        if nextismax == 0:
            break

        x += int(DIRTAB[octant, nexti, 0])
        y += int(DIRTAB[octant, nexti, 1])
        pp = x * w + y
        if out_i >= out_x.shape[0]:
            return out_i
        out_x[out_i] = px[pp]
        out_y[out_i] = py[pp]
        out_i += 1
        if label[pp] > 0:
            break
        label[pp] = line_id
        idx = indx[pp]
        if idx != 0:
            cross_done[idx - 1] = True

    return out_i


@njit(cache=True)
def _link_jit(cross_x, cross_y, cross_val, cross_done, ismax, px, py,
              alpha_arr, oct4_arr, indx, w, h, DIRTAB, CLEARTAB,
              max_angle_diff, max_points):
    area = cross_x.shape[0]
    label = np.zeros((w * h,), np.int32)
    pts_x = np.empty((max_points,), np.float32)
    pts_y = np.empty((max_points,), np.float32)
    line_offsets = np.empty((area + 1,), np.int32)
    line_count = 0
    out_i = 0
    indx_max = 0

    while True:
        while indx_max < area and cross_done[indx_max]:
            indx_max += 1
        if indx_max == area:
            break
        if cross_val[indx_max] == 0.0:
            break

        x0 = int(cross_x[indx_max])
        y0 = int(cross_y[indx_max])
        pos0 = x0 * w + y0
        line_id = line_count + 1
        label[pos0] = line_id
        idx0 = indx[pos0]
        if idx0 != 0:
            cross_done[idx0 - 1] = True
        if out_i >= max_points:
            break

        line_offsets[line_count] = out_i
        pts_x[out_i] = px[pos0]
        pts_y[out_i] = py[pos0]
        alpha0 = alpha_arr[pos0]
        out_i += 1

        octant0 = int(oct4_arr[pos0])
        for i in range(2):
            xx = x0 + int(CLEARTAB[octant0, i, 0])
            yy = y0 + int(CLEARTAB[octant0, i, 1])
            if 0 <= xx < h and 0 <= yy < w:
                p = xx * w + yy
                if ismax[p] > 0:
                    diff = abs(alpha0 - alpha_arr[p])
                    if diff >= 0.5 * math.pi:
                        diff = math.pi - diff
                    if diff < max_angle_diff:
                        label[p] = line_id
                        idxp = indx[p]
                        if idxp != 0:
                            cross_done[idxp - 1] = True

        tmp_start = out_i
        tmp_end = _trace_one_direction(x0, y0, 2, ismax, px, py, alpha_arr, oct4_arr,
                                       label, indx, cross_done, w, h, DIRTAB, CLEARTAB,
                                       max_angle_diff, line_id, pts_x, pts_y, tmp_start)
        # reverse [seed, b1..bk] -> [bk..b1, seed]; the original reversed only
        # b1..bk, which left the seed out of order at index 0
        rev_start = line_offsets[line_count]
        tmp_len = tmp_end - rev_start
        for k in range(tmp_len // 2):
            a = rev_start + k
            b = rev_start + (tmp_len - 1 - k)
            tx = pts_x[a]; pts_x[a] = pts_x[b]; pts_x[b] = tx
            ty = pts_y[a]; pts_y[a] = pts_y[b]; pts_y[b] = ty
        out_i = tmp_end

        out_i = _trace_one_direction(x0, y0, 1, ismax, px, py, alpha_arr, oct4_arr,
                                     label, indx, cross_done, w, h, DIRTAB, CLEARTAB,
                                     max_angle_diff, line_id, pts_x, pts_y, out_i)

        if out_i - line_offsets[line_count] > 1:
            line_count += 1
        else:
            out_i = line_offsets[line_count]
            label[pos0] = 0

    line_offsets[line_count] = out_i
    return pts_x[:out_i], pts_y[:out_i], line_offsets[:line_count + 1]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DTRidgeDetector:
    """Pixel-level ridge detector for distance transforms (Steger-compatible API)."""

    MODE_LIGHT, MODE_DARK = 1, 2
    MAX_ANGLE_DIFFERENCE = math.pi / 6.0

    _DIRTAB = np.array([
        [[ 1, 0], [ 1, -1], [ 1, 1]],
        [[ 1, 1], [ 1, 0], [ 0, 1]],
        [[ 0, 1], [ 1, 1], [-1, 1]],
        [[-1, 1], [ 0, 1], [-1, 0]],
        [[-1, 0], [-1, 1], [-1, -1]],
        [[-1, -1], [-1, 0], [ 0, -1]],
        [[ 0, -1], [-1, -1], [ 1, -1]],
        [[ 1, -1], [ 0, -1], [ 1, 0]],
    ], dtype=np.int8)

    _CLEARTAB = np.array([
        [[ 0, 1], [ 0, -1]],
        [[-1, 1], [ 1, -1]],
        [[-1, 0], [ 1, 0]],
        [[-1, -1], [ 1, 1]],
        [[ 0, -1], [ 0, 1]],
        [[ 1, -1], [-1, 1]],
        [[ 1, 0], [-1, 0]],
        [[ 1, 1], [-1, -1]],
    ], dtype=np.int8)

    # -------------------------
    # Public API
    # -------------------------

    @classmethod
    def detect_ridges(
        cls,
        image2d: np.ndarray,
        *,
        sigma: float = 1.0,
        low: float = 0.1,
        high: float = 0.25,
        polarity: str = "light",
        min_value: Optional[float] = None,
        max_offset: float = 1.0,
    ) -> List[np.ndarray]:
        """
        Same signature/output as the Steger RidgeDetector.detect_ridges:
        list of (N, 2) float32 arrays [x=col, y=row], one per linked ridge.
        Points are integer pixel centres (no sub-pixel offset).

        sigma     : pre-smoothing Gaussian sigma in px (0 = none)
        low, high : hysteresis thresholds on ridge strength -lambda_min of the
                    3x3 Hessian of the (smoothed) image, in image units / px^2
        polarity  : "light" (DT ridges) or "dark"
        min_value : pixels with value <= min_value are skipped outright
                    (in ridge polarity, i.e. after negation for "dark").
                    None -> 0.0 for "light", no limit for "dark".
        max_offset: reject pixels whose |gradient| > max_offset * strength,
                    i.e. whose Newton step to the peak would exceed max_offset px.
                    Removes noise maxima on fiber flanks. <= 0 or inf disables.
        """
        ismax, ev, nx, ny = cls._candidates(image2d, sigma, low, high, polarity, min_value, max_offset)
        h, w = ismax.shape
        return cls._link(ismax, ev, nx, ny, w, h)

    @classmethod
    def detect_ridge_pixels(
        cls,
        image2d: np.ndarray,
        *,
        sigma: float = 1.0,
        low: float = 0.1,
        high: float = 0.25,
        polarity: str = "light",
        min_value: Optional[float] = None,
        max_offset: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ridge pixels without ordering into lines (cheaper than detect_ridges).
        Weak pixels are kept only if 8-connected to a strong one.

        Returns
          xy       : (N, 2) int32 [x=col, y=row]
          strength : (N,)   float32, -lambda_min
          normal   : (N, 2) float32 unit cross-section direction [x, y]
        """
        ismax, ev, nx, ny = cls._candidates(image2d, sigma, low, high, polarity, min_value, max_offset)
        keep = _hysteresis_jit(ismax)
        rr, cc = np.nonzero(keep)
        xy = np.stack([cc, rr], axis=1).astype(np.int32)
        normal = np.stack([ny[rr, cc], nx[rr, cc]], axis=1)
        return xy, ev[rr, cc], normal

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _candidates(image2d, sigma, low, high, polarity, min_value, max_offset):
        img = np.asarray(image2d, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError("image must be 2D grayscale")
        h, w = img.shape
        if h < 3 or w < 3:
            raise ValueError("image must be at least 3x3")
        dark = polarity.lower() == "dark"
        if dark:
            img = -img
        img = np.ascontiguousarray(img)
        if sigma and sigma > 0.0:
            k, n = _gauss_kernel(float(sigma))
            if 2 * n + 1 > min(h, w):
                raise ValueError("sigma too large for image size")
            img = _smooth_jit(img, k, n)
        if min_value is None:
            min_value = -np.inf if dark else 0.0
        if max_offset is None or max_offset <= 0.0 or not np.isfinite(max_offset):
            max_off2 = np.inf
        else:
            max_off2 = float(max_offset) ** 2
        return _ridge_nms_jit(img, float(low), float(high), float(min_value), max_off2)

    @staticmethod
    def _precompute_angles(nx, ny):
        alpha = np.arctan2(nx, -ny).astype(np.float32)
        alpha = np.where(alpha < 0.0, alpha + 2.0 * np.pi, alpha)
        alpha = np.where(alpha >= np.pi, alpha - np.pi, alpha).astype(np.float32)
        oct4 = (np.floor((4.0 / np.pi) * alpha + 0.5).astype(np.int32) & 3).astype(np.uint8)
        return alpha, oct4

    @classmethod
    def _link(cls, ismax2d, ev2d, nx2d, ny2d, w, h) -> List[np.ndarray]:
        ismax = ismax2d.reshape(-1)
        strong = np.flatnonzero(ismax == 2)
        area = strong.size
        if area == 0:
            return []

        cross_val = ev2d.reshape(-1)[strong].astype(np.float32)
        order = np.argsort(-cross_val, kind="stable")
        strong = strong[order]
        cross_val = cross_val[order]
        cross_x = (strong // w).astype(np.int32)
        cross_y = (strong % w).astype(np.int32)
        cross_done = np.zeros(area, np.bool_)

        indx = np.zeros(w * h, np.int32)
        indx[strong] = np.arange(1, area + 1, dtype=np.int32)

        # integer pixel positions: px = row, py = col
        rows, cols = np.divmod(np.arange(w * h, dtype=np.int64), w)
        px = rows.astype(np.float32)
        py = cols.astype(np.float32)

        alpha, oct4 = cls._precompute_angles(nx2d.reshape(-1), ny2d.reshape(-1))

        # each ridge pixel is written once, plus at most one terminal duplicate
        # per traced direction per line (<= 2 per strong seed)
        max_points = int(np.count_nonzero(ismax)) + 2 * area + 8

        pts_x, pts_y, offsets = _link_jit(
            cross_x, cross_y, cross_val, cross_done, ismax, px, py,
            alpha, oct4, indx, w, h, cls._DIRTAB, cls._CLEARTAB,
            float(cls.MAX_ANGLE_DIFFERENCE), max_points,
        )

        lines = []
        for i in range(len(offsets) - 1):
            a = int(offsets[i]); b = int(offsets[i + 1])
            if b - a <= 1:
                continue
            pts = np.empty((b - a, 2), np.float32)
            pts[:, 0] = pts_y[a:b]       # x = col
            pts[:, 1] = pts_x[a:b]       # y = row
            lines.append(pts)
        return lines


# drop-in name
RidgeDetector = DTRidgeDetector
