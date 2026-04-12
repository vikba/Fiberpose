
"""
Steger ridge/line detection (exact math) with Numba speedups.

- RidgeDetector class with public detect_ridges()
- Returns (lines, junctions)
- Heavy pixel loops are JIT-compiled with numba (if available), no fastmath to preserve quality
- Linking stays in Python (topology-sensitive, list-heavy); still benefits from faster inputs

Usage:
    det = RidgeDetector()
    lines, junctions = det.detect_ridges(img, sigma=2.0, low=8, high=15, polarity="dark")

CLI demo:
    python ridge_detector_numba.py in.png out.png --sigma 2 --low 8 --high 15 --polarity dark
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Tuple
import math
import numpy as np

# Optional numba
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        """Return a pass-through decorator when Numba is unavailable.

        Args:
            *args (Any): Positional arguments accepted for API compatibility.
            **kwargs (Any): Keyword arguments accepted for API compatibility.

        Returns:
            Callable[[Callable[..., Any]], Callable[..., Any]]: Decorator that
                returns the wrapped function unchanged.
        """

        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            """Return `fn` unchanged to mirror `numba.njit` semantics."""
            return fn

        return wrap


# -------------------------
# Data model
# -------------------------

@dataclass(slots=True)
class RidgeLine:
    """Represent ridge line state used across processing and visualization workflows."""
    pts: np.ndarray            # (N,2) float32 [row, col]
    angle: np.ndarray          # (N,)  float32 normal angle beta (radians)
    response: np.ndarray       # (N,)  float32 interpolated ridge response
    width: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    asymmetry: Optional[np.ndarray] = None

    def __post_init__(self):
        """Normalize and validate dataclass fields after instance construction."""
        self.pts = np.asarray(self.pts, dtype=np.float32)
        self.angle = np.asarray(self.angle, dtype=np.float32)
        self.response = np.asarray(self.response, dtype=np.float32)


@dataclass(slots=True)
class Junction:
    """Represent junction state used across processing and visualization workflows."""
    pt: np.ndarray   # (2,) float32 [row, col]
    a: int           # index of line A
    b: int           # index of line B

    def __post_init__(self):
        """Normalize and validate dataclass fields after instance construction."""
        self.pt = np.asarray(self.pt, dtype=np.float32)


# -------------------------
# Numba kernels
# -------------------------

@njit(cache=True)
def _mirror_r_jit(r: int, h: int) -> int:
    """Provide an internal helper for mirror r jit.
    
    Args:
        r (int): Value for r interpreted by this routine.
        h (int): Value for h interpreted by this routine.
    
    Returns:
        int: Result object produced by the mirror r jit routine."""
    if r < 0:
        return -r
    if r >= h:
        return h - r + h - 2
    return r

@njit(cache=True)
def _mirror_c_jit(c: int, w: int) -> int:
    """Provide an internal helper for mirror c jit.
    
    Args:
        c (int): Value for c interpreted by this routine.
        w (int): Value for w interpreted by this routine.
    
    Returns:
        int: Result object produced by the mirror c jit routine."""
    if c < 0:
        return -c
    if c >= w:
        return w - c + w - 2
    return c

@njit(cache=True)
def _convolve_rows_flat(image: np.ndarray, mask: np.ndarray, n: int, w: int, h: int) -> np.ndarray:
    """Provide an internal helper for convolve rows flat.
    
    Args:
        image (np.ndarray): Array-like data carrying image values for analysis.
        mask (np.ndarray): Array-like data carrying mask values for analysis.
        n (int): Value for n interpreted by this routine.
        w (int): Value for w interpreted by this routine.
        h (int): Value for h interpreted by this routine.
    
    Returns:
        np.ndarray: Array containing the convolve rows flat output aligned with downstream expectations."""
    out = np.empty((w * h,), dtype=np.float32)

    # inner
    for r in range(n, h - n):
        base = r * w
        for c in range(w):
            l = base + c
            s = 0.0
            for j in range(-n, n + 1):
                s += float(image[l + j * w]) * float(mask[j + n])
            out[l] = np.float32(s)

    # borders top
    for r in range(0, n):
        base = r * w
        for c in range(w):
            l = base + c
            s = 0.0
            for j in range(-n, n + 1):
                rr = _mirror_r_jit(r + j, h)
                s += float(image[rr * w + c]) * float(mask[j + n])
            out[l] = np.float32(s)

    # borders bottom
    for r in range(h - n, h):
        base = r * w
        for c in range(w):
            l = base + c
            s = 0.0
            for j in range(-n, n + 1):
                rr = _mirror_r_jit(r + j, h)
                s += float(image[rr * w + c]) * float(mask[j + n])
            out[l] = np.float32(s)

    return out

@njit(cache=True)
def _convolve_cols_flat(tmp: np.ndarray, mask: np.ndarray, n: int, w: int, h: int) -> np.ndarray:
    """Provide an internal helper for convolve cols flat.
    
    Args:
        tmp (np.ndarray): Value for tmp interpreted by this routine.
        mask (np.ndarray): Array-like data carrying mask values for analysis.
        n (int): Value for n interpreted by this routine.
        w (int): Value for w interpreted by this routine.
        h (int): Value for h interpreted by this routine.
    
    Returns:
        np.ndarray: Array containing the convolve cols flat output aligned with downstream expectations."""
    out = np.empty((w * h,), dtype=np.float32)

    # inner
    for r in range(h):
        base = r * w
        for c in range(n, w - n):
            l = base + c
            s = 0.0
            for j in range(-n, n + 1):
                s += float(tmp[l + j]) * float(mask[j + n])
            out[l] = np.float32(s)

    # borders
    for r in range(h):
        base = r * w
        for c in range(0, n):
            l = base + c
            s = 0.0
            for j in range(-n, n + 1):
                cc = _mirror_c_jit(c + j, w)
                s += float(tmp[base + cc]) * float(mask[j + n])
            out[l] = np.float32(s)

        for c in range(w - n, w):
            l = base + c
            s = 0.0
            for j in range(-n, n + 1):
                cc = _mirror_c_jit(c + j, w)
                s += float(tmp[base + cc]) * float(mask[j + n])
            out[l] = np.float32(s)

    return out

@njit(cache=True)
def _solve_linear_jit(a: float, b: float) -> Tuple[int, float]:
    """Provide an internal helper for solve linear jit.
    
    Args:
        a (float): Value for a interpreted by this routine.
        b (float): Value for b interpreted by this routine.
    
    Returns:
        Tuple[int, float]: Tuple grouping the solve linear jit outputs in a stable positional order."""
    if a == 0.0:
        return 0, 0.0
    return 1, -b / a

@njit(cache=True)
def _eigen_2x2_jit(drr: float, drc: float, dcc: float) -> Tuple[float, float, float, float]:
    """
    Returns (e1, e2, v0x, v0y) where (v0x, v0y) is eigenvector for the principal eigenvalue
    following the same selection logic as the Java port.
    """
    if drc != 0.0:
        theta = 0.5 * (dcc - drr) / drc
        t = 1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0))
        if theta < 0.0:
            t = -t
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c
        e1 = drr - t * drc
        e2 = dcc + t * drc
    else:
        c = 1.0
        s = 0.0
        e1 = drr
        e2 = dcc

    n1 = c
    n2 = -s

    # choose ordering like original (by abs)
    if abs(e1) > abs(e2):
        return e1, e2, n1, n2
    elif abs(e1) < abs(e2):
        # swapped
        return e2, e1, -n2, n1
    else:
        if e1 < e2:
            return e1, e2, n1, n2
        else:
            return e2, e1, -n2, n1

@njit(cache=True)
def _ridge_candidates_jit(dr, dc, drr, drc, dcc, w: int, h: int,
                         low: float, high: float, mode: int,
                         pixel_boundary: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Provide an internal helper for ridge candidates jit.
    
    Args:
        dr (Any): Value for dr interpreted by this routine.
        dc (Any): Value for dc interpreted by this routine.
        drr (Any): Value for drr interpreted by this routine.
        drc (Any): Value for drc interpreted by this routine.
        dcc (Any): Value for dcc interpreted by this routine.
        w (int): Value for w interpreted by this routine.
        h (int): Value for h interpreted by this routine.
        low (float): Value for low interpreted by this routine.
        high (float): Value for high interpreted by this routine.
        mode (int): Value for mode interpreted by this routine.
        pixel_boundary (float): Value for pixel boundary interpreted by this routine.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Array containing the ridge candidates jit output aligned with downstream expectations."""
    N = w * h
    ismax = np.zeros((N,), dtype=np.uint8)
    ev = np.zeros((N,), dtype=np.float32)
    nx = np.zeros((N,), dtype=np.float32)
    ny = np.zeros((N,), dtype=np.float32)
    px = np.zeros((N,), dtype=np.float32)
    py = np.zeros((N,), dtype=np.float32)

    for r in range(h):
        for c in range(w):
            l = r * w + c

            e1, _, v0x, v0y = _eigen_2x2_jit(float(drr[l]), float(drc[l]), float(dcc[l]))

            val = -e1 if mode == 1 else e1  # 1=light, 2=dark
            if val <= 0.0:
                continue

            ev[l] = np.float32(val)

            n1 = v0x
            n2 = v0y

            a = float(drr[l]) * n1 * n1 + 2.0 * float(drc[l]) * n1 * n2 + float(dcc[l]) * n2 * n2
            b = float(dr[l]) * n1 + float(dc[l]) * n2

            ok, t = _solve_linear_jit(a, b)
            if ok == 0:
                continue

            p1 = t * n1
            p2 = t * n2

            if abs(p1) <= pixel_boundary and abs(p2) <= pixel_boundary:
                if val >= low:
                    ismax[l] = 2 if val >= high else 1
                nx[l] = np.float32(n1)
                ny[l] = np.float32(n2)
                px[l] = np.float32(r + p1)
                py[l] = np.float32(c + p2)

    return ismax, ev, nx, ny, px, py

# -------------------------
# Numba kernels for link_contours
# -------------------------

import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def _bilinear_ev(ev1d, x, y, w, h):
    # ev is 1D flattened (h*w,), x=row, y=col but px/py are subpixel in the same coord system you used.
    # Here x,y are float pixel coords in image space (row/col).
    """Provide an internal helper for bilinear ev.
    
    Args:
        ev1d (Any): Value for ev1d interpreted by this routine.
        x (Any): Coordinate component representing the x axis value.
        y (Any): Coordinate component representing the y axis value.
        w (Any): Value for w interpreted by this routine.
        h (Any): Value for h interpreted by this routine.
    
    Returns:
        Any: Object containing the bilinear ev outcome for downstream use."""
    if x < 0.0: x = 0.0
    if y < 0.0: y = 0.0
    if x > h - 1.001: x = h - 1.001
    if y > w - 1.001: y = w - 1.001

    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    if x1 >= h: x1 = h - 1
    if y1 >= w: y1 = w - 1

    dx = x - x0
    dy = y - y0

    p00 = x0 * w + y0
    p01 = x0 * w + y1
    p10 = x1 * w + y0
    p11 = x1 * w + y1

    v00 = ev1d[p00]
    v01 = ev1d[p01]
    v10 = ev1d[p10]
    v11 = ev1d[p11]

    v0 = v00 * (1.0 - dy) + v01 * dy
    v1 = v10 * (1.0 - dy) + v11 * dy
    return v0 * (1.0 - dx) + v1 * dx


@nb.njit(cache=True, fastmath=True, boundscheck=True)
def _trace_one_direction(
    x0, y0,
    dir_flag,                  # 1 or 2 (same semantics as your it loop)
    ismax, ev, px, py,
    alpha_arr, oct4_arr, beta0_arr,
    label, indx, cross_done,
    w, h,
    DIRTAB, CLEARTAB,
    max_angle_diff,
    line_id,
    # output buffers:
    out_x, out_y, out_ang, out_resp,
    out_start
):
    """
    Traces one direction starting at (x0,y0) excluding the seed itself.
    Writes points to flat arrays starting at out_start.
    Returns (out_end, hit_existing_label, hit_label_id_minus1, hit_x, hit_y).
    """
    x = x0
    y = y0
    p = x * w + y

    alpha_p = alpha_arr[p]
    last_oct = int(oct4_arr[p])  # 0..3
    if dir_flag == 2:
        last_oct += 4

    last_beta = alpha_p + 0.5 * np.pi
    if last_beta >= np.pi:
        last_beta -= np.pi

    out_i = out_start
    hit = 0
    hit_a = -1
    hit_b = -1
    hit_px = 0.0
    hit_py = 0.0

    while True:
        p = x * w + y
        pxx = px[p]
        pyy = py[p]

        alpha_p = alpha_arr[p]
        octant = int(oct4_arr[p])  # 0..3

        # unwrap octant to 0..7 as in your code
        if octant == 0 and 3 <= last_oct <= 5:
            octant = 4
        elif octant == 1 and 4 <= last_oct <= 6:
            octant = 5
        elif octant == 2 and 4 <= last_oct <= 7:
            octant = 6
        elif octant == 3 and (last_oct == 0 or last_oct >= 6):
            octant = 7
        last_oct = octant

        # choose next among 3 candidates
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

            npx = px[pp]
            npy = py[pp]
            dx = npx - pxx
            dy = npy - pyy
            dist = np.sqrt(dx*dx + dy*dy)

            na = alpha_arr[pp]
            diff = abs(alpha_p - na)
            if diff >= 0.5*np.pi:
                diff = np.pi - diff

            score = dist + diff
            if score < mindiff:
                mindiff = score
                nexti = i
            nextismax = 1

        # clear double responses (same as you do)
        for i in range(2):
            xx = x + int(CLEARTAB[octant, i, 0])
            yy = y + int(CLEARTAB[octant, i, 1])
            if 0 <= xx < h and 0 <= yy < w:
                pp = xx * w + yy
                if ismax[pp] > 0:
                    na = alpha_arr[pp]
                    diff = abs(alpha_p - na)
                    if diff >= 0.5*np.pi:
                        diff = np.pi - diff
                    if diff < max_angle_diff:
                        label[pp] = line_id
                        idx = indx[pp]
                        if idx != 0:
                            cross_done[idx - 1] = True

        if nextismax == 0:
            break

        # step
        x += int(DIRTAB[octant, nexti, 0])
        y += int(DIRTAB[octant, nexti, 1])
        pp = x * w + y

        if out_i >= out_x.shape[0]:
            return out_i, 0, -1, -1, 0.0, 0.0


        # append point (excluding seed)
        out_x[out_i] = px[pp]
        out_y[out_i] = py[pp]

        beta = beta0_arr[pp]  # [0,π)
        d1 = abs(beta - last_beta)
        if d1 >= np.pi:
            d1 = 2.0*np.pi - d1
        d2 = abs(beta + np.pi - last_beta)
        if d2 >= np.pi:
            d2 = 2.0*np.pi - d2

        if d1 < d2:
            out_ang[out_i] = beta
            last_beta = beta
        else:
            out_ang[out_i] = beta + np.pi
            last_beta = beta + np.pi

        out_resp[out_i] = _bilinear_ev(ev, x, y, w, h)  # uses integer x,y like your call did
        out_i += 1

        if label[pp] > 0:
            hit = 1
            hit_a = label[pp] - 1
            hit_b = line_id - 1
            hit_px = px[pp]
            hit_py = py[pp]
            break

        label[pp] = line_id
        idx = indx[pp]
        if idx != 0:
            cross_done[idx - 1] = True

    return out_i, hit, hit_a, hit_b, hit_px, hit_py


@nb.njit(cache=True, fastmath=True, boundscheck=True)
def _link_contours_trace_numba(
    cross_x, cross_y, cross_val, cross_done,
    ismax, ev, px, py,
    alpha_arr, oct4_arr, beta0_arr,
    indx,
    w, h,
    DIRTAB, CLEARTAB,
    max_angle_diff
):
    """Provide an internal helper for link contours trace numba.
    
    Args:
        cross_x (Any): Value for cross x interpreted by this routine.
        cross_y (Any): Value for cross y interpreted by this routine.
        cross_val (Any): Value for cross val interpreted by this routine.
        cross_done (Any): Value for cross done interpreted by this routine.
        ismax (Any): Value for ismax interpreted by this routine.
        ev (Any): Value for ev interpreted by this routine.
        px (Any): Value for px interpreted by this routine.
        py (Any): Value for py interpreted by this routine.
        alpha_arr (Any): Array-like data carrying alpha arr values for analysis.
        oct4_arr (Any): Array-like data carrying oct4 arr values for analysis.
        beta0_arr (Any): Array-like data carrying beta0 arr values for analysis.
        indx (Any): Value for indx interpreted by this routine.
        w (Any): Value for w interpreted by this routine.
        h (Any): Value for h interpreted by this routine.
        DIRTAB (Any): Filesystem location describing where d i r t a b is read or written.
        CLEARTAB (Any): Value for c l e a r t a b interpreted by this routine.
        max_angle_diff (Any): Numeric control parameter for max angle diff during computation.
    
    Returns:
        Any: Object containing the link contours trace numba outcome for downstream use."""
    area = cross_x.shape[0]
    label = np.zeros((w*h,), dtype=np.int32)

    # Worst-case points: each ridge pixel at most once per line, but safe upper bound is area*2
    max_points = area * 2 + 8
    pts_x = np.empty((max_points,), dtype=np.float32)
    pts_y = np.empty((max_points,), dtype=np.float32)
    pts_ang = np.empty((max_points,), dtype=np.float32)
    pts_resp = np.empty((max_points,), dtype=np.float32)

    # line offsets (start index into flat arrays)
    line_offsets = np.empty((area + 1,), dtype=np.int32)
    line_count = 0
    out_i = 0

    # junctions (upper bound: number of lines)
    j_x = np.empty((area,), dtype=np.float32)
    j_y = np.empty((area,), dtype=np.float32)
    j_a = np.empty((area,), dtype=np.int32)
    j_b = np.empty((area,), dtype=np.int32)
    j_count = 0

    indx_max = 0
    while True:
        while indx_max < area and cross_done[indx_max]:
            indx_max += 1
        if indx_max == area:
            break
        maxv = float(cross_val[indx_max])
        if maxv == 0.0:
            break

        x0 = int(cross_x[indx_max])
        y0 = int(cross_y[indx_max])
        pos0 = x0*w + y0

        line_id = line_count + 1
        label[pos0] = line_id
        idx0 = indx[pos0]
        if idx0 != 0:
            cross_done[idx0 - 1] = True

        if out_i >= pts_x.shape[0]:
            break  # or return what you have


        # seed point
        line_offsets[line_count] = out_i
        pts_x[out_i] = px[pos0]
        pts_y[out_i] = py[pos0]

        alpha0 = alpha_arr[pos0]               # [0,π)
        beta_seed = alpha0 + 0.5*np.pi
        if beta_seed >= np.pi:
            beta_seed -= np.pi
        pts_ang[out_i] = beta_seed
        pts_resp[out_i] = _bilinear_ev(ev, x0, y0, w, h)
        out_i += 1

        # clear double responses around seed (uses oct4)
        octant0 = int(oct4_arr[pos0])
        for i in range(2):
            xx = x0 + int(CLEARTAB[octant0, i, 0])
            yy = y0 + int(CLEARTAB[octant0, i, 1])
            if 0 <= xx < h and 0 <= yy < w:
                p = xx*w + yy
                if ismax[p] > 0:
                    na = alpha_arr[p]
                    diff = abs(alpha0 - na)
                    if diff >= 0.5*np.pi:
                        diff = np.pi - diff
                    if diff < max_angle_diff:
                        label[p] = line_id
                        idxp = indx[p]
                        if idxp != 0:
                            cross_done[idxp - 1] = True

        # Trace direction 2 (backward) into temp, then direction 1 (forward) into main, so order matches your reverse+append behavior.
        tmp_start = out_i
        tmp_end, hit2, ha2, hb2, hpx2, hpy2 = _trace_one_direction(
            x0, y0, 2,
            ismax, ev, px, py,
            alpha_arr, oct4_arr, beta0_arr,
            label, indx, cross_done,
            w, h,
            DIRTAB, CLEARTAB,
            max_angle_diff,
            line_id,
            pts_x, pts_y, pts_ang, pts_resp,
            tmp_start
        )
        tmp_len = tmp_end - tmp_start

        # reverse the backward segment in-place (so it becomes prepended)
        for k in range(tmp_len // 2):
            a = tmp_start + k
            b = tmp_start + (tmp_len - 1 - k)
            # swap x,y,ang,resp
            tx = pts_x[a]; pts_x[a] = pts_x[b]; pts_x[b] = tx
            ty = pts_y[a]; pts_y[a] = pts_y[b]; pts_y[b] = ty
            ta = pts_ang[a]; pts_ang[a] = pts_ang[b]; pts_ang[b] = ta
            tr = pts_resp[a]; pts_resp[a] = pts_resp[b]; pts_resp[b] = tr

        out_i = tmp_end

        # forward direction appends after seed+backward
        out_i, hit1, ha1, hb1, hpx1, hpy1 = _trace_one_direction(
            x0, y0, 1,
            ismax, ev, px, py,
            alpha_arr, oct4_arr, beta0_arr,
            label, indx, cross_done,
            w, h,
            DIRTAB, CLEARTAB,
            max_angle_diff,
            line_id,
            pts_x, pts_y, pts_ang, pts_resp,
            out_i
        )

        # junction (if either direction hit)
        if hit2 == 1:
            j_x[j_count] = hpx2; j_y[j_count] = hpy2
            j_a[j_count] = ha2;  j_b[j_count] = hb2
            j_count += 1
        elif hit1 == 1:
            j_x[j_count] = hpx1; j_y[j_count] = hpy1
            j_a[j_count] = ha1;  j_b[j_count] = hb1
            j_count += 1

        # keep only lines with >=2 points
        line_len = out_i - line_offsets[line_count]
        if line_len > 1:
            line_count += 1
        else:
            # drop seed point
            out_i = line_offsets[line_count]
            label[pos0] = 0

    line_offsets[line_count] = out_i

    # trim outputs
    pts_x = pts_x[:out_i]
    pts_y = pts_y[:out_i]
    pts_ang = pts_ang[:out_i]
    pts_resp = pts_resp[:out_i]
    line_offsets = line_offsets[:line_count+1]

    j_x = j_x[:j_count]; j_y = j_y[:j_count]
    j_a = j_a[:j_count]; j_b = j_b[:j_count]

    return pts_x, pts_y, pts_ang, pts_resp, line_offsets, j_x, j_y, j_a, j_b



# -------------------------
# Detector
# -------------------------

class RidgeDetector:
    """Represent ridge detector state used across processing and visualization workflows."""
    DERIV_R, DERIV_C, DERIV_RR, DERIV_RC, DERIV_CC = 1, 2, 3, 4, 5
    MODE_LIGHT, MODE_DARK = 1, 2

    MAX_SIZE_MASK_0 = 3.09023230616781
    MAX_SIZE_MASK_1 = 3.46087178201605
    MAX_SIZE_MASK_2 = 3.82922419517181

    SQRT_2_PI_INV = 0.398942280401432677939946059935
    PIXEL_BOUNDARY = 0.6
    MAX_ANGLE_DIFFERENCE = math.pi / 6.0

    _DIRTAB = np.array([
        [[ 1, 0],[ 1,-1],[ 1, 1]],
        [[ 1, 1],[ 1, 0],[ 0, 1]],
        [[ 0, 1],[ 1, 1],[-1, 1]],
        [[-1, 1],[ 0, 1],[-1, 0]],
        [[-1, 0],[-1, 1],[-1,-1]],
        [[-1,-1],[-1, 0],[ 0,-1]],
        [[ 0,-1],[-1,-1],[ 1,-1]],
        [[ 1,-1],[ 0,-1],[ 1, 0]],
    ], dtype=np.int32)

    _CLEARTAB = np.array([
        [[ 0, 1],[ 0,-1]],
        [[-1, 1],[ 1,-1]],
        [[-1, 0],[ 1, 0]],
        [[-1,-1],[ 1, 1]],
        [[ 0,-1],[ 0, 1]],
        [[ 1,-1],[-1, 1]],
        [[ 1, 0],[-1, 0]],
        [[ 1, 1],[-1,-1]],
    ], dtype=np.int32)

    # -------------------------
    # Public API
    # -------------------------

    @classmethod
    def detect_ridges(
        cls,
        image2d: np.ndarray,
        *,
        sigma: float = 1.5,
        low: float = 5.0,
        high: float = 10.0,
        polarity: str = "dark",
    ) -> Tuple[List[RidgeLine], List[Junction]]:
        """Orchestrate detect ridges within the current module workflow.
        
        Args:
            image2d (np.ndarray): Array-like data carrying image2d values for analysis.
            sigma (float): Numeric control parameter for sigma during computation.
            low (float): Value for low interpreted by this routine.
            high (float): Value for high interpreted by this routine.
            polarity (str): Value for polarity interpreted by this routine.
        
        Returns:
            Tuple[List[RidgeLine], List[Junction]]: Tuple grouping the detect ridges outputs in a stable positional order.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        img = np.asarray(image2d, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError("image must be 2D grayscale")

        h, w = img.shape
        mode = cls.MODE_DARK if polarity.lower() == "dark" else cls.MODE_LIGHT

        dr, dc, drr, drc, dcc = cls._gauss_derivatives(img, sigma)
        ismax, ev, nx, ny, px, py = cls._ridge_candidates(dr, dc, drr, drc, dcc, w, h, low, high, mode)

        alpha, oct4, beta0 = cls._precompute_angles(nx, ny, w, h)

        return cls._link_contours(ismax, ev, nx, ny, px, py, w, h, alpha, oct4, beta0)

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _mask_size(maxv: float, sigma: float) -> int:
        """Provide an internal helper for mask size.
        
        Args:
            maxv (float): Value for maxv interpreted by this routine.
            sigma (float): Numeric control parameter for sigma during computation.
        
        Returns:
            int: Result object produced by the mask size routine."""
        return int(math.ceil(maxv * sigma))

    @staticmethod
    def _lin(r: int, c: int, w: int) -> int:
        """Provide an internal helper for lin.
        
        Args:
            r (int): Value for r interpreted by this routine.
            c (int): Value for c interpreted by this routine.
            w (int): Value for w interpreted by this routine.
        
        Returns:
            int: Result object produced by the lin routine."""
        return r * w + c

    @staticmethod
    def _mirror_r(r: int, h: int) -> int:
        """Provide an internal helper for mirror r.
        
        Args:
            r (int): Value for r interpreted by this routine.
            h (int): Value for h interpreted by this routine.
        
        Returns:
            int: Result object produced by the mirror r routine."""
        return (-r) if r < 0 else (h - r + h - 2) if r >= h else r

    @staticmethod
    def _mirror_c(c: int, w: int) -> int:
        """Provide an internal helper for mirror c.
        
        Args:
            c (int): Value for c interpreted by this routine.
            w (int): Value for w interpreted by this routine.
        
        Returns:
            int: Result object produced by the mirror c routine."""
        return (-c) if c < 0 else (w - c + w - 2) if c >= w else c

    # -------------------------
    # Gaussian masks (small; Python is fine)
    # -------------------------

    @staticmethod
    def _phi0(x: float, sigma: float) -> float:
        """Provide an internal helper for phi0.
        
        Args:
            x (float): Coordinate component representing the x axis value.
            sigma (float): Numeric control parameter for sigma during computation.
        
        Returns:
            float: Result object produced by the phi0 routine."""
        return 0.5 * (1.0 + math.erf(x / (sigma * math.sqrt(2.0))))

    @classmethod
    def _phi1(cls, x: float, sigma: float) -> float:
        """Provide an internal helper for phi1.
        
        Args:
            x (float): Coordinate component representing the x axis value.
            sigma (float): Numeric control parameter for sigma during computation.
        
        Returns:
            float: Result object produced by the phi1 routine."""
        t = x / sigma
        return cls.SQRT_2_PI_INV / sigma * math.exp(-0.5 * t * t)

    @classmethod
    def _phi2(cls, x: float, sigma: float) -> float:
        """Provide an internal helper for phi2.
        
        Args:
            x (float): Coordinate component representing the x axis value.
            sigma (float): Numeric control parameter for sigma during computation.
        
        Returns:
            float: Result object produced by the phi2 routine."""
        t = x / sigma
        return -x * cls.SQRT_2_PI_INV / (sigma ** 3.0) * math.exp(-0.5 * t * t)

    @classmethod
    def _gauss_mask_0(cls, sigma: float) -> Tuple[np.ndarray, int]:
        """Provide an internal helper for gauss mask 0.
        
        Args:
            sigma (float): Numeric control parameter for sigma during computation.
        
        Returns:
            Tuple[np.ndarray, int]: Array containing the gauss mask 0 output aligned with downstream expectations."""
        n = cls._mask_size(cls.MAX_SIZE_MASK_0, sigma)
        h = np.zeros((2 * n + 1,), dtype=np.float64)
        for i in range(-n + 1, n):
            h[n + i] = cls._phi0(-i + 0.5, sigma) - cls._phi0(-i - 0.5, sigma)
        h[0] = 1.0 - cls._phi0(n - 0.5, sigma)
        h[2 * n] = cls._phi0(-n + 0.5, sigma)
        return h, n

    @classmethod
    def _gauss_mask_1(cls, sigma: float) -> Tuple[np.ndarray, int]:
        """Provide an internal helper for gauss mask 1.
        
        Args:
            sigma (float): Numeric control parameter for sigma during computation.
        
        Returns:
            Tuple[np.ndarray, int]: Array containing the gauss mask 1 output aligned with downstream expectations."""
        n = cls._mask_size(cls.MAX_SIZE_MASK_1, sigma)
        h = np.zeros((2 * n + 1,), dtype=np.float64)
        for i in range(-n + 1, n):
            h[n + i] = cls._phi1(-i + 0.5, sigma) - cls._phi1(-i - 0.5, sigma)
        h[0] = -cls._phi1(n - 0.5, sigma)
        h[2 * n] = cls._phi1(-n + 0.5, sigma)
        return h, n

    @classmethod
    def _gauss_mask_2(cls, sigma: float) -> Tuple[np.ndarray, int]:
        """Provide an internal helper for gauss mask 2.
        
        Args:
            sigma (float): Numeric control parameter for sigma during computation.
        
        Returns:
            Tuple[np.ndarray, int]: Array containing the gauss mask 2 output aligned with downstream expectations."""
        n = cls._mask_size(cls.MAX_SIZE_MASK_2, sigma)
        h = np.zeros((2 * n + 1,), dtype=np.float64)
        for i in range(-n + 1, n):
            h[n + i] = cls._phi2(-i + 0.5, sigma) - cls._phi2(-i - 0.5, sigma)
        h[0] = -cls._phi2(n - 0.5, sigma)
        h[2 * n] = cls._phi2(-n + 0.5, sigma)
        return h, n

    # -------------------------
    # Convolution (Numba-accelerated)
    # -------------------------

    @classmethod
    def _convolve_gauss(cls, image_flat: np.ndarray, w: int, h: int, sigma: float, deriv: int) -> np.ndarray:
        """Provide an internal helper for convolve gauss.
        
        Args:
            image_flat (np.ndarray): Array-like data carrying image flat values for analysis.
            w (int): Value for w interpreted by this routine.
            h (int): Value for h interpreted by this routine.
            sigma (float): Numeric control parameter for sigma during computation.
            deriv (int): Value for deriv interpreted by this routine.
        
        Returns:
            np.ndarray: Array containing the convolve gauss output aligned with downstream expectations.
        
        Raises:
            ValueError: If provided values violate validation or domain constraints."""
        if deriv == cls.DERIV_R:
            hr, nr = cls._gauss_mask_1(sigma); hc, nc = cls._gauss_mask_0(sigma)
        elif deriv == cls.DERIV_C:
            hr, nr = cls._gauss_mask_0(sigma); hc, nc = cls._gauss_mask_1(sigma)
        elif deriv == cls.DERIV_RR:
            hr, nr = cls._gauss_mask_2(sigma); hc, nc = cls._gauss_mask_0(sigma)
        elif deriv == cls.DERIV_RC:
            hr, nr = cls._gauss_mask_1(sigma); hc, nc = cls._gauss_mask_1(sigma)
        elif deriv == cls.DERIV_CC:
            hr, nr = cls._gauss_mask_0(sigma); hc, nc = cls._gauss_mask_2(sigma)
        else:
            raise ValueError("unknown derivative type")

        # Numba prefers contiguous arrays
        hr = np.ascontiguousarray(hr)
        hc = np.ascontiguousarray(hc)

        tmp = _convolve_rows_flat(image_flat, hr, nr, w, h)
        out = _convolve_cols_flat(tmp, hc, nc, w, h)
        return out

    @classmethod
    def _gauss_derivatives(cls, img: np.ndarray, sigma: float):
        """Provide an internal helper for gauss derivatives.
        
        Args:
            img (np.ndarray): Value for img interpreted by this routine.
            sigma (float): Numeric control parameter for sigma during computation.
        
        Returns:
            Any: Object containing the gauss derivatives outcome for downstream use."""
        h, w = img.shape
        flat = np.ascontiguousarray(img.reshape(-1))

        dr  = cls._convolve_gauss(flat, w, h, sigma, cls.DERIV_R)
        dc  = cls._convolve_gauss(flat, w, h, sigma, cls.DERIV_C)
        drr = cls._convolve_gauss(flat, w, h, sigma, cls.DERIV_RR)
        drc = cls._convolve_gauss(flat, w, h, sigma, cls.DERIV_RC)
        dcc = cls._convolve_gauss(flat, w, h, sigma, cls.DERIV_CC)
        return dr, dc, drr, drc, dcc

    # -------------------------
    # Ridge candidates (Numba-accelerated)
    # -------------------------

    @classmethod
    def _ridge_candidates(cls, dr, dc, drr, drc, dcc, w, h, low, high, mode):
        # mode is 1/2 already. Keep pixel boundary exact.
        """Provide an internal helper for ridge candidates.
        
        Args:
            dr (Any): Value for dr interpreted by this routine.
            dc (Any): Value for dc interpreted by this routine.
            drr (Any): Value for drr interpreted by this routine.
            drc (Any): Value for drc interpreted by this routine.
            dcc (Any): Value for dcc interpreted by this routine.
            w (Any): Value for w interpreted by this routine.
            h (Any): Value for h interpreted by this routine.
            low (Any): Value for low interpreted by this routine.
            high (Any): Value for high interpreted by this routine.
            mode (Any): Value for mode interpreted by this routine.
        
        Returns:
            Any: Object containing the ridge candidates outcome for downstream use."""
        return _ridge_candidates_jit(dr, dc, drr, drc, dcc, w, h, low, high, mode, cls.PIXEL_BOUNDARY)

    # -------------------------
    # Linking (Python; unchanged)
    # -------------------------


    @staticmethod
    def _threshold_runs(ismax: np.ndarray, minv: int, w: int, h: int) -> List[Tuple[int, int, int]]:
        """
        Vectorized replacement for _threshold_runs.
        ismax is flattened (w*h,) as in your code.
        Returns list of (r, cb, ce) with ismax[r, cb:ce+1] >= minv.
        """
        m = (ismax.reshape(h, w) >= minv)  # bool (h,w)

        # Find transitions per row: start where False->True, end where True->False
        # Pad with False at both ends so runs touching borders are handled.
        mp = np.pad(m, ((0, 0), (1, 1)), mode="constant", constant_values=False)

        starts = np.flatnonzero((~mp[:, :-1]) & (mp[:, 1:]))  # indices in flattened (h*(w+1))
        ends   = np.flatnonzero((mp[:, :-1]) & (~mp[:, 1:]))

        # Convert flattened indices to (row, col) in the padded grid
        Wp = w + 1
        rs = starts // Wp
        cs = starts % Wp
        re = ends // Wp
        ce = (ends % Wp) - 1  # inclusive end in original coords

        # starts/ends are aligned per row because of padding; rs==re always
        # Build python list of tuples (row, cb, ce)
        return list(zip(rs.astype(int).tolist(), cs.astype(int).tolist(), ce.astype(int).tolist()))


    @classmethod
    def _interpolate_response(cls, resp: np.ndarray, x: int, y: int, px: float, py: float, w: int, h: int) -> float:
        """Provide an internal helper for interpolate response.
        
        Args:
            resp (np.ndarray): Value for resp interpreted by this routine.
            x (int): Coordinate component representing the x axis value.
            y (int): Coordinate component representing the y axis value.
            px (float): Value for px interpreted by this routine.
            py (float): Value for py interpreted by this routine.
            w (int): Value for w interpreted by this routine.
            h (int): Value for h interpreted by this routine.
        
        Returns:
            float: Result object produced by the interpolate response routine."""
        def at(rr, cc):
            """Orchestrate at within the current module workflow.
            
            Args:
                rr (Any): Value for rr interpreted by this routine.
                cc (Any): Value for cc interpreted by this routine.
            
            Returns:
                Any: Object containing the at outcome for downstream use."""
            return float(resp[cls._lin(cls._mirror_r(rr, h), cls._mirror_c(cc, w), w)])

        i1 = at(x-1, y-1); i2 = at(x-1, y); i3 = at(x-1, y+1)
        i4 = at(x,   y-1); i5 = at(x,   y); i6 = at(x,   y+1)
        i7 = at(x+1, y-1); i8 = at(x+1, y); i9 = at(x+1, y+1)

        t1 = i1+i2+i3; t2 = i4+i5+i6; t3 = i7+i8+i9
        t4 = i1+i4+i7; t5 = i2+i5+i8; t6 = i3+i6+i9
        d   = (-i1 + 2*i2 - i3 + 2*i4 + 5*i5 + 2*i6 - i7 + 2*i8 - i9) / 9.0
        dr  = (t3 - t1) / 6.0
        dc  = (t6 - t4) / 6.0
        drr = (t1 - 2*t2 + t3) / 6.0
        dcc = (t4 - 2*t5 + t6) / 6.0
        drc = (i1 - i3 - i7 + i9) / 4.0
        xx = px - x
        yy = py - y
        return d + xx*dr + yy*dc + xx*xx*drr + xx*yy*drc + yy*yy*dcc
    
    @staticmethod
    def _precompute_angles(nx, ny, w, h):
        # Flatten to 1D to match your indexing (p = x*w + y)
        """Provide an internal helper for precompute angles.
        
        Args:
            nx (Any): Value for nx interpreted by this routine.
            ny (Any): Value for ny interpreted by this routine.
            w (Any): Value for w interpreted by this routine.
            h (Any): Value for h interpreted by this routine.
        
        Returns:
            Any: Object containing the precompute angles outcome for downstream use."""
        nx1 = np.asarray(nx, np.float32).reshape(-1)
        ny1 = np.asarray(ny, np.float32).reshape(-1)

        # alpha = atan2(nx, -ny) normalized to [0, π)
        alpha = np.arctan2(nx1, -ny1).astype(np.float32)
        alpha = np.where(alpha < 0.0, alpha + 2.0*np.pi, alpha)
        alpha = np.where(alpha >= np.pi, alpha - np.pi, alpha)  # now [0,π)

        # oct4 = round(4/pi*alpha) mod 4  (matches your floor(...+0.5))
        oct4 = (np.floor((4.0/np.pi) * alpha + 0.5).astype(np.int32) & 3).astype(np.uint8)

        # beta0 = atan2(ny, nx) normalized to [0, π)
        beta0 = np.arctan2(ny1, nx1).astype(np.float32)
        beta0 = np.where(beta0 < 0.0, beta0 + 2.0*np.pi, beta0)
        beta0 = np.where(beta0 >= np.pi, beta0 - np.pi, beta0)  # [0,π)

        return alpha, oct4, beta0

    @classmethod
    def _link_contours(cls, ismax, ev, nx, ny, px, py, w, h, alpha_arr, oct4_arr, beta0_arr) -> Tuple[List[RidgeLine], List[Junction]]:

        """Provide an internal helper for link contours.
        
        Args:
            ismax (Any): Value for ismax interpreted by this routine.
            ev (Any): Value for ev interpreted by this routine.
            nx (Any): Value for nx interpreted by this routine.
            ny (Any): Value for ny interpreted by this routine.
            px (Any): Value for px interpreted by this routine.
            py (Any): Value for py interpreted by this routine.
            w (Any): Value for w interpreted by this routine.
            h (Any): Value for h interpreted by this routine.
            alpha_arr (Any): Array-like data carrying alpha arr values for analysis.
            oct4_arr (Any): Array-like data carrying oct4 arr values for analysis.
            beta0_arr (Any): Array-like data carrying beta0 arr values for analysis.
        
        Returns:
            Tuple[List[RidgeLine], List[Junction]]: Tuple grouping the link contours outputs in a stable positional order."""
        label = np.zeros((w * h,), dtype=np.int32)
        indx  = np.zeros((w * h,), dtype=np.int32)

        runs = cls._threshold_runs(ismax, 2, w, h)
        area = sum((ce - cb + 1) for (_, cb, ce) in runs)

        if area == 0:
            return [], []

        cross_x = np.empty((area,), dtype=np.int16)
        cross_y = np.empty((area,), dtype=np.int16)
        cross_val = np.empty((area,), dtype=np.float64)
        cross_done = np.zeros((area,), dtype=np.bool_)
        k = 0
        for r, cb, ce in runs:
            base = r * w
            for c in range(cb, ce + 1):
                pos = base + c
                cross_x[k] = r
                cross_y[k] = c
                cross_val[k] = float(ev[pos])
                k += 1

        order = np.argsort(-cross_val, kind="quicksort")
        cross_x = cross_x[order]; cross_y = cross_y[order]; cross_val = cross_val[order]

        for i in range(area):
            pos = int(cross_x[i]) * w + int(cross_y[i])
            indx[pos] = i + 1


        # --- NUMBA TRACE HERE ---
        DIRTAB = np.asarray(cls._DIRTAB, dtype=np.int8)
        CLEARTAB = np.asarray(cls._CLEARTAB, dtype=np.int8)

        pts_x, pts_y, pts_ang, pts_resp, offsets, jx, jy, ja, jb = _link_contours_trace_numba(
            cross_x.astype(np.int32),
            cross_y.astype(np.int32),
            cross_val.astype(np.float32),
            cross_done,                        # bool ok
            ismax.astype(np.uint8),            # or np.int8
            ev.astype(np.float32),
            px.astype(np.float32),
            py.astype(np.float32),
            alpha_arr.astype(np.float32),
            oct4_arr.astype(np.uint8),
            beta0_arr.astype(np.float32),
            indx.astype(np.int32),
            w, h,
            DIRTAB, CLEARTAB,
            float(cls.MAX_ANGLE_DIFFERENCE),
        )

        # --- WRAP NUMBA OUTPUTS TO PY OBJECTS ---
        lines = []
        for i in range(len(offsets) - 1):
            a = int(offsets[i]); b = int(offsets[i+1])
            if b - a <= 1:
                continue
            pts = np.empty((b - a, 2), np.float32)
            pts[:, 0] = pts_x[a:b]
            pts[:, 1] = pts_y[a:b]
            lines.append(RidgeLine(
                pts=pts,
                angle=pts_ang[a:b].copy(),
                response=pts_resp[a:b].copy(),
            ))

        junctions = []
        for i in range(jx.shape[0]):
            junctions.append(Junction(
                pt=np.array([jx[i], jy[i]], dtype=np.float32),
                a=int(ja[i]),
                b=int(jb[i]),
            ))

        return lines, junctions


# -------------------------
# CLI demo
# -------------------------

def main():
    """Run the module entry point and orchestrate command-line execution."""
    import argparse
    from PIL import Image

    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--low", type=float, default=5.0)
    p.add_argument("--high", type=float, default=10.0)
    p.add_argument("--polarity", choices=["dark", "light"], default="light")
    args = p.parse_args()

    img = Image.open(args.input).convert("L")
    arr = np.asarray(img, dtype=np.float32)

    det = RidgeDetector()
    lines, junctions = det.detect_ridges(
        arr, sigma=args.sigma, low=args.low, high=args.high, polarity=args.polarity
    )

    vis = np.stack([arr, arr, arr], axis=-1)
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    for ln in lines:
        rr = np.clip(np.round(ln.pts[:, 0]).astype(int), 0, vis.shape[0] - 1)
        cc = np.clip(np.round(ln.pts[:, 1]).astype(int), 0, vis.shape[1] - 1)
        vis[rr, cc] = (255, 0, 0)

    for j in junctions:
        r = int(round(float(j.pt[0])))
        c = int(round(float(j.pt[1])))
        if 0 <= r < vis.shape[0] and 0 <= c < vis.shape[1]:
            vis[r, c] = (0, 255, 0)

    Image.fromarray(vis).save(args.output)
    print(f"Saved {args.output}. Lines={len(lines)} Junctions={len(junctions)} NumPy+Numba={_HAVE_NUMBA}")


if __name__ == "__main__":
    main()
