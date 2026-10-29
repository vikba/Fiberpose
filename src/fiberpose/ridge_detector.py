
"""
Steger ridge/line detection (exact math) with Numba speedups.

- RidgeDetector class with public detect_ridges()
- Public API returns detected polylines as plain NumPy arrays
- Convolution and candidate kernels are JIT-compiled with numba without fastmath to preserve quality
- Contour linking is traced in numba; Python only assembles the output polylines

Usage:
    det = RidgeDetector()
    polylines = det.detect_ridges(img, sigma=2.0, low=8, high=15, polarity="dark")

"""

from __future__ import annotations
from typing import List, Tuple
import math
import numpy as np
import numba as nb


# -------------------------
# Numba kernels
# -------------------------

@nb.njit(cache=True)
def _mirror_r_jit(r: int, h: int) -> int:
    if r < 0:
        return -r
    if r >= h:
        return h - r + h - 2
    return r

@nb.njit(cache=True)
def _mirror_c_jit(c: int, w: int) -> int:
    if c < 0:
        return -c
    if c >= w:
        return w - c + w - 2
    return c

@nb.njit(cache=True)
def _convolve_rows_flat(image: np.ndarray, mask: np.ndarray, n: int, w: int, h: int) -> np.ndarray:
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

@nb.njit(cache=True)
def _convolve_cols_flat(tmp: np.ndarray, mask: np.ndarray, n: int, w: int, h: int) -> np.ndarray:
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

@nb.njit(cache=True)
def _solve_linear_jit(a: float, b: float) -> Tuple[int, float]:
    if a == 0.0:
        return 0, 0.0
    return 1, -b / a

@nb.njit(cache=True)
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

@nb.njit(cache=True)
def _ridge_candidates_jit(dr, dc, drr, drc, dcc, w: int, h: int,
                         low: float, high: float, mode: int,
                         pixel_boundary: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

@nb.njit(cache=True, fastmath=True, boundscheck=True)
def _trace_one_direction(
    x0, y0,
    dir_flag,                  # 1 = forward, 2 = backward
    ismax, px, py,
    alpha_arr, oct4_arr,
    label, indx, cross_done,
    w, h,
    DIRTAB, CLEARTAB,
    max_angle_diff,
    line_id,
    out_x, out_y,
    out_start
):
    """
    Traces one direction starting at (x0,y0) excluding the seed itself.
    Writes points to flat arrays starting at out_start.
    Returns the next free output index.
    """
    x = x0
    y = y0
    p = x * w + y

    alpha_p = alpha_arr[p]
    last_oct = int(oct4_arr[p])  # 0..3
    if dir_flag == 2:
        last_oct += 4

    out_i = out_start

    while True:
        p = x * w + y
        pxx = px[p]
        pyy = py[p]

        alpha_p = alpha_arr[p]
        octant = int(oct4_arr[p])  # 0..3

        # unwrap octant to 0..7
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

        # clear double responses
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
            return out_i


        # append point (excluding seed)
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


@nb.njit(cache=True, fastmath=True, boundscheck=True)
def _link_contours_trace_numba(
    cross_x, cross_y, cross_val, cross_done,
    ismax, px, py,
    alpha_arr, oct4_arr,
    indx,
    w, h,
    DIRTAB, CLEARTAB,
    max_angle_diff
):
    area = cross_x.shape[0]
    label = np.zeros((w*h,), dtype=np.int32)

    # Worst-case points: each ridge pixel at most once per line, but safe upper bound is area*2
    max_points = area * 2 + 8
    pts_x = np.empty((max_points,), dtype=np.float32)
    pts_y = np.empty((max_points,), dtype=np.float32)

    # line offsets (start index into flat arrays)
    line_offsets = np.empty((area + 1,), dtype=np.int32)
    line_count = 0
    out_i = 0

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

        # Trace backward first and reverse it in place, then trace forward, so points run end to end.
        tmp_start = out_i
        tmp_end = _trace_one_direction(
            x0, y0, 2,
            ismax, px, py,
            alpha_arr, oct4_arr,
            label, indx, cross_done,
            w, h,
            DIRTAB, CLEARTAB,
            max_angle_diff,
            line_id,
            pts_x, pts_y,
            tmp_start
        )
        tmp_len = tmp_end - tmp_start

        # reverse the backward segment in-place (so it becomes prepended)
        for k in range(tmp_len // 2):
            a = tmp_start + k
            b = tmp_start + (tmp_len - 1 - k)
            # swap x,y
            tx = pts_x[a]; pts_x[a] = pts_x[b]; pts_x[b] = tx
            ty = pts_y[a]; pts_y[a] = pts_y[b]; pts_y[b] = ty

        out_i = tmp_end

        # forward direction appends after seed+backward
        out_i = _trace_one_direction(
            x0, y0, 1,
            ismax, px, py,
            alpha_arr, oct4_arr,
            label, indx, cross_done,
            w, h,
            DIRTAB, CLEARTAB,
            max_angle_diff,
            line_id,
            pts_x, pts_y,
            out_i
        )

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
    line_offsets = line_offsets[:line_count+1]

    return pts_x, pts_y, line_offsets



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
    ) -> List[np.ndarray]:
        img = np.asarray(image2d, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError("image must be 2D grayscale")

        h, w = img.shape
        mode = cls.MODE_DARK if polarity.lower() == "dark" else cls.MODE_LIGHT

        dr, dc, drr, drc, dcc = cls._gauss_derivatives(img, sigma)
        ismax, ev, nx, ny, px, py = cls._ridge_candidates(dr, dc, drr, drc, dcc, w, h, low, high, mode)

        alpha, oct4 = cls._precompute_angles(nx, ny)
        return cls._link_contours(ismax, ev, px, py, w, h, alpha, oct4)

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _mask_size(maxv: float, sigma: float) -> int:
        return int(math.ceil(maxv * sigma))




    @staticmethod
    def _phi0(x: float, sigma: float) -> float:
        return 0.5 * (1.0 + math.erf(x / (sigma * math.sqrt(2.0))))

    @classmethod
    def _phi1(cls, x: float, sigma: float) -> float:
        t = x / sigma
        return cls.SQRT_2_PI_INV / sigma * math.exp(-0.5 * t * t)

    @classmethod
    def _phi2(cls, x: float, sigma: float) -> float:
        t = x / sigma
        return -x * cls.SQRT_2_PI_INV / (sigma ** 3.0) * math.exp(-0.5 * t * t)

    @classmethod
    def _gauss_mask_0(cls, sigma: float) -> Tuple[np.ndarray, int]:
        n = cls._mask_size(cls.MAX_SIZE_MASK_0, sigma)
        h = np.zeros((2 * n + 1,), dtype=np.float64)
        for i in range(-n + 1, n):
            h[n + i] = cls._phi0(-i + 0.5, sigma) - cls._phi0(-i - 0.5, sigma)
        h[0] = 1.0 - cls._phi0(n - 0.5, sigma)
        h[2 * n] = cls._phi0(-n + 0.5, sigma)
        return h, n

    @classmethod
    def _gauss_mask_1(cls, sigma: float) -> Tuple[np.ndarray, int]:
        n = cls._mask_size(cls.MAX_SIZE_MASK_1, sigma)
        h = np.zeros((2 * n + 1,), dtype=np.float64)
        for i in range(-n + 1, n):
            h[n + i] = cls._phi1(-i + 0.5, sigma) - cls._phi1(-i - 0.5, sigma)
        h[0] = -cls._phi1(n - 0.5, sigma)
        h[2 * n] = cls._phi1(-n + 0.5, sigma)
        return h, n

    @classmethod
    def _gauss_mask_2(cls, sigma: float) -> Tuple[np.ndarray, int]:
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
        return _ridge_candidates_jit(dr, dc, drr, drc, dcc, w, h, low, high, mode, cls.PIXEL_BOUNDARY)

    # -------------------------
    # Linking
    # -------------------------


    @staticmethod
    def _threshold_runs(ismax: np.ndarray, minv: int, w: int, h: int) -> List[Tuple[int, int, int]]:
        """
        ismax is flattened (w*h,).
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


    @staticmethod
    def _precompute_angles(nx, ny):
        # Flatten to 1D to match the kernel indexing (p = x*w + y)
        nx1 = np.asarray(nx, np.float32).reshape(-1)
        ny1 = np.asarray(ny, np.float32).reshape(-1)

        # alpha = atan2(nx, -ny) normalized to [0, π)
        alpha = np.arctan2(nx1, -ny1).astype(np.float32)
        alpha = np.where(alpha < 0.0, alpha + 2.0*np.pi, alpha)
        alpha = np.where(alpha >= np.pi, alpha - np.pi, alpha)  # now [0,π)

        # oct4 = round(4/pi*alpha) mod 4
        oct4 = (np.floor((4.0/np.pi) * alpha + 0.5).astype(np.int32) & 3).astype(np.uint8)

        return alpha, oct4

    @classmethod
    def _link_contours(cls, ismax, ev, px, py, w, h, alpha_arr, oct4_arr) -> List[np.ndarray]:

        label = np.zeros((w * h,), dtype=np.int32)
        indx  = np.zeros((w * h,), dtype=np.int32)

        runs = cls._threshold_runs(ismax, 2, w, h)
        area = sum((ce - cb + 1) for (_, cb, ce) in runs)

        if area == 0:
            return []

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

        pts_x, pts_y, offsets = _link_contours_trace_numba(
            cross_x.astype(np.int32),
            cross_y.astype(np.int32),
            cross_val.astype(np.float32),
            cross_done,                        # bool ok
            ismax.astype(np.uint8),            # or np.int8
            px.astype(np.float32),
            py.astype(np.float32),
            alpha_arr.astype(np.float32),
            oct4_arr.astype(np.uint8),
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
            points_xy = np.empty((b - a, 2), np.float32)
            points_xy[:, 0] = pts_y[a:b]
            points_xy[:, 1] = pts_x[a:b]
            lines.append(points_xy)

        return lines
