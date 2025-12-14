
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
from typing import Optional, List, Tuple
import math
import numpy as np

# Optional numba
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn): return fn
        return wrap


# -------------------------
# Data model
# -------------------------

@dataclass(slots=True)
class RidgeLine:
    pts: np.ndarray            # (N,2) float32 [row, col]
    angle: np.ndarray          # (N,)  float32 normal angle beta (radians)
    response: np.ndarray       # (N,)  float32 interpolated ridge response
    width: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    asymmetry: Optional[np.ndarray] = None

    def __post_init__(self):
        self.pts = np.asarray(self.pts, dtype=np.float32)
        self.angle = np.asarray(self.angle, dtype=np.float32)
        self.response = np.asarray(self.response, dtype=np.float32)


@dataclass(slots=True)
class Junction:
    pt: np.ndarray   # (2,) float32 [row, col]
    a: int           # index of line A
    b: int           # index of line B

    def __post_init__(self):
        self.pt = np.asarray(self.pt, dtype=np.float32)


# -------------------------
# Numba kernels
# -------------------------

@njit(cache=True)
def _mirror_r_jit(r: int, h: int) -> int:
    if r < 0:
        return -r
    if r >= h:
        return h - r + h - 2
    return r

@njit(cache=True)
def _mirror_c_jit(c: int, w: int) -> int:
    if c < 0:
        return -c
    if c >= w:
        return w - c + w - 2
    return c

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
def _solve_linear_jit(a: float, b: float) -> Tuple[int, float]:
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
# Detector
# -------------------------

class RidgeDetector:
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

    def detect_ridges(
        self,
        image2d: np.ndarray,
        *,
        sigma: float = 1.5,
        low: float = 5.0,
        high: float = 10.0,
        polarity: str = "dark",
    ) -> Tuple[List[RidgeLine], List[Junction]]:
        img = np.asarray(image2d, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError("image must be 2D grayscale")

        h, w = img.shape
        mode = self.MODE_DARK if polarity.lower() == "dark" else self.MODE_LIGHT

        dr, dc, drr, drc, dcc = self._gauss_derivatives(img, sigma)
        ismax, ev, nx, ny, px, py = self._ridge_candidates(dr, dc, drr, drc, dcc, w, h, low, high, mode)

        return self._link_contours(ismax, ev, nx, ny, px, py, w, h)

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _mask_size(maxv: float, sigma: float) -> int:
        return int(math.ceil(maxv * sigma))

    @staticmethod
    def _lin(r: int, c: int, w: int) -> int:
        return r * w + c

    @staticmethod
    def _mirror_r(r: int, h: int) -> int:
        return (-r) if r < 0 else (h - r + h - 2) if r >= h else r

    @staticmethod
    def _mirror_c(c: int, w: int) -> int:
        return (-c) if c < 0 else (w - c + w - 2) if c >= w else c

    # -------------------------
    # Gaussian masks (small; Python is fine)
    # -------------------------

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
    # Linking (Python; unchanged)
    # -------------------------

    @classmethod
    def _threshold_runs(cls, ismax: np.ndarray, minv: int, w: int, h: int) -> List[Tuple[int, int, int]]:
        runs: List[Tuple[int, int, int]] = []
        inside = False
        cb = 0
        for r in range(h):
            for c in range(w):
                grey = int(ismax[cls._lin(r, c, w)])
                if grey >= minv:
                    if not inside:
                        inside = True
                        cb = c
                else:
                    if inside:
                        inside = False
                        runs.append((r, cb, c - 1))
            if inside:
                inside = False
                runs.append((r, cb, w - 1))
        return runs

    @classmethod
    def _interpolate_response(cls, resp: np.ndarray, x: int, y: int, px: float, py: float, w: int, h: int) -> float:
        def at(rr, cc):
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

    @classmethod
    def _link_contours(cls, ismax, ev, nx, ny, px, py, w, h) -> Tuple[List[RidgeLine], List[Junction]]:
        label = np.zeros((w * h,), dtype=np.int32)
        indx = np.zeros((w * h,), dtype=np.int32)

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
            for c in range(cb, ce + 1):
                pos = r * w + c
                cross_x[k] = r
                cross_y[k] = c
                cross_val[k] = float(ev[pos])
                k += 1

        order = np.argsort(-cross_val, kind="quicksort")
        cross_x = cross_x[order]; cross_y = cross_y[order]; cross_val = cross_val[order]

        for i in range(area):
            pos = int(cross_x[i]) * w + int(cross_y[i])
            indx[pos] = i + 1

        lines: List[RidgeLine] = []
        junctions: List[Junction] = []

        indx_max = 0
        while True:
            while indx_max < area and cross_done[indx_max]:
                indx_max += 1
            if indx_max == area:
                break
            maxv = float(cross_val[indx_max])
            if maxv == 0.0:
                break

            x0 = int(cross_x[indx_max]); y0 = int(cross_y[indx_max])
            pos0 = x0 * w + y0

            row: List[float] = []
            col: List[float] = []
            ang: List[float] = []
            resp: List[float] = []

            label[pos0] = len(lines) + 1
            if indx[pos0] != 0:
                cross_done[indx[pos0] - 1] = True

            row.append(float(px[pos0])); col.append(float(py[pos0]))

            nnx = -float(ny[pos0]); nny = float(nx[pos0])
            alpha = math.atan2(nny, nnx)
            if alpha < 0.0: alpha += 2.0 * math.pi
            if alpha >= math.pi: alpha -= math.pi
            octant = int(math.floor(4.0 / math.pi * alpha + 0.5)) % 4

            beta = alpha + math.pi / 2.0
            if beta >= 2.0 * math.pi: beta -= 2.0 * math.pi
            ang.append(beta)
            resp.append(cls._interpolate_response(ev, x0, y0, float(px[pos0]), float(py[pos0]), w, h))

            # clear "double responses"
            for i in range(2):
                xx = x0 + int(cls._CLEARTAB[octant, i, 0])
                yy = y0 + int(cls._CLEARTAB[octant, i, 1])
                if 0 <= xx < h and 0 <= yy < w:
                    p = xx * w + yy
                    if ismax[p] > 0:
                        nnx2 = -float(ny[p]); nny2 = float(nx[p])
                        na = math.atan2(nny2, nnx2)
                        if na < 0.0: na += 2.0 * math.pi
                        if na >= math.pi: na -= math.pi
                        diff = abs(alpha - na)
                        if diff >= math.pi / 2.0: diff = math.pi - diff
                        if diff < cls.MAX_ANGLE_DIFFERENCE:
                            label[p] = len(lines) + 1
                            if indx[p] != 0:
                                cross_done[indx[p] - 1] = True

            for it in (1, 2):
                x = x0; y = y0

                p = x * w + y
                nnx = -float(ny[p]); nny = float(nx[p])
                alpha = math.atan2(nny, nnx)
                if alpha < 0.0: alpha += 2.0 * math.pi
                if alpha >= math.pi: alpha -= math.pi
                last_oct = int(math.floor(4.0 / math.pi * alpha + 0.5)) % 4
                if it == 2:
                    last_oct += 4

                last_beta = alpha + math.pi / 2.0
                if last_beta >= 2.0 * math.pi: last_beta -= 2.0 * math.pi

                if it == 2:
                    row.reverse(); col.reverse(); ang.reverse(); resp.reverse()

                while True:
                    p = x * w + y
                    nnx = -float(ny[p]); nny = float(nx[p])
                    pxx = float(px[p]); pyy = float(py[p])

                    alpha = math.atan2(nny, nnx)
                    if alpha < 0.0: alpha += 2.0 * math.pi
                    if alpha >= math.pi: alpha -= math.pi
                    octant = int(math.floor(4.0 / math.pi * alpha + 0.5)) % 4

                    if octant == 0 and 3 <= last_oct <= 5: octant = 4
                    elif octant == 1 and 4 <= last_oct <= 6: octant = 5
                    elif octant == 2 and 4 <= last_oct <= 7: octant = 6
                    elif octant == 3 and (last_oct == 0 or last_oct >= 6): octant = 7
                    last_oct = octant

                    nextismax = False
                    nexti = 1
                    mindiff = float("inf")
                    for i in range(3):
                        xx = x + int(cls._DIRTAB[octant, i, 0])
                        yy = y + int(cls._DIRTAB[octant, i, 1])
                        if not (0 <= xx < h and 0 <= yy < w):
                            continue
                        pp = xx * w + yy
                        if ismax[pp] == 0:
                            continue
                        npx = float(px[pp]); npy = float(py[pp])
                        dx = npx - pxx; dy = npy - pyy
                        dist = math.sqrt(dx * dx + dy * dy)

                        nnx2 = -float(ny[pp]); nny2 = float(nx[pp])
                        na = math.atan2(nny2, nnx2)
                        if na < 0.0: na += 2.0 * math.pi
                        if na >= math.pi: na -= math.pi
                        diff = abs(alpha - na)
                        if diff >= math.pi / 2.0: diff = math.pi - diff
                        score = dist + diff
                        if score < mindiff:
                            mindiff = score
                            nexti = i
                        nextismax = True

                    # clear double responses
                    for i in range(2):
                        xx = x + int(cls._CLEARTAB[octant, i, 0])
                        yy = y + int(cls._CLEARTAB[octant, i, 1])
                        if 0 <= xx < h and 0 <= yy < w:
                            pp = xx * w + yy
                            if ismax[pp] > 0:
                                nnx2 = -float(ny[pp]); nny2 = float(nx[pp])
                                na = math.atan2(nny2, nnx2)
                                if na < 0.0: na += 2.0 * math.pi
                                if na >= math.pi: na -= math.pi
                                diff = abs(alpha - na)
                                if diff >= math.pi / 2.0: diff = math.pi - diff
                                if diff < cls.MAX_ANGLE_DIFFERENCE:
                                    label[pp] = len(lines) + 1
                                    if indx[pp] != 0:
                                        cross_done[indx[pp] - 1] = True

                    if not nextismax:
                        break

                    x += int(cls._DIRTAB[octant, nexti, 0])
                    y += int(cls._DIRTAB[octant, nexti, 1])
                    pp = x * w + y

                    row.append(float(px[pp]))
                    col.append(float(py[pp]))

                    # orient normal vs last_beta
                    beta = math.atan2(float(ny[pp]), float(nx[pp]))
                    if beta < 0.0: beta += 2.0 * math.pi
                    if beta >= math.pi: beta -= math.pi
                    d1 = abs(beta - last_beta);  d1 = 2.0*math.pi - d1 if d1 >= math.pi else d1
                    d2 = abs(beta + math.pi - last_beta); d2 = 2.0*math.pi - d2 if d2 >= math.pi else d2
                    if d1 < d2:
                        ang.append(beta); last_beta = beta
                    else:
                        ang.append(beta + math.pi); last_beta = beta + math.pi

                    resp.append(cls._interpolate_response(ev, x, y, float(px[pp]), float(py[pp]), w, h))

                    if label[pp] > 0:
                        jpt = np.array([float(px[pp]), float(py[pp])], dtype=np.float32)
                        junctions.append(Junction(pt=jpt, a=label[pp]-1, b=len(lines)))
                        break

                    label[pp] = len(lines) + 1
                    if indx[pp] != 0:
                        cross_done[indx[pp] - 1] = True

            if len(row) > 1:
                pts = np.empty((len(row), 2), dtype=np.float32)
                pts[:, 0] = np.array(row, dtype=np.float32)
                pts[:, 1] = np.array(col, dtype=np.float32)
                lines.append(RidgeLine(
                    pts=pts,
                    angle=np.array(ang, dtype=np.float32),
                    response=np.array(resp, dtype=np.float32),
                ))
            else:
                label[pos0] = 0

        return lines, junctions


# -------------------------
# CLI demo
# -------------------------

def main():
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
