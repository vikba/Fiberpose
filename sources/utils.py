import numpy as np

# ------------- core math helpers -------------
def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)

def _bilinear(arr, y, x):
    """Bilinear sample arr at (y, x) with clamped borders. arr is HxW."""
    h, w = arr.shape
    y0 = int(np.floor(y)); x0 = int(np.floor(x))
    y1 = y0 + 1;          x1 = x0 + 1
    y0c = np.clip(y0, 0, h-1); y1c = np.clip(y1, 0, h-1)
    x0c = np.clip(x0, 0, w-1); x1c = np.clip(x1, 0, w-1)
    wy = y - y0; wx = x - x0
    a = arr[y0c, x0c]
    b = arr[y0c, x1c]
    c = arr[y1c, x0c]
    d = arr[y1c, x1c]
    return (1-wy) * ((1-wx)*a + wx*b) + wy * ((1-wx)*c + wx*d)

def _sample_vec2(gx, gy, y, x):
    """Sample gradient vector [dy, dx] at (y,x)."""
    return np.array([_bilinear(gy, y, x), _bilinear(gx, y, x)])

def _tangent2d(centerline, k):
    """Unit tangent at index k using central/one-sided differences. centerline is Nx2 in (y,x)."""
    N = len(centerline)
    if k == 0:
        t = centerline[1] - centerline[0]
    elif k == N-1:
        t = centerline[-1] - centerline[-2]
    else:
        t = centerline[k+1] - centerline[k-1]
    return _normalize(t)

def _rot90_ccw(v):
    """Rotate (y,x) by +90° in image coordinates."""
    return np.array([-v[1], v[0]])

# ------------- normals from DT -------------
def normals_from_dt_2d(centerline, dt, gx, gy, alpha=0.5, iters=3, tau_g=1e-3, eps=1e-8):
    """
    centerline: (N,2) float array of (y,x) points (same pixel coords as dt)
    dt: HxW float distance transform
    gx, gy: HxW gradients of dt (e.g., gx, gy = np.gradient(dt))
    alpha: probe step as fraction of local radius (h = alpha * r)
    iters: fixed-point iterations per point
    tau_g: threshold on |(g+ - g-)| below which we fall back to n0/previous
    returns: (N,2) unit normals (y,x) with continuity enforced
    """
    N = len(centerline)
    normals = np.zeros((N, 2), dtype=np.float64)

    for k in range(N):
        y, x = centerline[k]
        r = _bilinear(dt, y, x)
        h = alpha * max(r, 1e-6)  # always probe a tiny bit, even if r~0

        # initial guess n0 from tangent (rotate by 90°)
        t = _tangent2d(centerline, k)
        n = _rot90_ccw(t)

        # prefer previous orientation if available (continuity)
        if k > 0 and np.dot(n, normals[k-1]) < 0:
            n = -n

        # fixed-point refinement
        for _ in range(max(1, iters)):
            yp, xp = y + h * n[0], x + h * n[1]
            ym, xm = y - h * n[0], x - h * n[1]
            gp = _sample_vec2(gx, gy, yp, xp)   # ∇DT at +h along n
            gm = _sample_vec2(gx, gy, ym, xm)   # ∇DT at -h along n
            d = 0.5 * (gp - gm)                 # averaged direction (normal candidate)
            mag = np.linalg.norm(d)
            if mag < tau_g:
                break
            n_new = d / (mag + eps)
            # keep orientation continuous along the curve
            if k > 0 and np.dot(n_new, normals[k-1]) < 0:
                n_new = -n_new
            n = n_new

        normals[k] = _normalize(n)

    return normals

# ------------- geometry helpers -------------
def resample_polyline_xy(points_xy: np.ndarray, step: float) -> np.ndarray:
    """Resample an open polyline (N,2) in XY to ~equal arc-length spacing of 'step'."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts.copy()

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = cum[-1]
    if L <= 1e-6:
        return pts[:1].copy()

    s_targets = np.arange(0.0, L + 1e-9, float(step))
    if s_targets[-1] < L:
        s_targets = np.append(s_targets, L)

    out = []
    k = 0
    for s in s_targets:
        while k + 1 < len(cum) and cum[k+1] < s:
            k += 1
        if k + 1 >= len(cum):
            out.append(pts[-1])
        else:
            ds = cum[k+1] - cum[k]
            if ds <= 1e-12:
                out.append(pts[k].copy())
            else:
                a = (s - cum[k]) / ds
                out.append(pts[k] * (1 - a) + pts[k+1] * a)
    return np.asarray(out, dtype=np.float64)

def clamp_to_bounds_xy(p: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clamp a 2D point [x, y] to image bounds [0..w-1], [0..h-1]."""
    x = float(np.clip(p[0], 0, w - 1))
    y = float(np.clip(p[1], 0, h - 1))
    return np.array([x, y], dtype=np.float64)
