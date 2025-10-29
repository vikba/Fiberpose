# ridge_detector.py
from __future__ import annotations
import os, glob
import jpype, jpype.imports
from jpype.types import JFloat
from jpype import getDefaultJVMPath
import numpy as np
import cv2  # used in processing

JDK_JVM = getDefaultJVMPath()

# ---------- defaults ----------
def _default_jars():
    here = os.path.dirname(os.path.abspath(__file__))
    jars_dir = os.path.join(here, "jars")
    return sorted(glob.glob(os.path.join(jars_dir, "*.jar")))

# ---------- ImageJ FloatProcessor builders ----------
def _to_floatprocessor_from_numpy(array: np.ndarray):
    from ij.process import FloatProcessor
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    h, w = arr.shape[:2]
    pixels = JFloat[:](arr.reshape(-1))
    return FloatProcessor(int(w), int(h), pixels)

# ---------- main detector facade ----------
class RidgeDetector:
    """
    A well-organized facade for ridge detection using the ImageJ plugin via JPype.
    Provides simple and intuitive methods for detecting lines and obtaining their contours.
    """
    def __init__(self, high_conf_fib: bool = False):
        self.jars = _default_jars()
        self._ensure_jvm()  # <-- start JVM before jpype.imports works
        from de.biomedical_imaging.ij.steger import LineDetector, OverlapOption  # type: ignore
        self._JLineDetector = LineDetector
        self._JOverlapOption = OverlapOption

        self.high_conf_fib = high_conf_fib

    def set_high_conf_filter(self, high_conf_fib: bool):
        self.high_conf_fib = high_conf_fib


    def _ensure_jvm(self):
        if not jpype.isJVMStarted():
            jpype.startJVM(JDK_JVM, classpath=os.pathsep.join(self.jars))

    @staticmethod
    def _lines_to_contours(lines_java) -> list:
        """
        Convert Java line objects into Python contours (numpy array shape (N,1,2)).
        """
        contours = []
        for line in lines_java:
            xs = np.array(line.getXCoordinates(), dtype=np.float32)
            ys = np.array(line.getYCoordinates(), dtype=np.float32)
            if xs.size >= 2:
                pts_xy = np.stack([xs, ys], axis=1)
                pts_xy = pts_xy.round().astype(np.int32).reshape(-1, 1, 2)
                contours.append(pts_xy)
        return contours

    def _detect_ridges(
        self,
        image: np.ndarray,
        sigma: float,
        upper_thresh: float,
        lower_thresh: float,
        min_length: float,
        max_length: float,
        is_dark_line: bool,
        do_correct_position: bool,
        do_estimate_width: bool = False,
        do_extend_line: bool = True,
        overlap_option: str = "NONE",
        bechatty: bool = False,
    ):
        """
        Detect ridges in the provided image using the Java class.

        Returns:
            contours, junction points, and used parameters.
        """

        '''
        Java part
        '''
        self._ensure_jvm()
        from de.biomedical_imaging.ij.steger import LineDetector, OverlapOption  # type: ignore
        fp = _to_floatprocessor_from_numpy(image)
        detector = self._JLineDetector()
        # Uncomment the next line for verbose Java output:
        # detector.bechatty = bechatty
        lines_java = detector.detectLines(
            fp, float(sigma), float(upper_thresh), float(lower_thresh),
            float(min_length), float(max_length),
            bool(is_dark_line), bool(do_correct_position),
            bool(do_estimate_width), bool(do_extend_line),
            self._JOverlapOption.valueOf(str(overlap_option))
        )
        junctions_java= detector.getJunctions()
        used_params = detector.getUsedParamters()


        '''
        Conversion of Java lines to contours
        '''
        contours = self._lines_to_contours(lines_java)
        junction_points = []
        for j in junctions_java:
            jx = int(round(j.getX()))
            jy = int(round(j.getY()))
            junction_points.append((jx, jy))
        return contours, junction_points, used_params


    def _split_and_trim_contour(self, contour, junction_points, trim=3, tol=3.0):
        """
        Private method:
        Splits a contour at points near any junction (within tolerance tol)
        and trims 'trim' pixels from both ends of each resulting segment.
        """
        pts_xy = contour.reshape(-1, 2)
        split_indices = []
        for (jx, jy) in junction_points:
            dists = np.sqrt((pts_xy[:, 0] - jx)**2 + (pts_xy[:, 1] - jy)**2)
            idxs = np.where(dists < tol)[0]
            if idxs.size > 0:
                min_idx = idxs[np.argmin(dists[idxs])]
                split_indices.append(int(min_idx))
        split_indices = sorted(set(split_indices))
        segments = []
        boundaries = [0] + split_indices + [len(pts_xy) - 1]
        for i in range(len(boundaries)-1):
            seg_xy = pts_xy[boundaries[i]:boundaries[i+1]+1]
            if len(seg_xy) > 2 * trim:
                seg_xy = seg_xy[trim:-trim]
            else:
                continue
            if len(seg_xy) >= 2:
                segments.append(seg_xy.reshape(-1, 1, 2))
        return segments

    def _sliding_window_filter(self, seg_xy, image, win_len=15.0, std_thresh=0.6, max_angle=20.0):
        """
        Moves a sliding window along the contour (seg_xy) and returns sub-segments (N,1,2)
        where BOTH hold:
        - std of grayscale intensities in the window < std_thresh
        - arctan(|I_last - I_first| / arc_length) < max_angle (deg)
        The window advances by half its length in arc-length coordinates.
        """
        import cv2
        pts_xy = seg_xy.reshape(-1, 2).astype(np.float32)
        if pts_xy.shape[0] < 2:
            return []

        # cumulative arc length along polyline
        cum_dist = [0.0]
        for i in range(1, len(pts_xy)):
            cum_dist.append(cum_dist[-1] + float(np.linalg.norm(pts_xy[i] - pts_xy[i-1])))
        cum_dist = np.asarray(cum_dist, dtype=np.float32)
        total_len = float(cum_dist[-1])
        if total_len <= 0:
            return []

        # grayscale image
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        H, W = gray.shape[:2]

        # nearest-neighbor sampler (clip to bounds)
        def sample_intensity(p_xy):
            x, y = int(round(p_xy[0])), int(round(p_xy[1]))
            if x < 0: x = 0
            if y < 0: y = 0
            if x >= W: x = W - 1
            if y >= H: y = H - 1
            return float(gray[y, x])

        valid_windows = []
        start_idx = 0
        half_step = float(win_len) / 2.0
        max_angle = float(max_angle)

        N = len(pts_xy)
        while start_idx < N - 1:
            start_s = float(cum_dist[start_idx])
            target_s = start_s + float(win_len)
            end_idx = int(np.searchsorted(cum_dist, target_s, side="left"))

            # ensure at least two points in window
            if end_idx <= start_idx + 1:
                end_idx = start_idx + 1
            if end_idx >= N:
                end_idx = N - 1

            # sample intensities for points in [start_idx, end_idx]
            window_pts_xy = pts_xy[start_idx:end_idx+1]
            intensities = np.fromiter((sample_intensity(p_xy) for p_xy in window_pts_xy), dtype=np.float32)
            if intensities.size >= 2:
                if np.std(intensities) < std_thresh:
                    ds = float(cum_dist[end_idx] - cum_dist[start_idx])
                    if ds > 0.0:
                        dI = abs(float(intensities[-1]) - float(intensities[0]))
                        angle_deg = np.degrees(np.arctan(dI / ds))
                        if angle_deg < max_angle:
                            valid_windows.append((start_idx, end_idx))

            # advance by half window length in arc space
            new_target_s = start_s + half_step
            new_start_idx = int(np.searchsorted(cum_dist, new_target_s, side="left"))
            if new_start_idx <= start_idx:
                break  # no progress possible
            start_idx = new_start_idx

        # merge overlapping/contiguous windows
        merged = []
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

        # build output sub-segments, enforce minimum arc-length >= win_len
        out_segments = []
        for s, e in merged:
            if float(cum_dist[e] - cum_dist[s]) >= float(win_len):
                sub_xy = pts_xy[s:e+1].copy().reshape(-1, 1, 2)
                out_segments.append(sub_xy)
        return out_segments



    def detect_constant_segments(
        self,
        image: np.ndarray,
        sigma: float, upperThresh: float, lowerThresh: float,
        minLength: float, maxLength: float,
        isDarkLine: bool, doCorrectPosition: bool,
        doEstimateWidth: bool = False,
        doExtendLine: bool = True,
        overlapOption: str = "NONE",
        trim: int = 3, tol: float = 3.0,
        win_len: float = 15.0, std_thresh: float = 0.6,
        max_angle: float = 20.0,
        bechatty: bool = False,
    ):

        # Step 1: run the Java ridge detector and collect raw polyline contours plus junction metadata.
        lines_contours, junction_points, used = self._detect_ridges(
            image, sigma, upperThresh, lowerThresh,
            minLength, maxLength,
            isDarkLine, doCorrectPosition,
            doEstimateWidth, doExtendLine,
            overlapOption, bechatty
        )
        processed = []
        # Step 2: split each contour at nearby junctions and trim the ends so spurious junction noise is removed.
        for contour in lines_contours:
            segs_xy = self._split_and_trim_contour(contour, junction_points, trim, tol)
            if segs_xy:
                processed.extend(segs_xy)
            else:
                pts_xy = contour.reshape(-1, 2)
                if len(pts_xy) > 2*trim:
                    trimmed_xy = pts_xy[trim:-trim].reshape(-1, 1, 2)
                    processed.append(trimmed_xy)
        constant_segments_xy = []
        
        # Step 3: slide an intensity window along each trimmed contour and keep only portions with low variation.
        for seg_xy in processed:
            subs_xy = self._sliding_window_filter(seg_xy, image, win_len, std_thresh, max_angle)
            constant_segments_xy.extend(subs_xy)

        if not self.high_conf_fib:
            # Step 4: compute per-vertex normals for every surviving segment so downstream code has orientation data.
            normals_xy = [self._polyline_normals(seg_xy) for seg_xy in constant_segments_xy]
            return constant_segments_xy, normals_xy
        else:
            print(f"RidgeDetector: Applying high confidence fiber filter, {len(constant_segments_xy)} segments before filtering.")
            normals_xy = [self._polyline_normals(seg_xy) for seg_xy in constant_segments_xy]
            return constant_segments_xy, normals_xy
    
    @staticmethod
    def _polyline_normals(pts_xy: np.ndarray,
                            window: int = 8,
                            angle_thresh_deg: float = 60.0,
                            side: str = "left") -> np.ndarray:
        """
        Compute per-vertex unit normals for an OPEN polyline.

        - Smooth segments: averages directions over a +/- window of segments.
        - Sharp corners: uses the (normalized) angle-bisector to retain shape.
        - Endpoints: uses first/last segment direction.
        - 'side': "left" (default) or "right" to flip normals.

        pts: (N,2) array of xy points (int/float).
        returns: (N,2) float32 normals.
        """
        pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
        N = len(pts_xy)
        if N < 2:
            raise ValueError("Need at least 2 points for an open polyline.")

        eps = 1e-12

        # Segment unit directions u[k] for segment k: pts[k+1]-pts[k], k=0..N-2
        seg = pts_xy[1:] - pts_xy[:-1]
        seg_norm = np.linalg.norm(seg, axis=1, keepdims=True) + eps
        u = seg / seg_norm  # (N-1,2)

        def rot90(v):  # rotate +90° (left-hand normal)
            return np.stack([-v[:, 1], v[:, 0]], axis=1)

        def unit(v):
            n = np.linalg.norm(v, axis=1, keepdims=True) + eps
            return v / n

        angle_thresh = np.deg2rad(angle_thresh_deg)

        normals = np.zeros((N, 2), dtype=np.float32)

        # Endpoint normals: just use first/last segment direction
        t0 = u[0:1]
        tn = u[-1:]
        n0 = rot90(t0)
        nn = rot90(tn)
        normals[0] = n0[0]
        normals[-1] = nn[0]

        # Interior vertices
        for i in range(1, N - 1):
            d_prev = u[i - 1:i]        # segment into vertex i
            d_next = u[i:i + 1]        # segment out of vertex i

            # Corner test via angle between adjacent segments
            cosang = np.clip(np.sum(d_prev * d_next, axis=1), -1.0, 1.0)[0]
            ang = np.arccos(cosang)

            if ang >= angle_thresh:
                # Sharp corner: use (normalized) angle bisector as tangent
                t = unit(d_prev + d_next)
                # If nearly opposite (degenerate bisector), fall back to d_next
                if np.linalg.norm(t) < 1e-6:
                    t = d_next
            else:
                # Smooth: average a local window of segment directions
                k0 = max(0, i - window)
                k1 = min(N - 1, i + window)  # note: segments index up to N-2, but slice end is exclusive on u
                # segments that touch vertex i are [i-1, i]; include neighbors around them
                k1 = min(k1, N - 1)  # safety
                # map to segment indices within [0, N-2]
                seg_lo = max(0, i - window)
                seg_hi = min(N - 2, i + window - 1)
                if seg_lo > seg_hi:
                    seg_lo, seg_hi = i - 1, i  # minimal
                inds = np.arange(seg_lo, seg_hi + 1)

                # triangular weights (center near i-0.5)
                center = i - 0.5
                dist = np.abs(inds + 0.5 - center)
                w = (window + 1) - dist  # larger near center
                w = np.clip(w, 1.0, None)
                w = (w / (w.sum() + eps)).reshape(-1, 1)

                t = unit((u[inds] * w).sum(axis=0, keepdims=True))

            n = rot90(t)
            normals[i] = n[0]

        if side == "right":
            normals = -normals
        return normals.astype(np.float32)
    

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect ridges in the image and return a binary mask.
        Uses default parameters for the Java LineDetector.
        """
        sigma = 2.23
        upper_thresh = 0.17
        lower_thresh = 0.0
        min_length = 10.0
        max_length = 0.0
        is_dark_line = True
        do_correct_position = True
        # Use _detect_ridges_java to get Java line objects.
        lines_java, _, _ = self._detect_ridges_java(
            image, sigma, upper_thresh, lower_thresh,
            min_length, max_length,
            is_dark_line, do_correct_position
        )
        contours = self._lines_to_contours(lines_java)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for cnt in contours:
            cnt = cnt.reshape((-1, 1, 2))
            cv2.polylines(mask, [cnt.astype(np.int32)], isClosed=False,
                          color=255, thickness=1, lineType=cv2.LINE_AA)
        return mask.astype(bool)

# ---------- CLI / main ----------
def _main(argv=None):
    import argparse, cv2, numpy as np
    p = argparse.ArgumentParser(
        description="Ridge detection wrapper (ImageJ LineDetector via JPype)"
    )
    p.add_argument("image", help="Path to input image")
    p.add_argument("--sigma", type=float, default=2.23)
    p.add_argument("--upper", type=float, default=0.17)
    p.add_argument("--lower", type=float, default=0.0)
    p.add_argument("--minLength", type=float, default=0.0)
    p.add_argument("--maxLength", type=float, default=0.0)
    p.add_argument("--dark", action="store_true", help="isDarkLine")
    p.add_argument("--correct", action="store_true", help="doCorrectPosition")
    p.add_argument("--estWidth", action="store_true", help="doEstimateWidth")
    p.add_argument("--no-extend", dest="extend", action="store_false", help="disable doExtendLine")
    p.add_argument("--overlap", default="NONE")
    p.add_argument("--out", default="overlay.png")
    p.add_argument("--trim", type=int, default=3, help="Trim n pixels from line ends")
    p.add_argument("--tol", type=float, default=3.0, help="Tolerance for matching junction point")
    p.add_argument("--winLen", type=float, default=25.0, help="Sliding window length in pixels")
    p.add_argument("--stdThresh", type=float, default=0.8, help="Max allowed std of brightness in window")
    p.add_argument("--maxAngle", type=float, default=5.0, help="Max allowed absolute angle in degrees")
    args = p.parse_args(argv)

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Could not read: {args.image}")

    rd = RidgeDetector()
    constant_segments_xy, _ = rd.detect_constant_segments(
        image=img,
        sigma=args.sigma, upperThresh=args.upper, lowerThresh=args.lower,
        minLength=args.minLength, maxLength=args.maxLength,
        isDarkLine=args.dark, doCorrectPosition=args.correct,
        doEstimateWidth=args.estWidth, doExtendLine=args.extend,
        overlapOption=args.overlap,
        trim=args.trim, tol=args.tol,
        win_len=args.winLen, std_thresh=args.stdThresh,
        max_angle=args.maxAngle
    )

    print("Total constant segments:", len(constant_segments_xy))
    overlay = img.copy()
    for seg_xy in constant_segments_xy:
        cv2.polylines(overlay, [seg_xy], isClosed=False, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(args.out, overlay)
    print(f"Saved overlay with constant segments: {args.out}")

if __name__ == "__main__":
    _main()