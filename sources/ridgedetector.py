# ridge_detector.py
from __future__ import annotations
import os, glob
import jpype, jpype.imports
from jpype.types import JFloat
from jpype import getDefaultJVMPath
import numpy as np
import cv2  # used in processing

from scipy.ndimage import gaussian_filter

import logging
logger = logging.getLogger(__name__)

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
    def __init__(self):
        self.jars = _default_jars()
        self._ensure_jvm()  # <-- start JVM before jpype.imports works
        from de.biomedical_imaging.ij.steger import LineDetector, OverlapOption  # type: ignore
        self._JLineDetector = LineDetector
        self._JOverlapOption = OverlapOption


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

    def detect_lines_contours(
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
        contours, _, _ = self.detect_lines_contours(
            image, sigma, upper_thresh, lower_thresh,
            min_length, max_length,
            is_dark_line, do_correct_position
        )
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for cnt in contours:
            cnt = cnt.reshape((-1, 1, 2))
            cv2.polylines(mask, [cnt.astype(np.int32)], isClosed=False,
                          color=255, thickness=1, lineType=cv2.LINE_AA)
        return mask.astype(bool)


# ---------- Steger Python implementation ----------

from scipy.spatial import cKDTree

def connect_ridge_points(coords, max_distance=2.0):
    """Connect nearby ridge points into contours."""
    if len(coords) == 0:
        return []
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(coords)
    visited = np.zeros(len(coords), dtype=bool)
    contours = []
    
    for i in range(len(coords)):
        if visited[i]:
            continue
            
        # Start a new contour
        contour = [coords[i]]
        visited[i] = True
        current = i
        
        # Trace forward
        while True:
            neighbors = tree.query_ball_point(coords[current], max_distance)
            neighbors = [n for n in neighbors if not visited[n]]
            if not neighbors:
                break
            current = neighbors[0]
            contour.append(coords[current])
            visited[current] = True
        
        if len(contour) >= 2:  # Only keep contours with at least 2 points
            contours.append(np.array(contour, dtype=np.float32).reshape(-1, 1, 2))
    
    return contours

class RidgeDetectorStegerPython:
    """
    A python implementation of the Steger C. subpixel ridge (line) detection in 2D (1998)
    """
    def __init__(self):
        # self.jars = _default_jars()
        # self._ensure_jvm()  # <-- start JVM before jpype.imports works
        # from de.biomedical_imaging.ij.steger import LineDetector, OverlapOption  # type: ignore
        # self._JLineDetector = LineDetector
        # self._JOverlapOption = OverlapOption
        pass

    def detect_lines_contours(
        self,
        img: np.ndarray,
        sigma: float = 1.0,
        mode:str = "bright",
        hess_thresh:float = 1e-4
    ):
        """
        Steger C. (1998): subpixel ridge (line) detection in 2D.
        Parameters
        ----------
        img : 2D ndarray (float)
            Grayscale image (e.g., a distance transform).
        sigma : float
            Scale (std) for Gaussian derivatives. Choose ~ line half-width.
        mode : {"bright","dark"}
            Detect bright lines on dark background or vice versa.
        hess_thresh : float
            Minimum |n^T H n| to accept (suppresses flat/noisy areas).

        Returns
        -------
        coords : (N, 2) float array
            Subpixel ridge coordinates as (row, col).
        mask : 2D bool array
            Pixel mask of accepted ridge points (pre-subpixel).
        """
        img = img.astype(np.float64)

        # First derivatives (smoothed)
        Ix  = gaussian_filter(img, sigma=sigma, order=(1, 0), mode="nearest")
        Iy  = gaussian_filter(img, sigma=sigma, order=(0, 1), mode="nearest")

        # Hessian entries
        Ixx = gaussian_filter(img, sigma=sigma, order=(2, 0), mode="nearest")
        Ixy = gaussian_filter(img, sigma=sigma, order=(1, 1), mode="nearest")
        Iyy = gaussian_filter(img, sigma=sigma, order=(0, 2), mode="nearest")

        # Build Hessian field, batched eigendecomp
        H = np.stack([np.stack([Ixx, Ixy], axis=-1),
                    np.stack([Ixy, Iyy], axis=-1)], axis=-2)   # shape (H,W,2,2)
        # eigh returns ascending eigenvalues; evecs along last dim
        evals, evecs = np.linalg.eigh(H)                        # shape (H,W,2), (H,W,2,2)

        # For ridges, normal n is eigenvector of the eigenvalue with larger magnitude curvature across the line.
        # Steger: use the eigenvector corresponding to the eigenvalue with *more negative* curvature for bright ridges.
        if mode == "bright":
            idx = 0  # smaller eigenvalue (more negative across a bright line)
            sign_ok = lambda d: d < 0
        else:
            idx = 1  # larger eigenvalue (more positive across a dark line)
            sign_ok = lambda d: d > 0

        n = evecs[..., idx]                    # (H,W,2), unit normal across ridge
        nx, ny = n[..., 0], n[..., 1]

        # Denominator: n^T H n  (second derivative along normal)
        Hn_x = Ixx*nx + Ixy*ny
        Hn_y = Ixy*nx + Iyy*ny
        denom = nx*Hn_x + ny*Hn_y

        # Numerator: ∇I · n
        numer = Ix*nx + Iy*ny

        # Subpixel offset r along n to the ridge center (zero-crossing of first directional derivative)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = -numer / denom

        # Steger acceptance tests:
        # 1) move within the pixel: |r| <= 0.5
        # 2) correct curvature sign: denom < 0 (bright) or > 0 (dark)
        # 3) sufficient curvature magnitude
        mask = (np.abs(r) <= 0.5) & sign_ok(denom) & (np.abs(denom) >= hess_thresh) & np.isfinite(r)

        # Subpixel coordinates (row,col): x' = x + r * n
        rr, cc = np.indices(img.shape)
        sub_r = rr + r*nx
        sub_c = cc + r*ny

        coords = np.stack([sub_r[mask], sub_c[mask]], axis=1)
        return coords, mask


# ---------- CLI / main ----------
def _main(argv=None):
    import argparse, cv2, numpy as np
    p = argparse.ArgumentParser(
        description="Ridge detection wrapper (ImageJ LineDetector via JPype)"
    )
    p.add_argument("image", help="Path to input image")
    p.add_argument("--sigma", type=float, default=2.23)
    p.add_argument("--upper", type=float, default=0.17)
    p.add_argument("--lower", type=float, default=0.1)
    p.add_argument("--minLength", type=float, default=0.0)
    p.add_argument("--maxLength", type=float, default=0.0)
    p.add_argument("--dark", action="store_true", help="isDarkLine")
    p.add_argument("--correct", action="store_true", help="doCorrectPosition")
    p.add_argument("--estWidth", action="store_true", help="doEstimateWidth")
    p.add_argument("--no-extend", dest="extend", action="store_false", help="disable doExtendLine")
    p.add_argument("--overlap", default="NONE")
    p.add_argument("--out", default="overlay.png")
    p.add_argument("--method", 
                   choices=["java", "steger", "numbacs"],  # Add your method names
                   default="java",
                   help="Detection method to use")
    args = p.parse_args(argv)

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Could not read: {args.image}")

    if img.ndim == 2:
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = img.copy()

    if args.method == "java":
        rd = RidgeDetector()
        contours, _, _ = rd.detect_lines_contours(
            image=img,
            sigma=args.sigma,
            upper_thresh=args.upper,
            lower_thresh=args.lower,
            min_length=args.minLength,
            max_length=args.maxLength,
            is_dark_line=args.dark,
            do_correct_position=args.correct,
            do_estimate_width=args.estWidth,
            do_extend_line=args.extend,
            overlap_option=args.overlap,
        )
        for cnt in contours:
            cv2.polylines(overlay, [cnt.astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    elif args.method == "steger":
        rd = RidgeDetectorStegerPython()
        contours_coords, _ = rd.detect_lines_contours(
            img=img,
            sigma=args.sigma,
            mode="dark" if args.dark else "bright",
            hess_thresh=args.lower
        )
        # Convert coords to contours format
        contours = []
        for coord in contours_coords:
            y, x = coord
            contours.append(np.array([[x, y]], dtype=np.float32))

        for cnt in contours:
            for coord in cnt:
                y, x = coord
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)
    else:
        raise NotImplementedError(f"Method '{args.method}' not implemented in this CLI.")

    print("Total ridge contours:", len(contours))

    # for cnt in contours:
    #     # print(cnt.shape)
    #     # cv2.polylines(overlay, [cnt.astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    #     for coord in cnt:
    #         y, x = coord
    #         cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)


    cv2.imwrite(args.out, overlay)
    print(f"Saved overlay with ridge contours: {args.out}")

if __name__ == "__main__":
    _main()
