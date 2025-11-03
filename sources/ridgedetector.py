# ridge_detector.py
from __future__ import annotations
import os, glob
import jpype, jpype.imports
from jpype.types import JFloat
from jpype import getDefaultJVMPath
import numpy as np
import cv2  # used in processing
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
    args = p.parse_args(argv)

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Could not read: {args.image}")

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

    print("Total ridge contours:", len(contours))
    if img.ndim == 2:
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = img.copy()
    for cnt in contours:
        cv2.polylines(overlay, [cnt.astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(args.out, overlay)
    print(f"Saved overlay with ridge contours: {args.out}")

if __name__ == "__main__":
    _main()
