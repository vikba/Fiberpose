# ridge_detector.py
from __future__ import annotations
import os, glob
import jpype, jpype.imports
from jpype.types import JFloat
from jpype import getDefaultJVMPath
import numpy as np
import cv2  # used in processing

from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings


import logging
logger = logging.getLogger(__name__)


# Base class for Ridge Detectors

def RidgeDetector(implementation: str = 'steger-java'):
    """Factory function that returns either RidgeDetectorPython or RidgeDetectorJava."""
    if implementation.lower() == 'steger-python':
        return RidgeDetectorPython()
    elif implementation.lower() == 'steger-java':
        return RidgeDetectorJava()
    else:
        raise ValueError(
            f"Invalid implementation: '{implementation}'. "
            f"Must be either 'steger-python' or 'steger-java'."
        )

# -------------------------------------------------------------------------------------
# ---------- Wrapper over ImageJ Java implementation of the Steger algorithm ----------
# -------------------------------------------------------------------------------------

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

class RidgeDetectorJava():
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
        junctions_java = detector.getJunctions()
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


# -------------------------------------------------------------------------------------
# ----------  Python implementation of Steger algorithm -------------------------------
# -------------------------------------------------------------------------------------

# ---------- Steger Python implementation ----------

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import numpy as np
from scipy.ndimage import gaussian_filter

class ContourClass(Enum):
    """Line classification based on junction locations"""
    CONT_NO_JUNC = 0      # No junctions
    CONT_START_JUNC = 1   # Junction at start
    CONT_END_JUNC = 2     # Junction at end
    CONT_BOTH_JUNC = 3    # Junctions at both ends
    CONT_CLOSED = 4       # Closed loop

class OverlapOption(Enum):
    """Options for handling overlapping lines"""
    NONE = 0
    SLOPE = 1

@dataclass
class Line:
    """Represents a detected line/contour"""
    row: np.ndarray          # Y coordinates
    col: np.ndarray          # X coordinates
    response: np.ndarray     # Ridge response strength
    angle: Optional[np.ndarray] = None
    width_l: Optional[np.ndarray] = None
    width_r: Optional[np.ndarray] = None
    asymmetry: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    contour_class: ContourClass = ContourClass.CONT_NO_JUNC
    line_id: int = -1
    frame: int = 0
    
    @property
    def num(self) -> int:
        """Number of points in the line"""
        return len(self.row)
    
    def estimate_length(self) -> float:
        """Estimate line length as sum of Euclidean distances"""
        if self.num < 2:
            return 0.0
        dx = np.diff(self.col)
        dy = np.diff(self.row)
        return np.sum(np.sqrt(dx**2 + dy**2))
    
    def get_start_or_end_position(self, x: float, y: float, tol: float = 0.5) -> int:
        """Check if point (x,y) is at start (0) or end (num-1), else return -1"""
        dist_start = np.sqrt((self.col[0] - x)**2 + (self.row[0] - y)**2)
        dist_end = np.sqrt((self.col[-1] - x)**2 + (self.row[-1] - y)**2)
        
        if dist_start < tol:
            return 0
        elif dist_end < tol:
            return self.num - 1
        return -1

@dataclass
class Junction:
    """Represents a junction between two lines"""
    x: float = 0.0
    y: float = 0.0
    cont1: int = 0  # Index of first line
    cont2: int = 0  # Index of second line
    pos: int = 0    # Position on first line
    line_cont1: Optional[Line] = None
    line_cont2: Optional[Line] = None
    
    def get_line1(self) -> Line:
        return self.line_cont1
    
    def get_line2(self) -> Line:
        return self.line_cont2

class RidgeDetectorPython():
    """
    Complete Steger line detection with junction handling and post-processing.
    Replicates the functionality of the Java LineDetector class.
    """
    
    def __init__(self):
        self.lines: List[Line] = []
        self.junctions: List[Junction] = []
        self.next_line_id = 0

    def detect_lines_contours(
        self,
        image: np.ndarray,
        sigma: float = 1.0,
        upper_thresh: float = 1.0,
        lower_thresh: float = 0.5,
        min_length: float = 0.0,
        max_length: float = 0.0,
        is_dark_line: bool = False,
        do_correct_position: bool = True,
        do_estimate_width: bool = False,
        do_extend_line: bool = False,
        overlap_option: OverlapOption = OverlapOption.NONE
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]], dict]:
        """
        Detect lines with full post-processing matching Java implementation.

        Returns:
        --------
        contours: List of (N, 2) arrays with (x, y) coordinates for each line
        junction_points: List of (x, y) tuples for junction locations
        used_params: Dictionary of parameters used
        """

        self.lines = []
        self.junctions = []
        self.next_line_id = 0
        
        # Step 1: Detect ridge points with Steger algorithm
        coords, mask, eigendata = self._detect_ridge_points(
            image, sigma, is_dark_line, upper_thresh
        )
        
        # Step 2: Extract width information if requested
        if do_estimate_width:
            coords = self._estimate_widths(coords, eigendata, mask)
        
        # Step 3: Link ridge points into contours using hysteresis
        self.lines = self._link_contours(
            coords, mask, eigendata, lower_thresh, upper_thresh
        )
        
        # Step 4: Detect junctions
        self.junctions = self._detect_junctions(self.lines)
        
        # Step 5: Fix and reconstruct contours
        self._fix_contours()
        
        # Step 6: Fix and reconstruct junctions
        self.junctions = self._fix_junctions()
        
        # Step 7: Assign line references to junctions
        self._assign_lines_to_junctions()
        
        # Step 8: Split lines at interior junctions
        self._add_additional_junction_points_and_lines()
        
        # Step 9: Reconstruct contour classes
        self._reconstruct_contour_classes()
        
        # Step 10: Handle overlaps if requested
        if overlap_option != OverlapOption.NONE:
            self.lines, self.junctions = self._resolve_overlaps(
                overlap_option
            )
        
        # Step 11: Prune lines by length
        if min_length > 0 or max_length > 0:
            self._prune_contours(min_length, max_length)

        # Convert to contours format (x, y coordinates)
        contours = []
        for line in self.lines:
            contour = np.column_stack([line.col, line.row])
            contours.append(contour)
        
        # Convert junctions to point list
        junction_points = [(int(round(j.x)), int(round(j.y))) for j in self.junctions]
        
        # Parameters dict
        used_params = {
            'sigma': sigma,
            'upper_thresh': upper_thresh,
            'lower_thresh': lower_thresh,
            'min_length': min_length,
            'max_length': max_length,
            'is_dark_line': is_dark_line,
            'do_correct_position': do_correct_position,
            'do_estimate_width': do_estimate_width,
            'do_extend_line': do_extend_line,
            'overlap_option': overlap_option
        }
        
        return contours, junction_points, used_params
    
    def _detect_ridge_points(
        self,
        image: np.ndarray,
        sigma: float,
        is_dark_line: bool,
        hess_thresh: float
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Core Steger algorithm for subpixel ridge detection.
        Returns coordinates, mask, and eigenvalue/eigenvector data.
        """
        image = image.astype(np.float64)
        
        # First derivatives
        Ix = gaussian_filter(image, sigma=sigma, order=(0, 1), mode="nearest")
        Iy = gaussian_filter(image, sigma=sigma, order=(1, 0), mode="nearest")
        
        # Hessian entries
        Ixx = gaussian_filter(image, sigma=sigma, order=(0, 2), mode="nearest")
        Ixy = gaussian_filter(image, sigma=sigma, order=(1, 1), mode="nearest")
        Iyy = gaussian_filter(image, sigma=sigma, order=(2, 0), mode="nearest")
        
        # Build Hessian field
        H = np.stack([np.stack([Ixx, Ixy], axis=-1),
                      np.stack([Ixy, Iyy], axis=-1)], axis=-2)
        evals, evecs = np.linalg.eigh(H)
        
        # Select normal direction based on line type
        mode = "dark" if is_dark_line else "bright"
        if mode == "bright":
            idx = 0  # More negative eigenvalue
            sign_ok = lambda d: d < 0
        else:
            idx = 1  # More positive eigenvalue
            sign_ok = lambda d: d > 0
        
        n = evecs[..., idx]
        nx, ny = n[..., 0], n[..., 1]
        
        # Second derivative along normal
        Hn_x = Ixx * nx + Ixy * ny
        Hn_y = Ixy * nx + Iyy * ny
        denom = nx * Hn_x + ny * Hn_y
        
        # First derivative along normal
        numer = Ix * nx + Iy * ny
        
        # Subpixel offset
        with np.errstate(divide='ignore', invalid='ignore'):
            r = -numer / denom
        
        # Ridge response (magnitude of second derivative)
        response = np.abs(denom)
        
        # Acceptance criteria
        mask = (
            (np.abs(r) <= 0.5) & 
            sign_ok(denom) & 
            (response >= hess_thresh) & 
            np.isfinite(r)
        )
        
        # Subpixel coordinates (row, col)
        rr, cc = np.indices(image.shape)
        sub_r = rr + r * ny
        sub_c = cc + r * nx
        
        coords = np.stack([sub_r[mask], sub_c[mask]], axis=1)
        
        eigendata = {
            'response': response,
            'nx': nx,
            'ny': ny,
            'evals': evals,
            'evecs': evecs
        }
        
        return coords, mask, eigendata
    
    def _estimate_widths(
        self,
        coords: np.ndarray,
        eigendata: dict,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Estimate line widths at each point (placeholder).
        Full implementation would analyze intensity profiles perpendicular to ridge.
        """
        # This is a simplified version - full implementation would be more complex
        return coords
    
    def _link_contours(
        self,
        coords: np.ndarray,
        mask: np.ndarray,
        eigendata: dict,
        lower_thresh: float,
        upper_thresh: float
    ) -> List[Line]:
        """
        Link ridge points into continuous contours using hysteresis thresholding.
        """
        if len(coords) == 0:
            return []
        
        response = eigendata['response']
        
        # Create response map for detected points
        rows = coords[:, 0].astype(int)
        cols = coords[:, 1].astype(int)
        
        # High threshold seeds
        high_mask = mask & (response >= upper_thresh)
        
        # Track processed pixels
        processed = np.zeros_like(mask, dtype=bool)
        
        lines = []
        
        # For each high-threshold point, grow a contour
        for r, c in zip(*np.where(high_mask)):
            if processed[r, c]:
                continue
            
            # Grow contour bidirectionally
            contour = self._grow_contour(
                r, c, mask, response, lower_thresh, processed
            )
            
            if len(contour) > 1:
                contour_arr = np.array(contour)
                line = Line(
                    row=contour_arr[:, 0],
                    col=contour_arr[:, 1],
                    response=response[contour_arr[:, 0].astype(int), 
                                     contour_arr[:, 1].astype(int)],
                    line_id=self.next_line_id
                )
                self.next_line_id += 1
                lines.append(line)
        
        return lines
    
    def _grow_contour(
        self,
        start_r: int,
        start_c: int,
        mask: np.ndarray,
        response: np.ndarray,
        lower_thresh: float,
        processed: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Grow a contour from a seed point by following ridge connectivity.
        """
        contour = [(float(start_r), float(start_c))]
        processed[start_r, start_c] = True
        
        # 8-connected neighborhood
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Grow in both directions
        for direction in [1, -1]:
            current_r, current_c = start_r, start_c
            
            while True:
                # Find best neighbor
                best_r, best_c = None, None
                best_response = -np.inf
                
                for dr, dc in neighbors:
                    nr, nc = current_r + dr, current_c + dc
                    
                    if (0 <= nr < mask.shape[0] and 
                        0 <= nc < mask.shape[1] and
                        mask[nr, nc] and
                        not processed[nr, nc] and
                        response[nr, nc] >= lower_thresh):
                        
                        if response[nr, nc] > best_response:
                            best_response = response[nr, nc]
                            best_r, best_c = nr, nc
                
                if best_r is None:
                    break
                
                processed[best_r, best_c] = True
                
                if direction == 1:
                    contour.append((float(best_r), float(best_c)))
                else:
                    contour.insert(0, (float(best_r), float(best_c)))
                
                current_r, current_c = best_r, best_c
        
        return contour
    
    def _detect_junctions(self, lines: List[Line]) -> List[Junction]:
        """
        Detect junctions where line endpoints are close to each other or to line interiors.
        """
        junctions = []
        
        for i, line1 in enumerate(lines):
            # Check both endpoints of line1
            for pos in [0, line1.num - 1]:
                x1, y1 = line1.col[pos], line1.row[pos]
                
                # Check against other lines
                for j, line2 in enumerate(lines):
                    if i == j:
                        continue
                    
                    # Find closest point on line2 to (x1, y1)
                    min_dist, min_pos = self._min_distance(line2, x1, y1)
                    
                    if min_dist < 2.0:  # Proximity threshold
                        junction = Junction(
                            x=x1,
                            y=y1,
                            cont1=i,
                            cont2=j,
                            pos=min_pos,
                            line_cont1=line1,
                            line_cont2=line2
                        )
                        junctions.append(junction)
        
        return junctions
    
    def _min_distance(
        self,
        line: Line,
        x: float,
        y: float
    ) -> Tuple[float, int]:
        """
        Find minimum distance from point (x,y) to line and return (distance, index).
        """
        distances = np.sqrt((line.col - x)**2 + (line.row - y)**2)
        min_idx = np.argmin(distances)
        return distances[min_idx], min_idx
    
    def _fix_contours(self):
        """
        Remove invalid contours (single points, duplicates).
        """
        # Remove single-point contours
        valid_lines = []
        for line in self.lines:
            if line.num > 1:
                valid_lines.append(line)
            else:
                self._delete_junctions_for_line(line)
        
        self.lines = valid_lines
        
        # Remove duplicate first/last entries
        if len(self.lines) >= 2:
            if self.lines[0].line_id == self.lines[-1].line_id:
                self.lines.pop()
        
        # Reset contour classes
        for line in self.lines:
            line.contour_class = ContourClass.CONT_NO_JUNC
    
    def _fix_junctions(self) -> List[Junction]:
        """
        Reconstruct junctions from line geometry to fix any corrupted data.
        """
        new_junctions = []
        processed_positions = set()
        
        for junction in self.junctions:
            pos_key = (round(junction.x * 10), round(junction.y * 10))
            
            if pos_key in processed_positions:
                continue
            
            processed_positions.add(pos_key)
            
            # Find all lines at this junction position
            main_line = None
            main_line_idx = -1
            main_line_pos = -1
            
            secondary_lines = []
            secondary_line_indices = []
            secondary_line_positions = []
            
            for j, line in enumerate(self.lines):
                min_dist, min_pos = self._min_distance(line, junction.x, junction.y)
                
                if min_dist < 0.5:
                    if min_pos == 0 or min_pos == line.num - 1:
                        # Endpoint - secondary line
                        secondary_lines.append(line)
                        secondary_line_indices.append(j)
                        secondary_line_positions.append(min_pos)
                    else:
                        # Interior point - main line
                        main_line = line
                        main_line_idx = j
                        main_line_pos = min_pos
            
            # Create junctions
            if main_line is not None:
                for k, sec_line in enumerate(secondary_lines):
                    new_junc = Junction(
                        x=junction.x,
                        y=junction.y,
                        cont1=main_line_idx,
                        cont2=secondary_line_indices[k],
                        pos=main_line_pos,
                        line_cont1=main_line,
                        line_cont2=sec_line
                    )
                    new_junctions.append(new_junc)
            else:
                # No main line - connect secondary lines pairwise
                unique_lines = []
                unique_indices = []
                unique_positions = []
                seen_ids = set()
                
                for k, sec_line in enumerate(secondary_lines):
                    if sec_line.line_id not in seen_ids:
                        seen_ids.add(sec_line.line_id)
                        unique_lines.append(sec_line)
                        unique_indices.append(secondary_line_indices[k])
                        unique_positions.append(secondary_line_positions[k])
                
                for k in range(len(unique_lines)):
                    for m in range(k + 1, len(unique_lines)):
                        new_junc = Junction(
                            x=junction.x,
                            y=junction.y,
                            cont1=unique_indices[k],
                            cont2=unique_indices[m],
                            pos=unique_positions[k],
                            line_cont1=unique_lines[k],
                            line_cont2=unique_lines[m]
                        )
                        new_junctions.append(new_junc)
        
        return new_junctions
    
    def _assign_lines_to_junctions(self):
        """
        Assign line references to junctions.
        """
        for junction in self.junctions:
            if 0 <= junction.cont1 < len(self.lines):
                junction.line_cont1 = self.lines[junction.cont1]
            if 0 <= junction.cont2 < len(self.lines):
                junction.line_cont2 = self.lines[junction.cont2]
    
    def _add_additional_junction_points_and_lines(self):
        """
        Split lines at interior junction points and create additional junction connections.
        """
        already_processed = set()
        i = 0
        
        while i < len(self.junctions):
            if i in already_processed:
                i += 1
                continue
            
            split_point = self.junctions[i]
            already_processed.add(i)
            
            # Find all junctions at the same position
            junctions_at_pos = [split_point]
            for j in range(i + 1, len(self.junctions)):
                if j not in already_processed:
                    junc2 = self.junctions[j]
                    if (abs(junc2.x - split_point.x) < 0.01 and 
                        abs(junc2.y - split_point.y) < 0.01):
                        already_processed.add(j)
                        junctions_at_pos.append(junc2)
            
            # Connect all secondary lines with each other
            connected_lines = []
            connected_indices = []
            for junc in junctions_at_pos:
                if junc.line_cont2 not in connected_lines:
                    connected_lines.append(junc.line_cont2)
                    connected_indices.append(junc.cont2)
            
            for j in range(len(connected_lines)):
                for k in range(j + 1, len(connected_lines)):
                    new_junc = Junction(
                        x=split_point.x,
                        y=split_point.y,
                        cont1=connected_indices[j],
                        cont2=connected_indices[k],
                        pos=connected_lines[j].get_start_or_end_position(
                            split_point.x, split_point.y
                        ),
                        line_cont1=connected_lines[j],
                        line_cont2=connected_lines[k]
                    )
                    self.junctions.append(new_junc)
                    already_processed.add(len(self.junctions) - 1)
            
            # Split main line if junction is interior
            line1 = split_point.line_cont1
            pos = split_point.pos
            
            if line1 is not None and pos not in [0, line1.num - 1]:
                self._split_line_at_position(line1, pos, split_point, already_processed)
            
            i += 1
    
    def _split_line_at_position(
        self,
        line: Line,
        pos: int,
        split_point: Junction,
        already_processed: Set[int]
    ):
        """
        Split a line at the given position into two lines.
        """
        # Check if closed contour
        is_closed = (line.col[0] == line.col[-1] and 
                     line.row[0] == line.row[-1])
        
        if is_closed or pos == 0 or pos == line.num - 1:
            return
        
        # Split arrays
        keep_length = pos + 1
        split_size = line.num - pos
        
        # Create new line with data from pos onwards
        new_line = Line(
            row=line.row[pos:].copy(),
            col=line.col[pos:].copy(),
            response=line.response[pos:].copy(),
            angle=line.angle[pos:].copy() if line.angle is not None else None,
            width_l=line.width_l[pos:].copy() if line.width_l is not None else None,
            width_r=line.width_r[pos:].copy() if line.width_r is not None else None,
            contour_class=line.contour_class,
            line_id=self.next_line_id
        )
        self.next_line_id += 1
        self.lines.append(new_line)
        new_line_idx = len(self.lines) - 1
        
        # Update original line to keep only data up to pos
        line.row = line.row[:keep_length].copy()
        line.col = line.col[:keep_length].copy()
        line.response = line.response[:keep_length].copy()
        if line.angle is not None:
            line.angle = line.angle[:keep_length].copy()
        if line.width_l is not None:
            line.width_l = line.width_l[:keep_length].copy()
        if line.width_r is not None:
            line.width_r = line.width_r[:keep_length].copy()
        
        # Update junctions that reference the split portion
        for j, junc in enumerate(self.junctions):
            if junc.cont1 == split_point.cont1 and junc.pos > pos:
                junc.cont1 = new_line_idx
                junc.line_cont1 = new_line
                junc.pos = junc.pos - pos
            
            if junc.cont2 == split_point.cont1:
                min_dist, min_pos = self._min_distance(junc.line_cont2, junc.x, junc.y)
                if min_pos > pos:
                    junc.cont2 = new_line_idx
                    junc.line_cont2 = new_line
    
    def _reconstruct_contour_classes(self):
        """
        Classify contours based on junction locations.
        """
        # Reset all to no junctions
        for line in self.lines:
            line.contour_class = ContourClass.CONT_NO_JUNC
        
        # Identify closed contours
        for line in self.lines:
            if (line.col[0] == line.col[-1] and 
                line.row[0] == line.row[-1]):
                line.contour_class = ContourClass.CONT_CLOSED
        
        # Update based on junctions
        for junction in self.junctions:
            if junction.line_cont1:
                junction.line_cont1.contour_class = self._update_contour_class(
                    junction.line_cont1, junction.pos
                )
            
            if junction.line_cont2:
                pos2 = junction.line_cont2.get_start_or_end_position(
                    junction.x, junction.y
                )
                junction.line_cont2.contour_class = self._update_contour_class(
                    junction.line_cont2, pos2
                )
    
    def _update_contour_class(
        self,
        line: Line,
        pos: int
    ) -> ContourClass:
        """
        Update contour class based on junction position.
        """
        current = line.contour_class
        has_start_junc = (pos == 0)
        has_end_junc = (pos == line.num - 1)
        
        if current == ContourClass.CONT_NO_JUNC:
            if has_start_junc:
                return ContourClass.CONT_START_JUNC
            if has_end_junc:
                return ContourClass.CONT_END_JUNC
        
        if current == ContourClass.CONT_START_JUNC and has_end_junc:
            return ContourClass.CONT_BOTH_JUNC
        
        if current == ContourClass.CONT_END_JUNC and has_start_junc:
            return ContourClass.CONT_BOTH_JUNC
        
        if current == ContourClass.CONT_CLOSED and (has_start_junc or has_end_junc):
            return ContourClass.CONT_BOTH_JUNC
        
        return current
    
    def _resolve_overlaps(
        self,
        overlap_option: OverlapOption
    ) -> Tuple[List[Line], List[Junction]]:
        """
        Handle overlapping lines (placeholder for complex overlap resolution).
        """
        # This is a simplified placeholder
        # Full implementation would be complex and depend on specific overlap strategy
        return self.lines, self.junctions
    
    def _prune_contours(self, min_length: float, max_length: float):
        """
        Remove lines that don't meet length criteria.
        """
        valid_lines = []
        
        for line in self.lines:
            length = line.estimate_length()
            
            if min_length > 0 and length < min_length:
                self._delete_junctions_for_line(line)
                continue
            
            if max_length > 0 and length > max_length:
                self._delete_junctions_for_line(line)
                continue
            
            valid_lines.append(line)
        
        self.lines = valid_lines
    
    def _delete_junctions_for_line(self, line: Line):
        """
        Remove all junctions that reference the given line.
        """
        to_remove = []
        for junction in self.junctions:
            if (junction.line_cont1 and junction.line_cont1.line_id == line.line_id) or \
               (junction.line_cont2 and junction.line_cont2.line_id == line.line_id):
                to_remove.append(junction)
        
        for junction in to_remove:
            self.junctions.remove(junction)

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
                   choices=["steger-java", "steger-python"],
                   default="steger-java",
                   help="Detection method to use")
    args = p.parse_args(argv)

    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Could not read: {args.image}")

    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    rd = RidgeDetector(args.method)

    contours, junctions, _ = rd.detect_lines_contours(
        image=image,
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

    for cnt in contours:
        cv2.polylines(overlay, [cnt.astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imwrite(args.out, overlay)
    print(f"Saved overlay with ridge contours: {args.out}")

if __name__ == "__main__":
    _main()
