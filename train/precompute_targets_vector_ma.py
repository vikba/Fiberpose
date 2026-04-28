"""Generate vector-style training targets and previews from mask annotations."""

import os
import json
import numpy as np
import cv2
import tifffile
from pathlib import Path
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
ROOT_DIR = Path("/home/vk/workspace/Training/dataset_extended")
INPUT_DIR = ROOT_DIR / "val_masks"
OUTPUT_DIR = ROOT_DIR / "val_targets"
PREVIEW_DIR = OUTPUT_DIR / "previews"
IMAGE_DIR = ROOT_DIR / "val_images"   # original images folder

MIN_MEAN_BRIGHTNESS = 1.0        # tune to your image scale
MIN_ASPECT_RATIO = 1.0            # major/minor axis or bbox ratio
MIN_AREA = 10

# Contour Smoothing Settings
SMOOTHING_WINDOW = 10  # odd integer, circular moving average

DIST_BG = -5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)




def to_gray_u8(img):
    """Convert values into gray u8 for downstream compatibility.
    
    Args:
        img (Any): Value for img interpreted by this routine.
    
    Returns:
        Any: Object containing the to gray u8 outcome for downstream use."""
    if img is None:
        return None
    if img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if gray.max() <= gray.min():
        return np.zeros_like(gray, dtype=np.uint8)

    # normalize only if not already uint8-like
    if gray.dtype == np.uint8 or (gray.min() >= 0 and gray.max() <= 255):
        return gray.astype(np.uint8)
    gray = 255.0 * (gray - gray.min()) / (gray.max() - gray.min())
    return gray.astype(np.uint8)

def component_mean_brightness(gray_img, comp):
    """Orchestrate component mean brightness within the current module workflow.
    
    Args:
        gray_img (Any): Value for gray img interpreted by this routine.
        comp (Any): Value for comp interpreted by this routine.
    
    Returns:
        Any: Object containing the component mean brightness outcome for downstream use."""
    vals = gray_img[comp > 0]
    if vals.size == 0:
        return 0.0
    return float(vals.mean())

def component_aspect_ratio(comp):
    """Orchestrate component aspect ratio within the current module workflow.
    
    Args:
        comp (Any): Value for comp interpreted by this routine.
    
    Returns:
        Any: Object containing the component aspect ratio outcome for downstream use."""
    ys, xs = np.where(comp > 0)
    if len(xs) < 5:
        return 0.0

    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)   # ((cx,cy),(w,h),angle)
    w, h = rect[1]
    if w <= 1e-6 or h <= 1e-6:
        return 0.0
    a = max(w, h) / max(1e-6, min(w, h))
    return float(a)

def component_circularity(comp):
    """Orchestrate component circularity within the current module workflow.
    
    Args:
        comp (Any): Value for comp interpreted by this routine.
    
    Returns:
        Any: Object containing the component circularity outcome for downstream use."""
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    if perim <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))

def get_gradients(dt):
    """
    Compute raw gradients (Flows).
    Scale=1/32.0 to ensure output range is approx -1.0 to 1.0
    """
    dt32 = dt.astype(np.float32)
    grad_x = cv2.Scharr(dt32, cv2.CV_32F, 1, 0, scale=1.0/32.0)
    grad_y = cv2.Scharr(dt32, cv2.CV_32F, 0, 1, scale=1.0/32.0)
    return grad_x, grad_y

def normalize_to_uint8(data, mode='minmax'):
    """Helper for visualization only."""
    data = data.astype(np.float32)
    
    if mode == 'minmax':
        # Normalizes 0..Max to 0..255
        if data.max() == data.min():
            return np.zeros_like(data, dtype=np.uint8)
        norm = (data - data.min()) / (data.max() - data.min())
        return (norm * 255).astype(np.uint8)
        
    elif mode == 'flow_grayscale':
        # Maps -1..1 to 0..255, centered at 128.
        # We use a factor of 100 instead of 127 to avoid 'clipping' 
        # slight overshoots (noise) into pure white/black.
        visual = 128 + (data * 100)
        return np.clip(visual, 0, 255).astype(np.uint8)

def save_hsv_preview(flow_x, flow_y, filename):
    """
    Creates a color-coded flow map.
    Color = Direction, Intensity = Magnitude.
    This is much better for visualizing smooth curves than grayscale.
    """
    h, w = flow_x.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Cartesian to Polar coordinates
    mag, ang = cv2.cartToPolar(flow_x, flow_y)
    
    # Angle (0..360) -> Hue (0..179 for OpenCV)
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Saturation: Max saturation everywhere
    hsv[..., 1] = 255
    
    # Value: Based on magnitude. 
    # Ideal magnitude is 1.0. We map 0..1.5 to 0..255
    # Pixels with 0 flow (background) will be black.
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to RGB
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(filename), bgr)

def smooth_contour_moving_average(points, window):
    """Orchestrate smooth contour moving average within the current module workflow.
    
    Args:
        points (Any): Value for points interpreted by this routine.
        window (Any): Value for window interpreted by this routine.
    
    Returns:
        Any: Object containing the smooth contour moving average outcome for downstream use."""
    if window <= 1 or len(points) < window:
        return points
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.concatenate([points[-half:], points, points[:half]], axis=0)
    kernel = np.ones(window, dtype=np.float32) / window
    x = np.convolve(padded[:, 0], kernel, mode="valid")
    y = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.stack([x, y], axis=1)

def process_instance_ma(mask_shape, contour, supersample=2):
    """
    Calculates DT on a higher resolution grid to capture the peak better,
    then downsamples back to original resolution.
    """
    h, w = mask_shape
    
    # Work at higher resolution to catch the sub-pixel peak
    sh, sw = h * supersample, w * supersample
    dt_super = np.zeros((sh, sw), dtype=np.float32)
    
    if len(contour) < 4:
        return np.zeros((h, w), dtype=np.float32)

    try:
        pts = contour.reshape(-1, 2).astype(np.float32)
        pts = smooth_contour_moving_average(pts, SMOOTHING_WINDOW)
        smooth_poly = (pts * supersample).reshape(-1, 1, 2).astype(np.float32)
        
        # Bounding Box (Scaled)
        x_b, y_b, w_b, h_b = cv2.boundingRect(smooth_poly)
        pad = 5 * supersample
        x0, x1 = max(0, x_b - pad), min(sw, x_b + w_b + pad)
        y0, y1 = max(0, y_b - pad), min(sh, y_b + h_b + pad)
        
        for r in range(y0, y1):
            for c in range(x0, x1):
                # Float calculation on high-res grid
                dist = cv2.pointPolygonTest(smooth_poly, (float(c), float(r)), True)
                if dist > 0:
                    dt_super[r, c] = dist
                    
        # Downsample back to original size
        # INTER_AREA preserves the peak energy best
        dt_final = cv2.resize(dt_super, (w, h), interpolation=cv2.INTER_AREA)
        
        # Divide distance by supersample factor to return to original units
        dt_final = dt_final / supersample
        
        return dt_final
                    
    except Exception:
        return np.zeros((h, w), dtype=np.float32)

def process_file(mask_path):
    """Orchestrate process file within the current module workflow.
    
    Args:
        mask_path (Any): Filesystem location describing where mask path is read or written.
    
    Returns:
        Any: Object containing the process file outcome for downstream use."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    image_path = str(mask_path).replace("/masks/", "/images/")
    image_path = image_path.replace("_masks.",".")
    orig = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    gray_orig = to_gray_u8(orig)

    if mask is None:
        return None, "Error reading"

    h, w = mask.shape[:2]
    final_dt = np.zeros((h, w), dtype=np.float32)

    component_count = 0

    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    if len(unique_ids) == 0:
        zeros = np.zeros((h, w), dtype=np.float32)
        stats = {"max_dist": 0.0, "num_instances": 0}
        return (zeros, zeros, zeros, stats), "Success"

    for uid in unique_ids:
        inst_mask = (mask == uid).astype(np.uint8)


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(inst_mask, cv2.MORPH_OPEN, kernel)

        # Respect 1 px black separators
        num_labels, labels = cv2.connectedComponents(closed, connectivity=4)

        for cc_id in range(1, num_labels):
            comp = (labels == cc_id).astype(np.uint8)
            area = int(comp.sum())
            if area < MIN_AREA:
                continue

            # brightness filter
            mean_brightness = component_mean_brightness(gray_orig, comp)
            if mean_brightness < MIN_MEAN_BRIGHTNESS:
                continue


            # elongation filter
            aspect_ratio = component_aspect_ratio(comp)
            if aspect_ratio < MIN_ASPECT_RATIO:
                continue

            component_count += 1

            # --- get all contours, including holes ---
            contours, hierarchy = cv2.findContours(
                comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue

            # smooth every contour independently
            smooth_contours = []
            for cnt in contours:
                pts = cnt.reshape(-1, 2).astype(np.float32)
                if len(pts) < 4:
                    continue
                pts = smooth_contour_moving_average(pts, SMOOTHING_WINDOW)
                smooth_contours.append(pts)

            if not smooth_contours:
                continue

            # supersampled working grid
            supersample = 2
            sh, sw = h * supersample, w * supersample
            dt_super = np.zeros((sh, sw), dtype=np.float32)

            ys, xs = np.where(comp > 0)
            if len(xs) == 0:
                continue

            # bbox in original pixels -> scaled bbox
            x0 = max(0, int(xs.min()) * supersample - 5 * supersample)
            x1 = min(sw, (int(xs.max()) + 1) * supersample + 5 * supersample)
            y0 = max(0, int(ys.min()) * supersample - 5 * supersample)
            y1 = min(sh, (int(ys.max()) + 1) * supersample + 5 * supersample)

            # upscale mask only for inside/outside membership
            comp_super = cv2.resize(
                comp, (sw, sh), interpolation=cv2.INTER_NEAREST
            )

            # pre-scale smoothed contours
            smooth_super = [
                (pts * supersample).reshape(-1, 1, 2).astype(np.float32)
                for pts in smooth_contours
            ]

            # determine which contours are outers vs holes
            outer_polys = []
            hole_polys = []
            if hierarchy is not None:
                hierarchy = hierarchy[0]  # OpenCV shape: (1, N, 4)
                for i, pts in enumerate(smooth_contours):
                    poly = (pts * supersample).reshape(-1, 1, 2).astype(np.float32)
                    parent = hierarchy[i][3]
                    if parent < 0:
                        outer_polys.append(poly)   # external contour
                    else:
                        hole_polys.append(poly)    # hole contour
            else:
                outer_polys = smooth_super

            for r in range(y0, y1):
                for c in range(x0, x1):
                    p = (float(c), float(r))

                    # inside any outer contour?
                    inside_outer = False
                    min_dist = np.inf

                    for poly in outer_polys:
                        d = cv2.pointPolygonTest(poly, p, True)
                        if d > 0:
                            inside_outer = True
                            min_dist = min(min_dist, d)

                    if not inside_outer:
                        continue

                    # reject holes
                    inside_hole = False
                    for poly in hole_polys:
                        d = cv2.pointPolygonTest(poly, p, True)
                        if d > 0:
                            inside_hole = True
                            break
                        min_dist = min(min_dist, abs(d))

                    if inside_hole:
                        continue

                    # also consider distance to hole boundaries
                    for poly in hole_polys:
                        d = abs(cv2.pointPolygonTest(poly, p, True))
                        min_dist = min(min_dist, d)

                    if np.isfinite(min_dist):
                        dt_super[r, c] = min_dist

            # downsample back to original resolution
            dt_final = cv2.resize(dt_super, (w, h), interpolation=cv2.INTER_AREA)
            dt_final = dt_final / supersample

            # hard clamp to original component
            #dt_final[comp == 0] = 0.0

            final_dt = np.maximum(final_dt, dt_final)

    flow_x, flow_y = get_gradients(final_dt)

    dt_out = final_dt.copy()
    dt_out[dt_out <= 0] = DIST_BG

    stats = {
        "max_dist": float(final_dt.max()),
        "num_instances": int(component_count)
    }

    return (dt_out, flow_x, flow_y, stats), "Success"

def main():
    """Run the module entry point and orchestrate command-line execution."""
    extensions = ['*.tif', '*.tiff', '*.png']
    files = []
    for ext in extensions:
        files.extend(list(INPUT_DIR.glob(ext)))
        
    print(f"Found {len(files)} mask files in {INPUT_DIR}")
    print(f"Processing with Smoothed Contours (window={SMOOTHING_WINDOW})...")
    
    for f in tqdm(files, desc="Processing"):
        result, msg = process_file(f)
        if result is None: continue
            
        dt, flow_x, flow_y, stats = result
        stem = f.stem
        
        # 1. Save Training Targets (Float32 TIF)
        tifffile.imwrite(OUTPUT_DIR / f"{stem}_dt.tif", dt)
        tifffile.imwrite(OUTPUT_DIR / f"{stem}_flow_x.tif", flow_x)
        tifffile.imwrite(OUTPUT_DIR / f"{stem}_flow_y.tif", flow_y)
        
        # 2. Save Visual Previews (PNG)
        # DT (Normalized 0-255)
        vis_dt = normalize_to_uint8(dt, mode='minmax')
        cv2.imwrite(str(PREVIEW_DIR / f"{stem}_dt.png"), vis_dt)
        
        # Grayscale Flow (Shifted 128, scaled softer)
        vis_fx = normalize_to_uint8(flow_x, mode='flow_grayscale')
        cv2.imwrite(str(PREVIEW_DIR / f"{stem}_flow_x.png"), vis_fx)
        
        vis_fy = normalize_to_uint8(flow_y, mode='flow_grayscale')
        cv2.imwrite(str(PREVIEW_DIR / f"{stem}_flow_y.png"), vis_fy)

        # HSV Flow (Best for checking smoothness)
        save_hsv_preview(flow_x, flow_y, PREVIEW_DIR / f"{stem}_flow_hsv.png")
        
        # 3. Save Stats
        with open(OUTPUT_DIR / f"{stem}_stats.json", 'w') as jf:
            json.dump(stats, jf, indent=4)

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
