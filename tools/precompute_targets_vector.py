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
ROOT_DIR = Path("/home/vk/workspace/Training/3D_MT/")
INPUT_DIR = ROOT_DIR / "dataset/masks"
OUTPUT_DIR = ROOT_DIR / "dataset/targets2/"
PREVIEW_DIR = OUTPUT_DIR / "previews"

# Contour Smoothing Settings
SMOOTHING_WINDOW = 5  # odd integer, circular moving average

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

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

def process_instance_simplified(mask_shape, contour, supersample=2):
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
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None: return None, "Error reading"

    h, w = mask.shape[:2]
    final_dt = np.zeros((h, w), dtype=np.float32)
    
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    if len(unique_ids) == 0:
        zeros = np.zeros((h, w), dtype=np.float32)
        stats = {"max_dist": 0.0, "num_instances": 0}
        return (zeros, zeros, zeros, stats), "Success"

    for uid in unique_ids:
        inst_mask = (mask == uid).astype(np.uint8)
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 10: continue
            dt_inst = process_instance_simplified((h, w), cnt)
            final_dt = np.maximum(final_dt, dt_inst)

    # Compute RAW Gradients (Un-normalized float32)
    flow_x, flow_y = get_gradients(final_dt)
    
    stats = {
        "max_dist": float(final_dt.max()),
        "num_instances": int(len(unique_ids))
    }

    return (final_dt, flow_x, flow_y, stats), "Success"

def main():
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
