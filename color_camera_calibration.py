import cv2
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
INPUT_ROOT = 'track1-1'
OUTPUT_ROOT = 'cleaned'

# --- NEW RECALIBRATED DATA (224x224) ---
MTX = np.array([[103.80088948,   0.0,         112.99926554],
                [  0.0,         133.49980795,  99.02992825],
                [  0.0,           0.0,           1.0        ]])

DIST = np.array([[-0.2964748, 0.07032658, 0.00985859, -0.00041996, -0.00725899]])

def apply_corrections(img):
    if img is None:
        return None
    
    # 1. GEOMETRIC CALIBRATION
    h, w = img.shape[:2]
    # alpha=0: Zoom in to remove all black edges
    # alpha=1: Keep all pixels (results in black curved edges)
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 0, (w, h))
    undistorted = cv2.undistort(img, MTX, DIST, None, new_camera_mtx)

    # 2. COLOR CORRECTION (Gray World)
    result = undistorted.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(result, axis=(0, 1))
    avg_gray = (avg_b + avg_g + avg_r) / 3

    if all(v > 0 for v in [avg_b, avg_g, avg_r]):
        result[:, :, 0] *= (avg_gray / avg_b)
        result[:, :, 1] *= (avg_gray / avg_g)
        result[:, :, 2] *= (avg_gray / avg_r)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def process_all_directories():
    input_base = Path(INPUT_ROOT)
    output_base = Path(OUTPUT_ROOT)

    image_files = list(input_base.rglob('*.jpg'))
    
    if not image_files:
        print(f"No images found in {INPUT_ROOT}. Check your folder names.")
        return

    print(f"Applying new calibration to {len(image_files)} images...")

    for img_p in image_files:
        relative_path = img_p.relative_to(input_base)
        target_path = output_base / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_p))
        corrected_img = apply_corrections(img)
        
        if corrected_img is not None:
            cv2.imwrite(str(target_path), corrected_img)

    print(f"\nDone! Your 224x224 dataset is cleaned in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    process_all_directories()