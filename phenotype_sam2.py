#!/usr/bin/env python
import os
import glob
import cv2
import numpy as np
import tifffile as tiff
import pandas as pd
import torch
from skimage.measure import label, regionprops
from dataclasses import dataclass

# --- SAM 2 IMPORTS ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: SAM 2 not found. Ensure 'sam2' folder is renamed to 'sam2_repo'.")
    exit(1)

# ==============================
# CONFIG
# ==============================
@dataclass
class Config:
    image_dir: str = "./leaves"
    output_dir: str = "./results_manual"
    sam_checkpoint: str = "sam2_hiera_large.pt"
    sam_model_cfg: str = "sam2_hiera_l.yaml"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    display_size: int = 1000  # Resize window for easier clicking (doesn't affect result)

# ==============================
# ROBUST LOADER
# ==============================
def load_image(path):
    try:
        img = tiff.imread(path)
    except:
        return None
    
    if img is None: return None
    
    # Fix Channel-First
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[1] > 10:
        img = np.transpose(img, (1, 2, 0))
    # Fix Grayscale
    if img.ndim == 2: img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 4: img = img[:,:,:3]

    # Normalize
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

# ==============================
# INTERACTIVE CLASS
# ==============================
class InteractiveSegmenter:
    def __init__(self, cfg):
        self.cfg = cfg
        print("Loading SAM 2...")
        sam2_model = build_sam2(cfg.sam_model_cfg, cfg.sam_checkpoint, device=cfg.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        self.points = []
        self.labels = []
        self.mask = None
        self.img_display = None
        self.scale = 1.0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left Click = Lesion (Positive)
            real_x, real_y = int(x / self.scale), int(y / self.scale)
            self.points.append([real_x, real_y])
            self.labels.append(1)
            cv2.circle(self.img_display, (x, y), 5, (0, 255, 0), -1) # Green
            cv2.imshow("Tagging", self.img_display)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right Click = Healthy/Background (Negative)
            real_x, real_y = int(x / self.scale), int(y / self.scale)
            self.points.append([real_x, real_y])
            self.labels.append(0)
            cv2.circle(self.img_display, (x, y), 5, (0, 0, 255), -1) # Red
            cv2.imshow("Tagging", self.img_display)

    def run(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.output_dir, "masks"), exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(self.cfg.image_dir, "*.tif*")))
        results = []
        
        print("\n=== CONTROLS ===")
        print("LEFT CLICK  : Add Lesion Point")
        print("RIGHT CLICK : Remove Area (Healthy)")
        print("SPACEBAR    : Update/Show Mask")
        print("S KEY       : Save & Next Image")
        print("ESC         : Quit")
        print("================\n")

        for idx, fpath in enumerate(files):
            print(f"[{idx+1}/{len(files)}] {os.path.basename(fpath)}")
            
            original_img = load_image(fpath)
            if original_img is None: continue
            
            # Setup SAM
            self.predictor.set_image(original_img)
            self.points = []
            self.labels = []
            self.mask = np.zeros(original_img.shape[:2], dtype=bool)
            
            # Resize for display only
            h, w = original_img.shape[:2]
            self.scale = self.cfg.display_size / max(h, w)
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            self.img_display = cv2.resize(original_img, (new_w, new_h))
            
            cv2.namedWindow("Tagging")
            cv2.setMouseCallback("Tagging", self.mouse_callback)
            
            while True:
                cv2.imshow("Tagging", self.img_display)
                key = cv2.waitKey(1) & 0xFF
                
                # UPDATE MASK (Spacebar)
                if key == 32: 
                    if len(self.points) > 0:
                        pts = np.array(self.points)
                        lbls = np.array(self.labels)
                        
                        masks, scores, _ = self.predictor.predict(
                            point_coords=pts,
                            point_labels=lbls,
                            multimask_output=False
                        )
                        self.mask = masks[0] > 0
                        
                        # Show overlay
                        overlay = self.img_display.copy()
                        # Resize mask to display size for visualization
                        mask_disp = cv2.resize(self.mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                        overlay[mask_disp == 1] = [0, 0, 255] # Red overlay
                        cv2.imshow("Tagging", overlay)

                # SAVE & NEXT (S)
                elif key == ord('s'):
                    # Save Mask
                    fname = os.path.basename(fpath)
                    mask_path = os.path.join(self.cfg.output_dir, "masks", fname + ".png")
                    cv2.imwrite(mask_path, (self.mask * 255).astype(np.uint8))
                    
                    # Calculate Metrics
                    labeled = label(self.mask)
                    areas = [p.area for p in regionprops(labeled)]
                    
                    results.append({
                        "file": fname,
                        "lesion_count": len(areas),
                        "total_lesion_area": np.sum(self.mask),
                        "avg_lesion_size": np.mean(areas) if areas else 0
                    })
                    print("Saved.")
                    break
                
                # QUIT (Esc)
                elif key == 27:
                    cv2.destroyAllWindows()
                    pd.DataFrame(results).to_csv(os.path.join(self.cfg.output_dir, "manual_results.csv"), index=False)
                    return

        # End loop
        cv2.destroyAllWindows()
        pd.DataFrame(results).to_csv(os.path.join(self.cfg.output_dir, "manual_results.csv"), index=False)
        print("Done.")

if __name__ == "__main__":
    cfg = Config()
    app = InteractiveSegmenter(cfg)
    app.run()