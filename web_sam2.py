import os
import glob
import numpy as np
import cv2
import tifffile as tiff
import pandas as pd
import torch
import gradio as gr
from skimage.measure import label, regionprops
from dataclasses import dataclass

# --- SAM 2 IMPORTS ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: SAM 2 not found.")
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

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(os.path.join(cfg.output_dir, "masks"), exist_ok=True)

# ==============================
# STATE
# ==============================
files = sorted(glob.glob(os.path.join(cfg.image_dir, "*.tif*")))
current_idx = 0
points = []
labels = []
current_img = None
current_mask = None

print("Loading SAM 2...")
sam2_model = build_sam2(cfg.sam_model_cfg, cfg.sam_checkpoint, device=cfg.device)
predictor = SAM2ImagePredictor(sam2_model)

# ==============================
# HELPERS
# ==============================
def load_image(path):
    try:
        img = tiff.imread(path)
    except:
        return None
    
    # Fix dimensions
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[1] > 10:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2: img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 4: img = img[:,:,:3]

    # Normalize
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

def get_overlay(img, mask):
    overlay = img.copy()
    if mask is not None:
        # Paint Mask Red
        overlay[mask == 1] = [255, 0, 0]
        
        # Draw points for feedback
        for i, (px, py) in enumerate(points):
            color = (0, 255, 0) if labels[i] == 1 else (0, 0, 255) # Green for +, Red for -
            cv2.circle(overlay, (px, py), 5, color, -1)
            
    return overlay

# ==============================
# LOGIC
# ==============================
def start_session():
    global current_img, points, labels, current_mask
    if len(files) == 0: return None, "No files found."
    
    fpath = files[current_idx]
    current_img = load_image(fpath)
    points = []
    labels = []
    current_mask = np.zeros(current_img.shape[:2], dtype=bool)
    
    predictor.set_image(current_img)
    return current_img, f"Image {current_idx+1}/{len(files)}: {os.path.basename(fpath)}"

def add_point(img, mode, evt: gr.SelectData):
    global points, labels, current_mask
    
    x, y = evt.index[0], evt.index[1]
    
    # Check Mode: Positive (1) or Negative (0)
    lbl = 1 if mode == "Add Lesion (+)" else 0
    
    points.append([x, y])
    labels.append(lbl)
    
    pts_arr = np.array(points)
    lbls_arr = np.array(labels)
    
    # --- CRITICAL FIX ---
    # We set multimask_output=True to get 3 masks.
    # We take index [0] because that is usually the "Sub-Part" (Lesion),
    # whereas index [2] is the "Whole Object" (Leaf).
    masks, scores, _ = predictor.predict(
        point_coords=pts_arr,
        point_labels=lbls_arr,
        multimask_output=True 
    )
    
    # Heuristic: If we only have 1 point, pick the smallest mask (Index 0).
    # If we have multiple points (refining), we might trust score, 
    # but usually Index 0 is best for lesions.
    best_idx = 0 
    current_mask = masks[best_idx] > 0
    
    return get_overlay(current_img, current_mask)

def undo_point():
    global points, labels, current_mask
    if points:
        points.pop()
        labels.pop()
        
        if not points:
            current_mask = np.zeros(current_img.shape[:2], dtype=bool)
            return current_img
            
        pts_arr = np.array(points)
        lbls_arr = np.array(labels)
        
        masks, scores, _ = predictor.predict(
            point_coords=pts_arr,
            point_labels=lbls_arr,
            multimask_output=True 
        )
        current_mask = masks[0] > 0 # Stick to Index 0
        return get_overlay(current_img, current_mask)
    return current_img

def save_and_next():
    global current_idx
    if current_idx >= len(files): return None, "Done!"
    
    fpath = files[current_idx]
    fname = os.path.basename(fpath)
    
    # Save Mask
    mask_path = os.path.join(cfg.output_dir, "masks", fname + ".png")
    cv2.imwrite(mask_path, (current_mask * 255).astype(np.uint8))
    
    # Save Metrics
    labeled = label(current_mask)
    areas = [p.area for p in regionprops(labeled)]
    
    csv_path = os.path.join(cfg.output_dir, "manual_results.csv")
    new_row = pd.DataFrame([{
        "file": fname,
        "lesion_count": len(areas),
        "total_lesion_area": np.sum(current_mask),
        "avg_lesion_size": np.mean(areas) if areas else 0
    }])
    
    if os.path.exists(csv_path):
        new_row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(csv_path, mode='w', header=True, index=False)
        
    current_idx += 1
    if current_idx >= len(files): return None, "All images processed!"
    
    return start_session()

# ==============================
# UI
# ==============================
with gr.Blocks(title="SAM 2 Phenotyper") as demo:
    gr.Markdown("# SAM 2 Interactive Phenotyper")
    
    with gr.Row():
        info_box = gr.Textbox(label="Status", interactive=False)
        
    with gr.Row():
        mode_radio = gr.Radio(
            ["Add Lesion (+)", "Remove Area (-)"], 
            value="Add Lesion (+)", 
            label="Click Mode"
        )
        
    with gr.Row():
        img_display = gr.Image(label="Leaf Image", interactive=False)
    
    with gr.Row():
        btn_undo = gr.Button("Undo Last Click")
        btn_save = gr.Button("Save & Next Image", variant="primary")
        
    demo.load(start_session, outputs=[img_display, info_box])
    
    # Pass 'mode_radio' as an input to add_point
    img_display.select(add_point, inputs=[img_display, mode_radio], outputs=[img_display])
    
    btn_undo.click(undo_point, outputs=[img_display])
    btn_save.click(save_and_next, outputs=[img_display, info_box])

print("Launching Web UI...")
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)