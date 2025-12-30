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
    print("Error: SAM 2 not found. Please install it.")
    exit(1)

# ==============================
# CONFIGURATION
# ==============================
@dataclass
class Config:
    image_dir: str = "./leaves"              # Folder with .tif images
    output_dir: str = "./results_bbox"       # Output folder
    sam_checkpoint: str = "sam2_hiera_large.pt"
    sam_model_cfg: str = "sam2_hiera_l.yaml"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(os.path.join(cfg.output_dir, "masks"), exist_ok=True)

# ==============================
# GLOBAL STATE
# ==============================
files = sorted(glob.glob(os.path.join(cfg.image_dir, "*.tif*")))
current_idx = 0
current_raw_img = None  # Clean image for prediction
current_mask_global = None # Accumulates all lesions for this image

print(f"Loading SAM 2 on {cfg.device}...")
sam2_model = build_sam2(cfg.sam_model_cfg, cfg.sam_checkpoint, device=cfg.device)
predictor = SAM2ImagePredictor(sam2_model)

# ==============================
# HELPER FUNCTIONS
# ==============================
def load_image(path):
    try:
        img = tiff.imread(path)
    except:
        return None
    
    # Handle Image Dimensions (16-bit to 8-bit RGB)
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[1] > 10:
        img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
    if img.ndim == 2: 
        img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] == 1: 
        img = np.repeat(img, 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] > 3: 
        img = img[:,:,:3] # Drop Alpha

    # Normalize float to uint8
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

def apply_overlay(img, mask):
    """Draws the Red Mask on top of the image."""
    overlay = img.copy()
    if mask is not None:
        # Create Red channel overlay
        red_mask = np.zeros_like(img)
        red_mask[:, :, 0] = 255 # Red (in RGB)
        
        # Blend where mask is True
        alpha = 0.4
        overlay[mask] = cv2.addWeighted(overlay[mask], 1-alpha, red_mask[mask], alpha, 0)
    return overlay

# ==============================
# CORE LOGIC
# ==============================
def start_session():
    global current_raw_img, current_mask_global
    if len(files) == 0: return None, "No files found."
    
    fpath = files[current_idx]
    current_raw_img = load_image(fpath)
    current_mask_global = np.zeros(current_raw_img.shape[:2], dtype=bool)
    
    # Initialize SAM predictor once per image
    predictor.set_image(current_raw_img)
    
    # Return image to the Editor. 
    # ImageEditor accepts just the numpy array to set the background.
    return current_raw_img, f"Image {current_idx+1}/{len(files)}: {os.path.basename(fpath)}"

def process_box_draw(editor_data):
    """
    Called when user draws a box. 
    'editor_data' is a dict from gr.ImageEditor
    """
    global current_mask_global
    
    if editor_data is None: return current_raw_img
    
    # GRADIO 4.X FIX: Data comes as {'background': ..., 'layers': [rgba_array], 'composite': ...}
    # We look at layers[0] which contains the drawing stroke.
    layers = editor_data.get("layers", [])
    if not layers: return apply_overlay(current_raw_img, current_mask_global)
    
    draw_layer = layers[0] # RGBA layer of the drawing
    
    # 2. Find Bounding Box of the drawing
    if draw_layer is None or np.max(draw_layer) == 0:
        return apply_overlay(current_raw_img, current_mask_global)

    ys, xs = np.where(draw_layer[:, :, -1] > 0) # Alpha channel check
    if len(xs) == 0: return apply_overlay(current_raw_img, current_mask_global)
    
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    
    # 3. Prompt SAM with this Box
    box_prompt = np.array([x_min, y_min, x_max, y_max])
    
    try:
        masks, scores, _ = predictor.predict(
            box=box_prompt,
            multimask_output=True 
        )
        # 4. Pick best mask (usually index 0 for lesions)
        new_lesion_mask = masks[0] > 0
        
        # 5. Add to Global Accumulator (Union)
        current_mask_global = np.logical_or(current_mask_global, new_lesion_mask)
    except Exception as e:
        print(f"SAM Prediction error: {e}")
    
    # 6. Return Overlay to the Editor
    # Note: We return the overlaid image as the new background
    return apply_overlay(current_raw_img, current_mask_global)

def clear_mask():
    global current_mask_global
    if current_raw_img is None: return None
    current_mask_global = np.zeros(current_raw_img.shape[:2], dtype=bool)
    return current_raw_img # Return clean image

def save_and_next():
    global current_idx
    if current_idx >= len(files): return None, "Done!"
    
    fpath = files[current_idx]
    fname = os.path.basename(fpath)
    
    # Save Visual Overlay
    vis_path = os.path.join(cfg.output_dir, fname + "_vis.jpg")
    final_vis = apply_overlay(current_raw_img, current_mask_global)
    cv2.imwrite(vis_path, cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))
    
    # Save Binary Mask
    mask_path = os.path.join(cfg.output_dir, "masks", fname + ".png")
    cv2.imwrite(mask_path, (current_mask_global * 255).astype(np.uint8))
    
    # Calculate Stats
    labeled = label(current_mask_global)
    props = regionprops(labeled)
    count = len(props)
    total_area = np.sum(current_mask_global)
    avg_size = np.mean([p.area for p in props]) if props else 0
    
    # Save CSV
    csv_path = os.path.join(cfg.output_dir, "lesion_stats.csv")
    df = pd.DataFrame([{
        "filename": fname,
        "lesion_count": count,
        "total_lesion_area_px": total_area,
        "average_lesion_size_px": avg_size
    }])
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
        
    print(f"Saved {fname} -> Count: {count}")
    
    # Next Image
    current_idx += 1
    if current_idx >= len(files): return None, "All images processed!"
    
    return start_session()

# ==============================
# UI LAYOUT
# ==============================
with gr.Blocks(title="SAM 2 BBox Phenotyper") as demo:
    gr.Markdown("## 🍂 SAM 2 Lesion Segmentor (Box Mode)")
    gr.Markdown("1. **Draw a yellow box** around a lesion using the Brush tool. \n2. Wait a split second for SAM to snap to it. \n3. Click 'Save & Next Image'.")
    
    with gr.Row():
        info_box = gr.Textbox(label="Status", interactive=False)
        
    with gr.Row():
        # FIX: Replaced gr.Image with gr.ImageEditor
        # type="numpy" ensures we get arrays, not file paths
        img_input = gr.ImageEditor(
            label="Draw Box Here", 
            type="numpy", 
            interactive=True,
            brush=gr.Brush(colors=["#FFFF00"], default_size=10, color_mode="fixed"), # Force yellow brush
            transforms=[] # Disable crop/rotate tools to keep it simple
        )
    
    with gr.Row():
        btn_clear = gr.Button("Clear / Reset", variant="secondary")
        btn_save = gr.Button("Save & Next Image", variant="primary")
        
    # --- Events ---
    demo.load(start_session, outputs=[img_input, info_box])
    
    # Use 'change' to detect drawing updates
    img_input.change(process_box_draw, inputs=[img_input], outputs=[img_input])
    
    btn_clear.click(clear_mask, outputs=[img_input])
    btn_save.click(save_and_next, outputs=[img_input, info_box])

print("Launching... Access via browser (http://localhost:7860)")
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)