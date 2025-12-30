import cv2
import numpy as np
import pandas as pd
import os
from skimage.measure import label, regionprops

# ================= CONFIG =================
IMG_DIR = "/deltos/c/maize/image_processing/data/leaf/22r/aleph/18.7"
MASK_DIR = "unet_direct_results" # Where you saved the UNet masks
OUTPUT_CSV = "unet_biological_analysis.csv"

# ================= SCORING FUNCTIONS =================
# (These are reused from your optimizer script)
def grad_alignment_score(mask_u8, img_gray):
    edges = cv2.Canny(mask_u8, 50, 150)
    
    # Calculate Gradient Magnitude of the image
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag_u8 = np.uint8(255 * grad_mag / (grad_mag.max() + 1e-8))
    
    edge_vals = grad_mag_u8[edges > 0]
    if edge_vals.size == 0: return 0.0
    
    # Ratio of Edge Brightness vs Global Brightness
    return float(edge_vals.mean()) / (float(grad_mag_u8.mean()) + 1e-6)

def color_distance_score(mask_bool, img_bgr):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    
    fg_mask = mask_bool
    bg_mask = ~mask_bool
    
    if np.sum(fg_mask) < 10: return 0.0
    
    # Get A and B channels (Color)
    lesion_colors = img_lab[..., 1:3][fg_mask]
    leaf_colors   = img_lab[..., 1:3][bg_mask]
    
    lesion_mean = np.mean(lesion_colors, axis=0)
    leaf_mean   = np.mean(leaf_colors, axis=0)
    
    # Euclidean Distance
    dist = np.linalg.norm(lesion_mean - leaf_mean)
    return float(dist)

# ================= MAIN =================
data = []
files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith("_unet_mask.png")])

print(f"Analyzing {len(files)} predictions...")

for f in files:
    # Filename matching (assuming "DSC_123_unet_mask.png" -> "DSC_123.tif")
    # You might need to adjust this string replacement depending on your filenames
    original_name = f.replace("_unet_mask.png", "") # Try exact match first
    
    # Try to find original
    img_path = os.path.join(IMG_DIR, original_name)
    if not os.path.exists(img_path):
        # Try appending extensions if missing
        for ext in ['.tif', '.tiff', '.jpg', '.png']:
            if os.path.exists(img_path + ext):
                img_path = img_path + ext
                break
    
    if not os.path.exists(img_path):
        print(f"Skipping {f}, cannot find original image.")
        continue

    # Load
    img = cv2.imread(img_path)
    mask = cv2.imread(os.path.join(MASK_DIR, f), cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None: continue
    
    # Resize mask if needed (UNet output might be 256x256, Image might be huge)
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask > 127
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- CALCULATE METRICS ---
    g_score = grad_alignment_score(mask, img_gray)
    c_score = color_distance_score(mask_bool, img)
    
    # Morphology
    labeled = label(mask_bool)
    props = regionprops(labeled)
    n_spots = len(props)
    total_area = np.sum(mask_bool)
    
    # Save
    data.append({
        "filename": original_name,
        "gradient_score": g_score,  # How sharp are the edges?
        "color_score": c_score,     # How brown is it compared to the leaf?
        "lesion_count": n_spots,
        "total_area_px": total_area
    })

# Save CSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved analysis to {OUTPUT_CSV}")