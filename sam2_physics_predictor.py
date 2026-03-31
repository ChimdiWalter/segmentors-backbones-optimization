import os
import cv2
import torch
import numpy as np
import tifffile as tiff
from torch import nn
from torchvision import transforms
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# --- SAM 2 IMPORTS ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: SAM 2 not found.")
    exit()

# ================= CONFIGURATION =================
# 1. DINO Model (Your Best Model)
DINO_MODEL_PATH = "dino_param_predictor.pth"

# 2. SAM 2 (The Refiner)
SAM_CHECKPOINT = "sam2_hiera_large.pt"
SAM_CONFIG = "sam2_hiera_l.yaml"

# 3. Data
INPUT_DIR = "/deltos/c/maize/image_processing/data/leaf/22r/aleph/18.7"
OUTPUT_DIR = "final_dino_sam2_results"

# Visualization Color (Pink)
COLOR_PINK = (255, 100, 255)

# ================= 1. DINO MODEL DEFINITION =================
class DINOv2Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

def load_dino(path, device):
    model = DINOv2Regressor()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_params(model, device, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = tf(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(input_tensor).cpu().numpy()[0]
    
    return {
        "mu": float(preds[0]), "lambda": float(preds[1]), "diffusion_rate": float(preds[2]),
        "alpha": float(preds[3]), "beta": float(preds[4]), "gamma": float(preds[5]),
        "energy_threshold": int(preds[6] * 100.0)
    }

# ================= 2. PHYSICS ENGINE (Rough Spot Finder) =================
def run_physics_mask(img_bgr, params):
    # 1. Fuse Channels
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    candidates = [img_bgr[...,1], gray, lab[...,1], lab[...,2]]
    work_u8 = None
    for c in candidates:
        c = cv2.equalizeHist(c)
        work_u8 = c if work_u8 is None else np.maximum(work_u8, c)
        
    # 2. Gradients
    gx = cv2.Sobel(work_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(work_u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    # 3. PDE
    diffused = np.sqrt(cv2.GaussianBlur(gx,(0,0),1.5)**2 + cv2.GaussianBlur(gy,(0,0),1.5)**2)
    edge_mask = (mag > params["energy_threshold"]).astype(float)
    
    for _ in range(15): # Fast PDE
        lap = laplace(diffused)
        dy, dx = np.gradient(diffused)
        diffused += params["diffusion_rate"] * (params["mu"]*lap + (params["lambda"]+params["mu"])*(dx+dy) + edge_mask*(mag - diffused))
    
    energy = np.uint8(np.clip(diffused, 0, 255))
    
    # 4. Snake / Threshold
    labeled = label(energy > params["energy_threshold"])
    return labeled # This is our Rough Mask

# ================= 3. MAIN LOOP =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. Loading DINO (The Brain)...")
    dino_model = load_dino(DINO_MODEL_PATH, device)

    print("2. Loading SAM 2 (The Sniper)...")
    sam2_model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))])
    print(f"Processing {len(files)} images...")

    for i, fname in enumerate(files):
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)
        if img is None: continue

        # --- A. Predict Params ---
        params = predict_params(dino_model, device, img)
        
        # --- B. Run Physics to get Rough Spots ---
        # We don't need a perfect snake here, just blobs where lesions are
        labeled_map = run_physics_mask(img, params)
        
        # --- C. Extract Points for SAM ---
        points = []
        labels = []
        for region in regionprops(labeled_map):
            if region.area < 10: continue # Ignore noise
            yc, xc = region.centroid
            points.append([xc, yc])
            labels.append(1)

        # --- D. SAM 2 Refinement ---
        final_mask = np.zeros(img.shape[:2], dtype=bool)
        
        if len(points) > 0:
            predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Predict each spot
            for pt in points:
                masks, _, _ = predictor.predict(
                    point_coords=np.array([pt]),
                    point_labels=np.array([1]),
                    multimask_output=True
                )
                # Take index 0 (Lesion scale)
                final_mask = np.logical_or(final_mask, masks[0])

        # --- E. Save ---
        mask_u8 = (final_mask.astype(np.uint8)) * 255
        
        overlay = img.copy()
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, COLOR_PINK, 2)
        
        # Draw points
        for pt in points:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_final_vis.jpg"), overlay)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_final_mask.png"), mask_u8)
        
        if i % 10 == 0: print(f"Processed {i}/{len(files)}")

    print(f"Done! Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()