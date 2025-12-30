import os
import cv2
import torch
import numpy as np
import tifffile as tiff
from torch import nn
from torchvision import models, transforms
from skimage.measure import label, regionprops
from scipy.ndimage import laplace

# --- SAM 2 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: SAM 2 not found.")
    exit()

# ================= CONFIGURATION =================
RESNET_MODEL_PATH = "param_predictor_resnet.pth"
SAM_CHECKPOINT = "sam2_hiera_large.pt"
SAM_CONFIG = "sam2_hiera_l.yaml"

INPUT_DIR = "/deltos/c/maize/image_processing/data/leaf/22r/aleph/18.7"
OUTPUT_DIR = "sam2_high_sensitivity_results"

COLOR_PINK = (255, 100, 255)

# ================= 1. MODEL =================
def get_resnet_model():
    model = models.resnet18(weights=None) 
    model.fc = nn.Linear(model.fc.in_features, 7)
    return model

def load_resnet(path, device):
    model = get_resnet_model()
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

def run_physics_mask(img_bgr, params):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    candidates = [img_bgr[...,1], gray, lab[...,1], lab[...,2]]
    work_u8 = None
    for c in candidates:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        c = clahe.apply(c if c.dtype == np.uint8 else (c/256).astype(np.uint8))
        work_u8 = c if work_u8 is None else np.maximum(work_u8, c)
        
    gx = cv2.Sobel(work_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(work_u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    diffused = np.sqrt(cv2.GaussianBlur(gx,(0,0),1.5)**2 + cv2.GaussianBlur(gy,(0,0),1.5)**2)
    edge_mask = (mag > params["energy_threshold"]).astype(float)
    
    for _ in range(15): 
        lap = laplace(diffused)
        dy, dx = np.gradient(diffused)
        diffused += params["diffusion_rate"] * (params["mu"]*lap + (params["lambda"]+params["mu"])*(dx+dy) + edge_mask*(mag - diffused))
    
    energy = np.uint8(np.clip(diffused, 0, 255))
    return label(energy > params["energy_threshold"])

# ================= 2. MAIN =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Models...")
    resnet_model = load_resnet(RESNET_MODEL_PATH, device)
    sam2_model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))])
    print(f"Processing {len(files)} images (High Sensitivity Mode)...")

    for i, fname in enumerate(files):
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)
        if img is None: continue

        # A. Predict & Physics
        params = predict_params(resnet_model, device, img)
        labeled_map = run_physics_mask(img, params)
        
        # B. Get Points
        points = []
        for region in regionprops(labeled_map):
            if region.area < 15: continue
            yc, xc = region.centroid
            points.append([xc, yc])

        # C. SAM 2 Refinement (The Fix)
        final_mask = np.zeros(img.shape[:2], dtype=bool)
        
        if len(points) > 0:
            predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for pt in points:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([pt]),
                    point_labels=np.array([1]),
                    multimask_output=True
                )
                
                # --- THE FIX ---
                # Check Mask 0 (Strict)
                mask_0 = masks[0]
                mask_1 = masks[1]
                
                # If Mask 0 is empty (SAM rejected it) OR Score 1 is much better than Score 0
                if np.sum(mask_0) < 10 or (scores[1] > scores[0] + 0.1):
                    # Use Mask 1 (Medium/Loose)
                    final_mask = np.logical_or(final_mask, mask_1)
                else:
                    # Use Mask 0 (Strict/Precise)
                    final_mask = np.logical_or(final_mask, mask_0)

        # D. Save
        mask_u8 = (final_mask.astype(np.uint8)) * 255
        overlay = img.copy()
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, COLOR_PINK, 2)

        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_sens_vis.jpg"), overlay)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_sens_mask.png"), mask_u8)
        
        if i % 10 == 0: 
            print(f"[{i+1}/{len(files)}] {fname}")

    print(f"Done! Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()