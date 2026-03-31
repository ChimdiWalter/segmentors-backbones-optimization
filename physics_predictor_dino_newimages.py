#!/usr/bin/env python3
# run_fast_physics_dino.py
# Uses the DINOv2 Neural Net to pick parameters instantly.

import os
import cv2
import torch
import numpy as np
import tifffile as tiff
from torchvision import transforms
from torch import nn
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# ================= CONFIGURATION =================
MODEL_PATH = "dino_param_predictor.pth" 
INPUT_DIR = "/deltos/c/maize/image_processing/data/leaf/22r/aleph/18.7"
OUTPUT_DIR = "fast_physics_results_dino"

# Colors (8-bit BGR for overlay)
COLOR_INNER = (255, 100, 255) # Pink

# ================= 1. MODEL SETUP (DINO) =================
class DINOv2Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load DINOv2 Small
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 7) # 7 Params
        )

    def forward(self, x):
        return self.head(self.backbone(x))

def load_predictor(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv2Regressor()
    
    # Load weights (strict=False might be needed if hub changed paths, but usually fine)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model, device

def predict_params(model, device, img_bgr):
    # DINO requires image multiples of 14. We use 224x224.
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

    params = {
        "mu": float(preds[0]),
        "lambda": float(preds[1]),
        "diffusion_rate": float(preds[2]),
        "alpha": float(preds[3]),
        "beta": float(preds[4]),
        "gamma": float(preds[5]),
        "energy_threshold": int(preds[6] * 100.0)
    }
    return params

# ================= 2. PHYSICS ENGINE =================
def compute_gradients(u8):
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, mag

def build_seed(gx, gy):
    sgx = cv2.GaussianBlur(gx, (0,0), 1.5, 1.5)
    sgy = cv2.GaussianBlur(gy, (0,0), 1.5, 1.5)
    smag = np.sqrt(sgx**2 + sgy**2)
    return sgx, sgy, smag

def run_pde(seed_mag, img_mag, igx, igy, sgx, sgy, p):
    diffused = seed_mag.astype(np.float64).copy()
    edge_mask = (img_mag > p["energy_threshold"]).astype(np.float64)
    for _ in range(20):
        lap = laplace(diffused)
        gy, gx = np.gradient(diffused)
        div = gx + gy
        ftx = igx - sgx
        fty = igy - sgy
        ftm = np.sqrt(ftx**2 + fty**2)
        diffused += p["diffusion_rate"] * (p["mu"]*lap + (p["lambda"]+p["mu"])*div + edge_mask*(ftm - diffused))
    
    diffused = np.maximum(diffused, 0)
    mx = float(diffused.max())
    if mx < 1e-8: return np.zeros_like(diffused, dtype=np.uint8)
    return np.uint8(255.0 * diffused / mx)

def run_snake(img_u8, energy, p):
    labeled = label(energy > p["energy_threshold"])
    out_mask = np.zeros(img_u8.shape, dtype=bool)
    
    for rgn in regionprops(labeled):
        if rgn.area < 16: continue 
        minr, minc, maxr, maxc = rgn.bbox
        if maxr-minr < 5 or maxc-minc < 5: continue
        
        crop_u8 = img_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(crop_u8.astype(float)/255.0, 3)
        s = np.linspace(0, 2*np.pi, 200, endpoint=False)
        rr = (maxr-minr)/2.0
        cc = (maxc-minc)/2.0
        init = np.vstack([rr*np.sin(s)+rr, cc*np.cos(s)+cc]).T
        try:
            snake = active_contour(crop_f, init, alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"], max_num_iter=60)
            si = np.round(snake).astype(int)
            si[:,0] = np.clip(si[:,0], 0, maxr-minr-1)
            si[:,1] = np.clip(si[:,1], 0, maxc-minc-1)
            rr_fill, cc_fill = polygon(si[:,0], si[:,1], crop_u8.shape)
            out_mask[minr:maxr, minc:maxc][rr_fill, cc_fill] = True
        except: pass
    return out_mask

def _to_uint8(img):
    if img.dtype == np.uint16: return cv2.convertScaleAbs(img, alpha=255.0/65535.0)
    return img

def fused_channel(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    candidates = [bgr[...,1], gray, lab[...,1], lab[...,2]]
    fused = None
    for c in candidates:
        if c.dtype!=np.uint8: c = _to_uint8(c)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        c = clahe.apply(c)
        fused = c if fused is None else np.maximum(fused, c)
    return fused

# ================= 3. MAIN LOOP =================
def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Train it first.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading DINOv2 Predictor...")
    model, device = load_predictor(MODEL_PATH)
    
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))])
    print(f"Running Fast Physics (DINO) on {len(files)} images...")
    
    for i, fname in enumerate(files):
        path = os.path.join(INPUT_DIR, fname)
        raw = cv2.imread(path, -1) 
        bgr = cv2.imread(path, 1) 
        if bgr is None: continue
        
        # Predict
        params = predict_params(model, device, bgr)
        if i % 10 == 0:
            print(f"[{i+1}/{len(files)}] {fname} -> Thresh:{params['energy_threshold']} Beta:{params['beta']:.3f}")
        
        # Physics
        work_u8 = fused_channel(bgr)
        igx, igy, imag = compute_gradients(work_u8)
        sgx, sgy, smag = build_seed(igx, igy)
        energy = run_pde(smag, imag, igx, igy, sgx, sgy, params)
        mask_bool = run_snake(work_u8, energy, params)
        
        # Save
        overlay = _to_uint8(raw)
        if overlay.ndim == 2: overlay = cv2.merge([overlay, overlay, overlay])
        elif overlay.shape[2] == 1: overlay = cv2.merge([overlay, overlay, overlay])
        
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, COLOR_INNER, 2)
        
        mask16 = (mask_bool.astype(np.uint16)) * 65535
        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_dino_overlay.tif"), overlay)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_dino_mask.tif"), mask16)

    print("Done! Results in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()