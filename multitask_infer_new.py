#!/usr/bin/env python3
"""
FIXED Inference script for MultiTaskLesionNet.
Fixes: 
1. Exact normalization match with training.
2. Removes complex 16-bit logic that might confuse the Neural Net.
3. Ensures output mask alignment.
"""

import os
import json
import argparse
import numpy as np
import cv2
import tifffile as tiff
import torch
import torch.nn.functional as F
import pandas as pd
from multitask_lesion_model import MultiTaskLesionNet 

# Allowed image extensions
ALLOWED_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
PINK_16 = (65535, 5140, 37779) # 16-bit Pink
PARAM_COLS = ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "energy_threshold"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_and_preprocess_image(img_path: str, size: int = 256):
    """
    Standardized loading to match training exactly.
    """
    # 1. Load as standard color image (drops alpha/16-bit complexity for the model)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read: {img_path}")

    # 2. Keep a raw copy for overlay (if it's 16-bit TIF, load separately)
    # We try to load high-res raw for the overlay, but use 8-bit for the neural net
    try:
        raw_overlay = tiff.imread(img_path)
    except:
        raw_overlay = img_bgr # Fallback

    # 3. Preprocess for Model (EXACTLY like training)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    
    # Normalize 0-1
    img_tensor = img_resized.astype(np.float32) / 255.0
    
    # HWC -> CHW
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    
    return raw_overlay, img_tensor

def _prepare_overlay16(raw):
    """Ensure overlay is 3-channel 16-bit for drawing."""
    # Normalize any input to 16-bit RGB range
    if raw.dtype == np.uint8:
        raw = (raw.astype(np.uint16) * 257)
    
    if raw.ndim == 2:
        return cv2.merge([raw, raw, raw])
    elif raw.ndim == 3 and raw.shape[2] == 3:
        return raw
    elif raw.ndim == 3 and raw.shape[2] == 1:
        g = raw[..., 0]
        return cv2.merge([g, g, g])
    return raw

def load_trained_model(ckpt_path, device):
    print(f"[Model] Loading {ckpt_path}...")
    # FIX: Allow numpy arrays in the checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = MultiTaskLesionNet(in_channels=3, num_params=len(PARAM_COLS))
    
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        stats = (ckpt.get("param_means"), ckpt.get("param_stds"))
    else:
        model.load_state_dict(ckpt)
        stats = (None, None) # Params will be un-normalized
        
    model.to(device)
    model.eval()
    return model, stats

def infer(ckpt_path, input_dir, output_dir, img_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(output_dir)
    
    model, (p_mean, p_std) = load_trained_model(ckpt_path, device)
    
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(ALLOWED_EXT)])
    print(f"[Infer] Processing {len(files)} images...")
    
    for fname in files:
        path = os.path.join(input_dir, fname)
        try:
            raw_vis, img_chw = load_and_preprocess_image(path, size=img_size)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
            
        # Inference
        img_t = torch.from_numpy(img_chw).unsqueeze(0).to(device)
        
        with torch.no_grad():
            seg_logits, param_preds = model(img_t)
            
            # 1. Process Mask
            prob = torch.sigmoid(seg_logits)[0, 0] # 256x256
            mask = (prob > 0.5).cpu().numpy().astype(np.uint8)
            
            # Resize mask back to original image size
            H, W = raw_vis.shape[:2]
            mask_full = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # 2. Process Params
            if p_mean is not None:
                params = (param_preds[0].cpu().numpy() * p_std) + p_mean
            else:
                params = param_preds[0].cpu().numpy()
                
        # Save Results
        base = os.path.splitext(fname)[0]
        
        # Save Mask (16-bit)
        mask16 = mask_full.astype(np.uint16) * 65535
        tiff.imwrite(os.path.join(output_dir, f"{base}_mask.tif"), mask16)
        
        # Save Overlay (Pink Contours)
        overlay = _prepare_overlay16(raw_vis).copy()
        cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, PINK_16, thickness=4) # Thicker line
        tiff.imwrite(os.path.join(output_dir, f"{base}_overlay.tif"), overlay)
        
        # Save Params
        p_dict = {k: float(v) for k, v in zip(PARAM_COLS, params)}
        with open(os.path.join(output_dir, f"{base}_params.json"), "w") as f:
            json.dump(p_dict, f, indent=2)
            
        print(f"Saved: {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    infer(args.ckpt, args.input_dir, args.output_dir)