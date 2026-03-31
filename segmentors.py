#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leaf Lesion Segmentation (Full Image; ViTs auto-padded to patch multiples)
- Backbones: color, resnet, deeplab, dinov2, swav, moco, mae, sam
- DINOv2 (ViT/14) and MAE (ViT/16) are padded to 14x / 16x respectively, then cropped back.
- KMeans clustering on native feature grid, pink contours overlays, CSVs.
"""

from __future__ import annotations
import argparse, glob, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image

# TIFF (16-bit)
try:
    import tifffile as tiff
    HAVE_TIFFFILE = True
except Exception:
    HAVE_TIFFFILE = False

from skimage import color as skcolor
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import KMeans

# --------- Globals ---------
DEVICE_PREF: str = "auto"
PINK_BGR = (180, 105, 255)

# --------- IO helpers ---------
def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, 2.0); hi = np.percentile(arr, 99.8)
    if hi <= lo: hi = arr.max() if arr.max() > lo else lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255.0 + 0.5).astype(np.uint8)

def imread_rgb(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if HAVE_TIFFFILE and ext in {".tif",".tiff"}:
        data = tiff.imread(path)
        if data.ndim == 3 and data.shape[0] > 4 and data.shape[2] not in (3,4):
            data = data[0]
        if data.ndim == 2:
            img8 = _normalize_to_uint8(data)
            return np.stack([img8]*3, axis=-1)
        if data.ndim == 3:
            if data.shape[0] in (1,3,4) and data.shape[-1] not in (3,4):
                data = np.moveaxis(data, 0, -1)
            if data.shape[-1] >= 3: data = data[..., :3]
            else: data = np.stack([data[..., -1]]*3, axis=-1)
            if data.dtype != np.uint8: data = _normalize_to_uint8(data)
            return data
        raise RuntimeError(f"Unsupported TIFF shape: {data.shape}")
    bgr = cv.imread(path, cv.IMREAD_COLOR)
    if bgr is None: raise RuntimeError(f"Failed to read image: {path}")
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), (mask.astype(np.uint8)*255))

def save_overlay(path: Path, rgb: np.ndarray, contours: List[np.ndarray], thickness: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR).copy()
    cv.drawContours(bgr, contours, -1, PINK_BGR, thickness, lineType=cv.LINE_AA)
    cv.imwrite(str(path), bgr)

# --------- Leaf mask ---------
def robust_leaf_mask(rgb: np.ndarray, s_thr=0.15, v_thr=0.15, close_radius=3, open_radius=2) -> np.ndarray:
    hsv = skcolor.rgb2hsv(rgb); s, v = hsv[...,1], hsv[...,2]
    seed = ((s > s_thr) & (v > v_thr)).astype(np.uint8)
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*close_radius+1,)*2)
    k_open  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*open_radius+1,)*2)
    seed = cv.morphologyEx(seed, cv.MORPH_CLOSE, k_close, iterations=1)
    seed = cv.morphologyEx(seed, cv.MORPH_OPEN,  k_open,  iterations=1)
    n, labels = cv.connectedComponents(seed, 8)
    if n <= 1: return seed.astype(bool)
    areas = np.bincount(labels.ravel()); areas[0] = 0
    leaf = (labels == areas.argmax())
    return binary_fill_holes(leaf)

# --------- Backbones ---------
class FeatureBackbone:
    name: str = "none"
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ColorTexture(FeatureBackbone):
    name = "color_texture"
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        lab = skcolor.rgb2lab(rgb).astype(np.float32)
        hsv = skcolor.rgb2hsv(rgb).astype(np.float32)
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY).astype(np.float32)/255.0
        gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx*gx + gy*gy)[..., None]
        return np.concatenate([lab, hsv, grad], axis=-1)

def select_device():
    import torch
    if DEVICE_PREF == "cpu": return torch.device("cpu")
    if DEVICE_PREF == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TorchBackbone(FeatureBackbone):
    def __init__(self, model, preprocess, name: str):
        self.model = model.eval()
        self.preprocess = preprocess
        self.name = name
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        pil = Image.fromarray(rgb)
        x = self.preprocess(pil).unsqueeze(0)
        dev = select_device(); self.model.to(dev)
        try:
            with torch.no_grad(): y = self.model(x.to(dev))
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and dev.type == "cuda":
                self.model.to("cpu")
                with torch.no_grad(): y = self.model(x.to("cpu"))
            else: raise
        if isinstance(y, (list, tuple)): y = y[-1]
        if hasattr(y, "ndim") and y.ndim == 4:
            return y[0].permute(1,2,0).contiguous().cpu().float().numpy()
        if hasattr(y, "ndim") and y.ndim == 3:
            B,N,C = y.shape; side = int(np.sqrt(N))
            if side*side == N: return y[0].reshape(side,side,C).cpu().float().numpy()
            v = y.mean(dim=1)[0].cpu().float().numpy()
            return v[None,None,:]
        arr = np.asarray(y.detach().cpu().float())
        if arr.ndim == 2: return arr[0][None,None,:]
        raise RuntimeError("Unsupported model output shape.")

# ---- helpers for ViTs ----
def pad_to_multiple(img: np.ndarray, mult: int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    H, W = img.shape[:2]
    Ht = ((H + mult - 1)//mult)*mult
    Wt = ((W + mult - 1)//mult)*mult
    pb = Ht - H; pr = Wt - W
    if pb == 0 and pr == 0: return img, (0,0,0,0)
    pad = cv.copyMakeBorder(img, 0, pb, 0, pr, borderType=cv.BORDER_REPLICATE)
    return pad, (0,pb,0,pr)

def make_resnet() -> FeatureBackbone|None:
    try:
        import torch, torch.nn as nn
        from torchvision import models
        res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
        body = nn.Sequential(*list(res.children())[:-2]).eval()
        class Feat(nn.Module):
            def __init__(self, b): super().__init__(); self.b=b
            def forward(self, x): return self.b(x)
        feat = Feat(body); preprocess = models.ResNet50_Weights.DEFAULT.transforms()
        return TorchBackbone(feat, preprocess, "resnet50")
    except Exception as e:
        raise RuntimeError(f"resnet factory error: {e}")

def make_deeplab() -> FeatureBackbone|None:
    try:
        import torch, torch.nn as nn
        from torchvision import models
        mdl = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        class Feat(nn.Module):
            def __init__(self, base): super().__init__(); self.bb=base.backbone
            def forward(self, x): return self.bb(x)["out"]
        feat = Feat(mdl).eval()
        preprocess = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT.transforms()
        return TorchBackbone(feat, preprocess, "deeplabv3_resnet50")
    except Exception as e:
        raise RuntimeError(f"deeplab factory error: {e}")

def make_dinov2() -> FeatureBackbone|None:
    try:
        import torch
        from torchvision import transforms
        mdl = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
        # IMPORTANT: no Resize/CenterCrop so we preserve spatial tokens; we will pad to 14x instead
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        class DINO(TorchBackbone):
            def __call__(self, rgb):
                import torch
                # pad to multiple of 14
                rgb_pad, (pt,pb,pl,pr) = pad_to_multiple(rgb, 14)
                pil = Image.fromarray(rgb_pad)
                x = self.preprocess(pil).unsqueeze(0)
                dev = select_device(); self.model.to(dev)
                with torch.no_grad():
                    try:
                        feats = self.model.get_intermediate_layers(x.to(dev), n=1)[0]  # [B,N,C]
                    except Exception:
                        out = getattr(self.model, "forward_features")(x.to(dev))
                        feats = out.get("x_norm_patchtokens", None) or out.get("x_norm_clstoken", None)
                        if feats is None: feats = list(out.values())[-1]
                if feats.ndim == 3:
                    B,N,C = feats.shape; side = int(np.sqrt(N))
                    if side*side == N:
                        g = feats[0].reshape(side,side,C).cpu().float().numpy()
                    else:
                        g = feats.mean(dim=1)[0][None,None,:].cpu().float().numpy()
                else:
                    arr = np.asarray(feats.detach().cpu().float())
                    g = arr[0][None,None,:] if arr.ndim==2 else arr
                # crop tokens back using patch multiple (14)
                if pb or pr:
                    gh, gw = g.shape[:2]
                    gh_c = gh - (pb // 14)
                    gw_c = gw - (pr // 14)
                    g = g[:gh_c, :gw_c, :]
                return g
        return DINO(mdl, preprocess, "dinov2_vitb14")
    except Exception as e:
        raise RuntimeError(f"dinov2 factory error: {e}")

def make_swav() -> FeatureBackbone|None:
    try:
        import torch, torch.nn as nn
        from torchvision import transforms
        mdl = torch.hub.load("facebookresearch/swav", "resnet50")
        mdl.fc = nn.Identity()
        body = nn.Sequential(*list(mdl.children())[:-2]).eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        return TorchBackbone(body, preprocess, "swav_resnet50")
    except Exception as e:
        raise RuntimeError(f"swav factory error: {e}")

def make_moco() -> FeatureBackbone|None:
    try:
        import torch, torch.nn as nn
        from torchvision import transforms
        mdl = torch.hub.load("facebookresearch/moco", "moco_v2_800ep_pretrain", force_reload=False)
        base = None
        for n,m in mdl.named_children():
            if "encoder" in n or "base_encoder" in n: base = m; break
        base = base or mdl
        if hasattr(base,"fc"): base.fc = nn.Identity()
        body = nn.Sequential(*list(base.children())[:-2]).eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        return TorchBackbone(body, preprocess, "moco_v2_resnet50")
    except Exception as e:
        raise RuntimeError(f"moco factory error: {e}")

def make_mae() -> FeatureBackbone|None:
    try:
        import torch, timm
        from torchvision import transforms
        mdl = timm.create_model("vit_base_patch16_224.mae", pretrained=True).eval()
        # IMPORTANT: no Resize/CenterCrop; we pad to 16x so variable spatial tokens are valid.
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        class MAE(TorchBackbone):
            def __call__(self, rgb):
                import torch
                rgb_pad, (pt,pb,pl,pr) = pad_to_multiple(rgb, 16)
                x = self.preprocess(Image.fromarray(rgb_pad)).unsqueeze(0)
                dev = select_device(); self.model.to(dev)
                with torch.no_grad(): out = self.model.forward_features(x.to(dev))
                t = out.get("x", None) if isinstance(out, dict) else out
                if t.ndim == 3:
                    B,N,C = t.shape
                    if int(np.sqrt(N-1))**2 == (N-1):
                        t = t[:,1:,:]; N -= 1
                    side = int(np.sqrt(N))
                    if side*side == N:
                        g = t[0].reshape(side,side,C).cpu().float().numpy()
                    else:
                        g = t.mean(dim=1)[0][None,None,:].cpu().float().numpy()
                else:
                    arr = np.asarray(t.detach().cpu().float())
                    g = arr[0][None,None,:] if arr.ndim==2 else arr
                if pb or pr:
                    gh, gw = g.shape[:2]
                    gh_c = gh - (pb // 16)
                    gw_c = gw - (pr // 16)
                    g = g[:gh_c, :gw_c, :]
                return g
        return MAE(mdl, preprocess, "mae_vit_base")
    except Exception as e:
        raise RuntimeError(f"mae factory error: {e}")

def make_sam(sam_checkpoint: str | None) -> FeatureBackbone|None:
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        ckpt = sam_checkpoint or os.environ.get("SAM_CHECKPOINT", None)
        sam = sam_model_registry.get("vit_h")(checkpoint=ckpt)
        gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            box_nms_thresh=0.7,
            min_mask_region_area=64,
        )
        class SAM(FeatureBackbone):
            name="sam_auto"
            def __call__(self, rgb):
                anns = gen.generate(cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
                H,W,_ = rgb.shape
                feat = np.zeros((H,W,2), np.float32)
                for a in anns:
                    m = a.get("segmentation")
                    if m is None: continue
                    score = float(a.get("stability_score",0.5))
                    area  = float(a.get("area",1.0))
                    w = score*np.log1p(area)
                    feat[m,0] += w
                    feat[m,1]  = np.maximum(feat[m,1], score)
                return feat
        return SAM()
    except Exception as e:
        raise RuntimeError(f"sam factory error: {e}")

BACKBONE_FACTORIES = {
    "color":   lambda **kw: ColorTexture(),
    "resnet":  lambda **kw: make_resnet(),
    "deeplab": lambda **kw: make_deeplab(),
    "dinov2":  lambda **kw: make_dinov2(),
    "swav":    lambda **kw: make_swav(),
    "moco":    lambda **kw: make_moco(),
    "mae":     lambda **kw: make_mae(),
    "sam":     lambda **kw: make_sam(kw.get("sam_checkpoint")),
}

# --------- Clustering & post ---------
def cluster_to_lesions(rgb_small, feats, leaf_small, k=3, lesion_ptile=66.0,
                       open_radius=2, close_radius=2, min_area_px=32) -> np.ndarray:
    h,w,C = feats.shape
    idx = np.where(leaf_small.ravel())[0]
    if len(idx) < 16: return np.zeros((h,w), bool)

    X = feats.reshape(-1, C).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    X = X[idx]

    k = int(np.clip(k, 2, 8))
    lab_id = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)

    Lab = skcolor.rgb2lab(rgb_small).reshape(-1,3)
    gray = cv.cvtColor(rgb_small, cv.COLOR_RGB2GRAY).reshape(-1).astype(np.float32)

    scores = []
    for c in range(k):
        sel = (lab_id == c)
        if not np.any(sel): scores.append(-1e9); continue
        pix = idx[sel]
        Lm = float(Lab[pix,0].mean()); am = float(Lab[pix,1].mean()); gm = float(gray[pix].mean())
        score = (-0.9*Lm) + (0.8*am) - (0.2*gm)
        scores.append(score)
    scores = np.array(scores)
    thr = np.percentile(scores, lesion_ptile)
    lesion_clusters = np.where(scores >= thr)[0]

    mask = np.zeros(h*w, bool)
    mask[idx[np.isin(lab_id, lesion_clusters)]] = True
    mask = mask.reshape(h,w)

    k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*open_radius+1,)*2)
    k_close= cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*close_radius+1,)*2)
    m = mask.astype(np.uint8)
    if open_radius>0:  m = cv.morphologyEx(m, cv.MORPH_OPEN,  k_open, 1)
    if close_radius>0: m = cv.morphologyEx(m, cv.MORPH_CLOSE, k_close, 1)

    cnts, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    m2 = np.zeros_like(m)
    for c in cnts:
        if cv.contourArea(c) >= float(min_area_px):
            cv.drawContours(m2, [c], -1, 1, thickness=-1)
    return m2.astype(bool)

def contour_stats(mask: np.ndarray):
    m8 = (mask.astype(np.uint8)*255)
    cnts, _ = cv.findContours(m8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rows = []
    for i,c in enumerate(cnts):
        a = float(cv.contourArea(c))
        if a < 1: continue
        p = float(cv.arcLength(c, True))
        x,y,w,h = cv.boundingRect(c)
        M = cv.moments(c)
        cx, cy = (x + w/2, y + h/2) if abs(M["m00"]) < 1e-6 else (M["m10"]/M["m00"], M["m01"]/M["m00"])
        circ = 0.0 if p==0 else float(4*np.pi*a/(p*p))
        rows.append(dict(contour_id=i, area_px=a, perimeter_px=p,
                         bbox_x=x, bbox_y=y, width_px=w, height_px=h,
                         centroid_x=cx, centroid_y=cy, circularity=circ))
    return cnts, pd.DataFrame(rows)

# --------- Builder ---------
def build_backbones(requested: List[str], sam_checkpoint: str|None) -> Dict[str, FeatureBackbone]:
    print(f"[debug] requested backbones: {requested}")
    got: Dict[str, FeatureBackbone] = {}
    for key in requested:
        mk = BACKBONE_FACTORIES.get(key)
        if not mk: print(f"[skip] {key}: no factory"); continue
        try:
            bb = mk(sam_checkpoint=sam_checkpoint)
        except Exception as e:
            print(f"[skip] {key}: {repr(e)}"); bb = None
        if bb is not None:
            name = getattr(bb, "name", key); print(f"[ok] built backbone: {name}")
            got[name] = bb
        else:
            print(f"[skip] {key}: factory returned None")
    if not got:
        print("[warn] no backbones built; falling back to color_texture")
        got["color_texture"] = ColorTexture()
    return got

# --------- Image pipeline ---------
def process_image(path: str, outdir: Path, backend: FeatureBackbone, *,
                  tag: str, k: int, max_edge: int,
                  s_thr: float, v_thr: float,
                  leaf_close: int, leaf_open: int,
                  lesion_ptile: float, lesion_close: int, lesion_open: int, min_area: int) -> pd.DataFrame:
    rgb = imread_rgb(path)
    H0, W0 = rgb.shape[:2]
    work = rgb
    if max_edge and max(H0,W0) > max_edge:
        scale = max_edge / float(max(H0,W0))
        work = cv.resize(rgb, (int(round(W0*scale)), int(round(H0*scale))), interpolation=cv.INTER_AREA)

    leaf_work = robust_leaf_mask(work, s_thr, v_thr, leaf_close, leaf_open)

    try:
        feats = backend(work)
        if feats.ndim != 3: raise RuntimeError("invalid feature shape")
    except Exception as e:
        warnings.warn(f"Backend failed on {os.path.basename(path)} with {getattr(backend,'name','bk')}: {e}; using color_texture")
        feats = ColorTexture()(work); tag += "+fallback"

    fh, fw, _ = feats.shape
    img_small = work if (work.shape[0]==fh and work.shape[1]==fw) else cv.resize(work, (fw, fh), interpolation=cv.INTER_AREA)
    leaf_small = leaf_work if (leaf_work.shape[0]==fh and leaf_work.shape[1]==fw) else cv.resize(leaf_work.astype(np.uint8), (fw, fh), interpolation=cv.INTER_NEAREST).astype(bool)

    lesion_small = cluster_to_lesions(img_small, feats, leaf_small, k=k,
                                      lesion_ptile=lesion_ptile,
                                      open_radius=lesion_open, close_radius=lesion_close,
                                      min_area_px=min_area)

    if (fh,fw) != work.shape[:2]:
        lesion_work = cv.resize(lesion_small.astype(np.uint8), (work.shape[1], work.shape[0]), interpolation=cv.INTER_NEAREST).astype(bool)
    else:
        lesion_work = lesion_small

    if work.shape[:2] != (H0,W0):
        leaf = cv.resize(leaf_work.astype(np.uint8), (W0,H0), interpolation=cv.INTER_NEAREST).astype(bool)
        lesion = cv.resize(lesion_work.astype(np.uint8), (W0,H0), interpolation=cv.INTER_NEAREST).astype(bool)
    else:
        leaf, lesion = leaf_work, lesion_work

    lesion &= leaf

    stem = Path(path).stem
    subdir = outdir / tag; subdir.mkdir(parents=True, exist_ok=True)
    save_mask(subdir / f"{stem}_{tag}_leaf_mask.png",   leaf)
    save_mask(subdir / f"{stem}_{tag}_lesion_mask.png", lesion)
    cnts, df = contour_stats(lesion)
    save_overlay(subdir / f"{stem}_{tag}_overlay_contours.png", rgb, cnts, thickness=2)

    df.insert(0, "backend", tag)
    df.insert(1, "image", os.path.basename(path))
    df.to_csv(subdir / f"{stem}_{tag}_contours.csv", index=False)

    print(f"Processed {path} with {tag}: {len(df)} contours")
    return df

# --------- Modes ---------
def apply_mode(args):
    if args.mode == "fine":
        args.k = max(args.k, 4)
        args.lesion_ptile = min(args.lesion_ptile, 60.0)
        args.lesion_open  = min(args.lesion_open, 1)
        args.lesion_close = min(args.lesion_close, 1)
        args.min_area     = min(args.min_area, 16)
    elif args.mode == "coarse":
        args.k = min(args.k, 3)
        args.lesion_ptile = max(args.lesion_ptile, 75.0)
        args.lesion_open  = max(args.lesion_open, 2)
        args.lesion_close = max(args.lesion_close, 2)
        args.min_area     = max(args.min_area, 64)
    return args

# --------- Main ---------
def main():
    global DEVICE_PREF
    ap = argparse.ArgumentParser(description="Leaf lesion segmentation (full-image, ViTs padded)")
    ap.add_argument("--input", required=True, help="Folder or glob (*.tif/*.png/*.jpg)")
    ap.add_argument("--outdir", default="./out", help="Output directory")
    ap.add_argument("--backends", default="color", help='Comma list (color,resnet,deeplab,dinov2,swav,moco,mae,sam) or "all"')
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Device preference")
    ap.add_argument("--max-edge", type=int, default=0, help="If >0, downscale longest edge for feature extraction only")
    ap.add_argument("--k", type=int, default=3, help="KMeans clusters (2..8)")
    ap.add_argument("--mode", choices=["fine","balanced","coarse"], default="balanced")
    ap.add_argument("--lesion-ptile", type=float, default=66.0)
    ap.add_argument("--min-area", type=int, default=32)
    ap.add_argument("--lesion-open",  type=int, default=2)
    ap.add_argument("--lesion-close", type=int, default=2)
    ap.add_argument("--s-thr", type=float, default=0.15)
    ap.add_argument("--v-thr", type=float, default=0.15)
    ap.add_argument("--leaf-open",  type=int, default=2)
    ap.add_argument("--leaf-close", type=int, default=3)
    ap.add_argument("--sam-checkpoint", type=str, default=None, help="Path to SAM .pth checkpoint (vit_h)")
    args = ap.parse_args()
    args = apply_mode(args); DEVICE_PREF = args.device

    # sanitize CUDA alloc conf
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if args.device == "cpu" and alloc:
        print(f"[info] Unsetting PYTORCH_CUDA_ALLOC_CONF for CPU run (was: {alloc})")
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    elif args.device in ("auto","cuda") and alloc and "expandable_segments:true" in alloc:
        fixed = alloc.replace("expandable_segments:true", "expandable_segments:True")
        if fixed != alloc:
            print(f"[info] Normalizing PYTORCH_CUDA_ALLOC_CONF -> {fixed}")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = fixed

    # inputs
    if os.path.isdir(args.input):
        paths = []
        for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"):
            paths.extend(sorted(glob.glob(os.path.join(args.input, ext))))
    else:
        paths = sorted(glob.glob(args.input))
    if not paths: print("No input images found."); sys.exit(1)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    requested = ["color","resnet","deeplab","dinov2","swav","moco","mae","sam"] if args.backends.lower()=="all" else \
                [b.strip().lower() for b in args.backends.split(",") if b.strip()]

    built = build_backbones(requested, sam_checkpoint=args.sam_checkpoint)
    print("Backbones selected:", ", ".join(built.keys()))

    all_rows = []
    for p in paths:
        for key, bb in built.items():
            tag = key
            df = process_image(
                p, outdir, bb, tag=tag, k=args.k, max_edge=args.max_edge,
                s_thr=args.s_thr, v_thr=args.v_thr,
                leaf_close=args.leaf_close, leaf_open=args.leaf_open,
                lesion_ptile=args.lesion_ptile, lesion_close=args.lesion_close,
                lesion_open=args.lesion_open, min_area=args.min_area
            )
            all_rows.append(df)

    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        full.to_csv(outdir / "all_contours_summary_by_backend.csv", index=False)
        per_img = (full.groupby(["backend","image"]).agg(
            n_contours=("contour_id","count"),
            total_area_px=("area_px","sum"),
            mean_circularity=("circularity","mean"),
            mean_width_px=("width_px","mean"),
            mean_height_px=("height_px","mean"),
            mean_perimeter_px=("perimeter_px","mean"),
        ).reset_index())
        per_img.to_csv(outdir / "per_image_summary_by_backend.csv", index=False)
        by_backend = (full.groupby("backend").agg(
            total_contours=("contour_id","count"),
            mean_area_px=("area_px","mean"),
            mean_circularity=("circularity","mean"),
        ).reset_index())
        by_backend.to_csv(outdir / "backend_overview_summary.csv", index=False)
        print(f"Saved comparative summaries in {outdir}")

if __name__ == "__main__":
    warnings.simplefilter("default", category=DeprecationWarning)
    main()
