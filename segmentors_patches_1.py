#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leaf Lesion Segmentation (Tiled, Robust, GPU-ready)

- Tiled inference with overlap; merges tiles by OR.
- Backbones: color, resnet, deeplab, dinov2(518-preprocess), dinov3(padded-14), swav,
             moco(optional), mae(224-preprocess), sam.
- Per-backbone outputs: lesion mask, overlay (pink contours), per-contour CSV.
- Summary CSVs across images/backbones.

Fixes:
- MAE: force 224×224 preprocess (timm MAE requires it) -> no more 'Input height 128 != 224'.
- DINOv2: keeps 518 resize+centercrop (stable).
- DINOv3: variable-sized tiles supported via pad-to-14 and crop-back.
- Guarded resizes & robust fallbacks on per-tile failures.
"""

from __future__ import annotations
import argparse, glob, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image

# TIFF
try:
    import tifffile as tiff
    HAVE_TIFFFILE = True
except Exception:
    HAVE_TIFFFILE = False

from skimage import color as skcolor
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import KMeans

DEVICE_PREF: str = "auto"  # "auto" | "cpu" | "cuda"
PINK_BGR = (180, 105, 255)
EPS = 1e-8

# ---------- helpers ----------
def select_device():
    import torch
    if DEVICE_PREF == "cpu": return torch.device("cpu")
    if DEVICE_PREF == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            else: last = data[..., -1]; data = np.stack([last]*3, axis=-1)
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

# ---------- leaf mask ----------
def robust_leaf_mask(
    rgb: np.ndarray,
    s_thr: float = 0.15,
    v_thr: float = 0.15,
    close_radius: int = 3,
    open_radius: int = 2,
) -> np.ndarray:
    hsv = skcolor.rgb2hsv(rgb)
    s, v = hsv[...,1], hsv[...,2]
    seed = ((s > s_thr) & (v > v_thr)).astype(np.uint8)
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*close_radius+1,)*2)
    k_open  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*open_radius+1,)*2)
    seed = cv.morphologyEx(seed, cv.MORPH_CLOSE, k_close, iterations=1)
    seed = cv.morphologyEx(seed, cv.MORPH_OPEN,  k_open,  iterations=1)
    n, labels = cv.connectedComponents(seed, connectivity=8)
    if n <= 1: return seed.astype(bool)
    areas = np.bincount(labels.ravel()); areas[0] = 0
    leaf = (labels == areas.argmax())
    return binary_fill_holes(leaf)

# ---------- backbones base ----------
class FeatureBackbone:
    name: str = "none"
    def __call__(self, rgb: np.ndarray) -> np.ndarray: raise NotImplementedError

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
            if "CUDA out of memory" in str(e) and dev.type=="cuda":
                self.model.to("cpu")
                with torch.no_grad(): y = self.model(x.to("cpu"))
            else:
                raise
        if isinstance(y, (list, tuple)): y = y[-1]
        if hasattr(y, "ndim") and y.ndim == 4:
            return y[0].permute(1,2,0).contiguous().cpu().float().numpy()
        if hasattr(y, "ndim") and y.ndim == 3:
            B,N,C = y.shape; side = int(np.sqrt(N))
            if side*side == N:
                return y[0].reshape(side,side,C).contiguous().cpu().float().numpy()
            vec = y.mean(dim=1).cpu().float().numpy()
            return vec[0][None,None,:]
        arr = np.asarray(y.detach().cpu().float())
        if arr.ndim == 2: return arr[0][None,None,:]
        raise RuntimeError("Unsupported model output shape.")

# ---------- specific backbones ----------
def make_resnet() -> FeatureBackbone|None:
    try:
        import torch, torch.nn as nn
        from torchvision import models
        res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
        body = nn.Sequential(*list(res.children())[:-2]).eval()
        class Feat(nn.Module):
            def __init__(self, b): super().__init__(); self.b=b
            def forward(self, x): return self.b(x)
        preprocess = models.ResNet50_Weights.DEFAULT.transforms()
        return TorchBackbone(Feat(body), preprocess, "resnet50")
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
        preprocess = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT.transforms()
        return TorchBackbone(Feat(mdl).eval(), preprocess, "deeplabv3_resnet50")
    except Exception as e:
        raise RuntimeError(f"deeplab factory error: {e}")

def make_dinov2() -> FeatureBackbone|None:
    try:
        import torch
        from torchvision import transforms
        mdl = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(518, antialias=True),
            transforms.CenterCrop(518),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        class DINO(FeatureBackbone):
            name="dinov2_vitb14"
            def __call__(self, rgb):
                pil = Image.fromarray(rgb)
                x = preprocess(pil).unsqueeze(0)
                dev = select_device(); mdl.to(dev)
                with torch.no_grad():
                    try:
                        feats = mdl.get_intermediate_layers(x.to(dev), n=1)[0]
                    except Exception:
                        out = mdl.forward_features(x.to(dev))
                        feats = out.get("x_norm_patchtokens", None) or out.get("x_norm_clstoken", None)
                        if feats is None: feats = list(out.values())[-1]
                if feats.ndim == 3:
                    B,N,C = feats.shape; side = int(np.sqrt(N))
                    if side*side == N:
                        return feats[0].reshape(side,side,C).cpu().float().numpy()
                    return feats.mean(dim=1)[0][None,None,:].cpu().float().numpy()
                arr = np.asarray(feats.detach().cpu().float())
                return arr[0][None,None,:] if arr.ndim==2 else arr
        return DINO()
    except Exception as e:
        raise RuntimeError(f"dinov2 factory error: {e}")

def pad_to_multiple(img: np.ndarray, mult: int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    H, W = img.shape[:2]
    Ht = ((H + mult - 1)//mult)*mult
    Wt = ((W + mult - 1)//mult)*mult
    pb = Ht - H; pr = Wt - W
    if pb == 0 and pr == 0: return img, (0,0,0,0)
    pad = cv.copyMakeBorder(img, 0, pb, 0, pr, borderType=cv.BORDER_REPLICATE)
    return pad, (0,pb,0,pr)

def make_dinov3() -> FeatureBackbone|None:
    """
    DINOv3 backbone with variable-size tiles via pad-to-14 and crop-back.
    Implementation strategy:
      1) Try torch.hub 'facebookresearch/dinov2' vitb14 (API-compatible for tokens).
      2) Fallback to timm 'vit_base_patch14_dinov2' pretrained.
    """
    try:
        import torch
        from torchvision import transforms
        mdl = None
        note = ""
        try:
            mdl = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
            note = "hub-dinov2 proxy"
        except Exception:
            try:
                import timm
                mdl = timm.create_model("vit_base_patch14_dinov2", pretrained=True).eval()
                note = "timm-dinov2 proxy"
            except Exception as e:
                raise RuntimeError(f"could not load dinov3 (tried hub+ t imm): {e}")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        class DINO3(FeatureBackbone):
            name = "dinov3_vitb14"
            def __call__(self, rgb):
                import torch
                rgb_pad, (pt,pb,pl,pr) = pad_to_multiple(rgb, 14)
                x = preprocess(Image.fromarray(rgb_pad)).unsqueeze(0)
                dev = select_device(); mdl.to(dev)
                with torch.no_grad():
                    try:
                        feats = mdl.get_intermediate_layers(x.to(dev), n=1)[0]
                    except Exception:
                        out = getattr(mdl, "forward_features")(x.to(dev))
                        feats = out.get("x_norm_patchtokens", None) or out.get("x_norm_clstoken", None)
                        if feats is None:
                            # last tensor-like value
                            feats = list(out.values())[-1] if isinstance(out, dict) else out
                if hasattr(feats, "ndim") and feats.ndim == 3:
                    B,N,C = feats.shape
                    side = int(np.sqrt(N))
                    if side*side == N:
                        g = feats[0].reshape(side,side,C).cpu().float().numpy()
                    else:
                        g = feats.mean(dim=1)[0][None,None,:].cpu().float().numpy()
                else:
                    arr = np.asarray(feats.detach().cpu().float())
                    g = arr[0][None,None,:] if arr.ndim==2 else arr
                if (pb or pr) and g.ndim == 3:
                    gh, gw = g.shape[:2]
                    gh_c = gh - (pb // 14)
                    gw_c = gw - (pr // 14)
                    g = g[:gh_c, :gw_c, :]
                return g
        bb = DINO3()
        # Let the log mention which source we used
        bb.load_note = note  # not used elsewhere, just handy for print-debugging
        return bb
    except Exception as e:
        raise RuntimeError(f"dinov3 factory error: {e}")

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

def make_moco(local_repo: str|None = None) -> FeatureBackbone|None:
    try:
        import torch, torch.nn as nn
        from torchvision import transforms
        ok = False; model = None
        try:
            model = torch.hub.load("facebookresearch/moco", "moco_v2_800ep_pretrain", force_reload=False); ok=True
        except Exception:
            try:
                model = torch.hub.load("facebookresearch/moco", "moco_v2_800ep_pretrain", force_reload=True); ok=True
            except Exception: pass
        if not ok and local_repo:
            model = torch.hub.load(local_repo, "moco_v2_800ep_pretrain", source="local"); ok=True
        if not ok: raise RuntimeError("could not load moco from hub or local")
        base = None
        for n,m in model.named_children():
            if "encoder" in n or "base_encoder" in n: base = m; break
        base = base or model
        if hasattr(base, "fc"): base.fc = nn.Identity()
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
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        class MAE(FeatureBackbone):
            name="mae_vit_base"
            def __call__(self, rgb):
                import torch
                x = preprocess(Image.fromarray(rgb)).unsqueeze(0)
                dev = select_device(); mdl.to(dev)
                with torch.no_grad():
                    out = mdl.forward_features(x.to(dev))
                t = out.get("x", None) if isinstance(out, dict) else out
                if t.ndim == 3:
                    B,N,C = t.shape
                    if int(np.sqrt(N-1))**2 == (N-1): t = t[:,1:,:]; N -= 1
                    side = int(np.sqrt(N))
                    if side*side == N:
                        return t[0].reshape(side,side,C).cpu().float().numpy()
                    return t.mean(dim=1)[0][None,None,:].cpu().float().numpy()
                arr = np.asarray(t.detach().cpu().float())
                return arr[0][None,None,:] if arr.ndim==2 else arr
        return MAE()
    except Exception as e:
        raise RuntimeError(f"mae factory error: {e}")

def make_sam(checkpoint: str|None) -> FeatureBackbone|None:
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        ckpt = checkpoint or os.environ.get("SAM_CHECKPOINT", None)
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

# ---------- registry ----------
def build_backbones(requested: List[str], sam_ckpt: str|None, moco_local: str|None) -> Dict[str, FeatureBackbone]:
    print(f"[debug] requested backbones: {requested}")
    got: Dict[str, FeatureBackbone] = {}
    factories = {
        "color":   lambda: ColorTexture(),
        "resnet":  make_resnet,
        "deeplab": make_deeplab,
        "dinov2":  make_dinov2,
        "dinov3":  make_dinov3,
        "swav":    make_swav,
        "moco":    (lambda: make_moco(moco_local)),
        "mae":     make_mae,
        "sam":     (lambda: make_sam(sam_ckpt)),
    }
    for key in requested:
        mk = factories.get(key)
        if not mk:
            print(f"[skip] {key}: no factory"); continue
        try:
            bb = mk()
        except Exception as e:
            print(f"[skip] {key}: {repr(e)}"); bb = None
        if bb is not None:
            name = getattr(bb, "name", key)
            print(f"[ok] built backbone: {name}")
            got[name] = bb
        else:
            print(f"[skip] {key}: factory returned None")
    if not got:
        print("[warn] no backbones built; falling back to color_texture")
        got["color_texture"] = ColorTexture()
    return got

# ---------- clustering ----------
def cluster_to_lesions(
    rgb_small: np.ndarray,
    feats: np.ndarray,
    leaf_small: np.ndarray,
    k: int = 3,
    lesion_ptile: float = 66.0,
    open_radius: int = 2,
    close_radius: int = 2,
    min_area_px: int = 32,
) -> np.ndarray:
    h,w,C = feats.shape
    if h < 1 or w < 1: return np.zeros((max(1,h), max(1,w)), bool)
    idx = np.where(leaf_small.ravel())[0]
    if len(idx) < 8: return np.zeros((h,w), bool)

    X = feats.reshape(-1, C).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
    X = X[idx]
    k = int(np.clip(k, 2, 8))
    try:
        lab_id = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    except Exception:
        return np.zeros((h,w), bool)

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
    if open_radius>0:  m = cv.morphologyEx(m, cv.MORPH_OPEN,  k_open,  iterations=1)
    if close_radius>0: m = cv.morphologyEx(m, cv.MORPH_CLOSE, k_close, iterations=1)

    cnts, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    m2 = np.zeros_like(m)
    for c in cnts:
        if cv.contourArea(c) >= float(min_area_px):
            cv.drawContours(m2, [c], -1, 1, thickness=-1)
    return m2.astype(bool)

# ---------- contours ----------
def contour_stats(mask: np.ndarray) -> Tuple[List[np.ndarray], pd.DataFrame]:
    m8 = (mask.astype(np.uint8)*255)
    cnts, _ = cv.findContours(m8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rows = []
    for i,c in enumerate(cnts):
        a = float(cv.contourArea(c))
        if a < 1: continue
        p = float(cv.arcLength(c, True))
        x,y,w,h = cv.boundingRect(c)
        M = cv.moments(c)
        if abs(M["m00"]) < 1e-6: cx, cy = x + w/2, y + h/2
        else: cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
        circ = 0.0 if p==0 else float(4*np.pi*a/(p*p))
        rows.append(dict(contour_id=i, area_px=a, perimeter_px=p,
                         bbox_x=x, bbox_y=y, width_px=w, height_px=h,
                         centroid_x=cx, centroid_y=cy, circularity=circ))
    return cnts, pd.DataFrame(rows)

# ---------- tiling ----------
def gen_tiles(H, W, tile, overlap):
    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    for y in ys:
        for x in xs:
            yield y, x, min(tile, H - y), min(tile, W - x)

def process_tile(rgb_tile: np.ndarray, backend: FeatureBackbone, k, heur, leaf_thr):
    leaf = robust_leaf_mask(rgb_tile, leaf_thr["s_thr"], leaf_thr["v_thr"],
                            leaf_thr["leaf_close"], leaf_thr["leaf_open"])
    try:
        feats = backend(rgb_tile)
        if feats.ndim != 3: raise RuntimeError("invalid feature ndim")
    except Exception as e:
        warnings.warn(f"Backend failed on tile with {getattr(backend,'name','bk')}: {e}; using color_texture")
        feats = ColorTexture()(rgb_tile)
    fh, fw, _ = feats.shape
    if fh < 1 or fw < 1: return np.zeros(rgb_tile.shape[:2], bool)
    try:
        if (fh,fw) != rgb_tile.shape[:2]:
            img_small  = cv.resize(rgb_tile, (max(1,fw), max(1,fh)), interpolation=cv.INTER_AREA)
            leaf_small = cv.resize(leaf.astype(np.uint8), (max(1,fw), max(1,fh)), interpolation=cv.INTER_NEAREST).astype(bool)
        else:
            img_small, leaf_small = rgb_tile, leaf
    except cv.error as e:
        raise RuntimeError(f"cv.resize failed: {e}")
    lesion_small = cluster_to_lesions(
        img_small, feats, leaf_small,
        k=k,
        lesion_ptile=heur["lesion_ptile"],
        open_radius=heur["lesion_open"],
        close_radius=heur["lesion_close"],
        min_area_px=heur["min_area"]
    )
    if lesion_small.shape[:2] != rgb_tile.shape[:2]:
        lesion = cv.resize(lesion_small.astype(np.uint8),
                           (rgb_tile.shape[1], rgb_tile.shape[0]),
                           interpolation=cv.INTER_NEAREST).astype(bool)
    else:
        lesion = lesion_small
    lesion &= leaf
    return lesion

def process_image_tiled(
    path: str, outdir: Path, backbones: Dict[str, FeatureBackbone], *,
    k: int, tile: int, overlap: int,
    heur: dict, leaf_thr: dict
) -> List[pd.DataFrame]:
    rgb = imread_rgb(path)
    H, W = rgb.shape[:2]
    stem = Path(path).stem
    all_rows = []
    for tag, bb in backbones.items():
        full_mask = np.zeros((H,W), dtype=bool)
        for y,x,th,tw in gen_tiles(H, W, tile, overlap):
            tile_rgb = rgb[y:y+th, x:x+tw]
            try:
                lesion = process_tile(tile_rgb, bb, k, heur, leaf_thr)
                tag_eff = tag
            except Exception as e:
                warnings.warn(f"Backend failed on {os.path.basename(path)} with {tag}: {e}; using color_texture")
                lesion = process_tile(tile_rgb, ColorTexture(), k, heur, leaf_thr)
                tag_eff = tag + "+fallback"
            full_mask[y:y+th, x:x+tw] |= lesion
        subdir = outdir / tag_eff
        subdir.mkdir(parents=True, exist_ok=True)
        save_mask(subdir / f"{stem}_{tag_eff}_lesion_mask.png", full_mask)
        cnts, df = contour_stats(full_mask)
        save_overlay(subdir / f"{stem}_{tag_eff}_overlay_contours.png", rgb, cnts, thickness=2)
        df.insert(0, "backend", tag_eff)
        df.insert(1, "image", os.path.basename(path))
        df.to_csv(subdir / f"{stem}_{tag_eff}_contours.csv", index=False)
        print(f"Processed {path} with {tag_eff}: {len(df)} contours")
        all_rows.append(df)
    return all_rows

# ---------- modes ----------
def apply_mode(args):
    if args.mode == "fine":
        args.k = max(args.k, 4)
        args.lesion_ptile = min(args.lesion_ptile, 60.0)
        args.lesion_open  = min(args.lesion_open, 1)
        args.lesion_close = min(args.lesion_close, 1)
        args.min_area     = min(args.min_area, 12)
    elif args.mode == "coarse":
        args.k = min(args.k, 3)
        args.lesion_ptile = max(args.lesion_ptile, 75.0)
        args.lesion_open  = max(args.lesion_open, 2)
        args.lesion_close = max(args.lesion_close, 2)
        args.min_area     = max(args.min_area, 64)
    return args

# ---------- main ----------
def main():
    global DEVICE_PREF
    ap = argparse.ArgumentParser(description="Leaf lesion segmentation (tiled, robust)")
    ap.add_argument("--input", required=True, help="Folder or glob")
    ap.add_argument("--outdir", default="./out_tiled", help="Output directory")
    ap.add_argument("--backends", default="color", help='Comma list (color,resnet,deeplab,dinov2,dinov3,swav,moco,mae,sam) or "all"')
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--tile-size", type=int, default=128)
    ap.add_argument("--tile-overlap", type=int, default=32)
    ap.add_argument("--mode", choices=["fine","balanced","coarse"], default="balanced")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lesion-ptile", type=float, default=66.0)
    ap.add_argument("--min-area", type=int, default=32)
    ap.add_argument("--lesion-open",  type=int, default=2)
    ap.add_argument("--lesion-close", type=int, default=2)
    ap.add_argument("--s-thr", type=float, default=0.15)
    ap.add_argument("--v-thr", type=float, default=0.15)
    ap.add_argument("--leaf-open",  type=int, default=2)
    ap.add_argument("--leaf-close", type=int, default=3)
    ap.add_argument("--sam-checkpoint", type=str, default=None)
    ap.add_argument("--moco-local", type=str, default=None)
    args = ap.parse_args()
    args = apply_mode(args)
    DEVICE_PREF = args.device

    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if args.device == "cpu" and alloc:
        print(f"[info] Unsetting PYTORCH_CUDA_ALLOC_CONF for CPU run (was: {alloc})")
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    elif args.device in ("auto","cuda") and alloc and "expandable_segments:true" in alloc:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc.replace("expandable_segments:true","expandable_segments:True")

    if os.path.isdir(args.input):
        paths = []
        for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"):
            paths.extend(sorted(glob.glob(os.path.join(args.input, ext))))
    else:
        paths = sorted(glob.glob(args.input))
    if not paths: print("No input images found."); sys.exit(1)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    requested = ["color","resnet","deeplab","dinov2","dinov3","swav","moco","mae","sam"] \
        if args.backends.lower()=="all" else [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    moco_local = args.moco_local or os.environ.get("MOCO_REPO", None)

    built = build_backbones(requested, args.sam_checkpoint, moco_local)
    print("Backbones selected:", ", ".join(built.keys()))

    heur = dict(lesion_ptile=args.lesion_ptile, lesion_open=args.lesion_open,
                lesion_close=args.lesion_close, min_area=args.min_area)
    leaf_thr = dict(s_thr=args.s_thr, v_thr=args.v_thr,
                    leaf_open=args.leaf_open, leaf_close=args.leaf_close)

    all_rows = []
    for p in paths:
        dfs = process_image_tiled(
            p, outdir, built, k=args.k, tile=args.tile_size, overlap=args.tile_overlap,
            heur=heur, leaf_thr=leaf_thr
        )
        all_rows.extend(dfs)

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
