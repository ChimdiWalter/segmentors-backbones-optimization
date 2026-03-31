#!/usr/bin/env python3
# nv_optimize_robust_v3_gpu.py
"""
GPU-accelerated Navier+Snake optimization.
- Moves gradients + diffusion/energy iterations to PyTorch (CUDA when --device cuda).
- Keeps scikit-image snakes + OpenCV I/O/contours.
- Same CLI as your v2, with extra flags: --device, --gpu-batch.
"""

import os, sys, csv, time, math, json, random, argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2

import torch
import torch.nn.functional as F

from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace as _laplace_cpu   # fallback (rare)
from skimage.draw import polygon

# ----------------------------- Constants / Config -----------------------------
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# 16-bit pink (B, G, R)
PINK_16 = (65535, 5140, 37779)

# Search bounds (aligned but widened)
BOUNDS = {
    "mu":               (0.03, 0.80),
    "lambda":           (0.03, 0.80),
    "diffusion_rate":   (0.03, 0.60),
    "alpha":            (0.05, 0.70),
    "beta":             (0.05, 1.20),
    "gamma":            (0.01, 0.60),
    "energy_threshold": (15, 140),  # integer
}

# Objective weights
TARGET_AREA_FRAC = 0.05
W_AREA   = 2.0
W_SMALL  = 0.5
SMALL_MIN_AREA = 25
MAX_AREA_FRAC_HARD = 0.60
MIN_AREA_FRAC_HARD = 0.0005

# ----------------------------- Utilities -------------------------------------
def ensure_dir(path): os.makedirs(path, exist_ok=True)
def fmt(v): return str(v).replace('.', 'p') if isinstance(v, float) else str(v)

def _to_uint8(img):
    if img is None: return None
    if img.dtype == np.uint8: return img
    if img.dtype == np.uint16: return cv2.convertScaleAbs(img, alpha=255.0/65535.0)
    if img.dtype in (np.float32, np.float64):
        im = np.clip(img, 0.0, 1.0); return (im * 255.0).astype(np.uint8)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx <= mn + 1e-12: return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * (img - mn) / (mx - mn)).astype(np.uint8)

def _to_float01(img):
    if img.dtype == np.uint8: return (img.astype(np.float32))/255.0
    if img.dtype == np.uint16: return (img.astype(np.float32))/65535.0
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx <= mn + 1e-12: return np.zeros_like(img, dtype=np.float32)
    return (img - mn)/(mx - mn)

def read_color_anydepth(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None: 
        return None, None, None
    if raw.ndim == 2:
        raw_gray = raw
        bgr8 = _to_uint8(raw_gray)
        if bgr8.ndim == 2:
            bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_GRAY2BGR)
        gray8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        return raw_gray, bgr8, gray8
    else:
        bgr8 = _to_uint8(raw)
        if bgr8.ndim == 2:
            bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_GRAY2BGR)
        gray8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        return raw, bgr8, gray8

def binary_threshold_mask(gray8, threshold=127):
    _, mask = cv2.threshold(gray8, threshold, 255, cv2.THRESH_BINARY)
    return mask

def apply_mask(image_u8, mask_u8):
    if mask_u8 is None: 
        return image_u8
    return cv2.bitwise_and(image_u8, image_u8, mask=mask_u8)

def _prepare_overlay16(raw):
    if raw.dtype == np.uint16:
        if raw.ndim == 2: return cv2.merge([raw, raw, raw])
        if raw.ndim == 3 and raw.shape[2] == 3: return raw.copy()
        if raw.ndim == 3 and raw.shape[2] == 1:
            g = raw[..., 0]; return cv2.merge([g, g, g])

    if raw.ndim == 2:
        raw8 = raw if raw.dtype == np.uint8 else _to_uint8(raw)
        raw16 = (raw8.astype(np.uint16)) * 257
        return cv2.merge([raw16, raw16, raw16])
    if raw.ndim == 3 and raw.shape[2] == 3:
        raw8 = raw if raw.dtype == np.uint8 else _to_uint8(raw)
        return (raw8.astype(np.uint16)) * 257

    raw8 = _to_uint8(raw)
    if raw8.ndim == 2:
        raw16 = (raw8.astype(np.uint16)) * 257
        return cv2.merge([raw16, raw16, raw16])
    else:
        return (raw8.astype(np.uint16)) * 257

def fused_channel_u8(bgr8, binary_mask=None):
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)
    candidates = [bgr8[..., 1], gray, lab[..., 1]]  # G, GRAY, a*

    fused = None
    for ch in candidates:
        chm = apply_mask(ch, binary_mask) if binary_mask is not None else ch
        chf = chm.astype(np.float32)
        mx  = float(chf.max())
        chn = (255.0 * chf / mx).astype(np.uint8) if mx > 1e-8 else np.zeros_like(chm, dtype=np.uint8)
        fused = chn if fused is None else np.maximum(fused, chn)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(fused)
    return apply_mask(enh, binary_mask) if binary_mask is not None else enh

# ----------------------------- GPU kernels -----------------------------------
def _make_conv(weight_2d: torch.Tensor, device):
    """Create a conv2d weight with shape [1,1,k,k] for single-channel."""
    k = weight_2d.to(device=device, dtype=torch.float32)
    k = k.view(1,1,k.shape[0],k.shape[1])
    return k

def _sobel_kernels(device):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
    return _make_conv(kx, device), _make_conv(ky, device)

def _laplace_kernel(device):
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
    return _make_conv(k, device)

@torch.no_grad()
def gpu_gradients(img_u8: np.ndarray, device: str):
    """
    img_u8: HxW uint8
    returns (mag_u8, gx_f32, gy_f32) on CPU numpy
    """
    device = torch.device(device)
    x = torch.from_numpy(img_u8.astype(np.float32)/255.0).to(device)
    x = x[None,None,...]  # [N=1,C=1,H,W]

    kx, ky = _sobel_kernels(device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy).clamp_min(1e-8)

    # return to CPU
    gx_np = gx.squeeze().detach().cpu().numpy()
    gy_np = gy.squeeze().detach().cpu().numpy()
    mag_u8 = (mag.squeeze().detach().cpu().numpy() / mag.max().item() * 255.0).astype(np.uint8)
    return mag_u8, gx_np, gy_np

@torch.no_grad()
def gpu_energy_map(
    image_u8: np.ndarray,
    seed_gx: np.ndarray,
    seed_gy: np.ndarray,
    iterations: int,
    diffusion_rate: float,
    mu: float,
    lam: float,
    edge_thresh: float,
    device: str
) -> np.ndarray:
    """
    Perform the diffusion / deformation iterations on GPU using PyTorch.
    Returns uint8 energy map on CPU (HxW).
    """
    device = torch.device(device)

    # Inputs to tensors
    img = torch.from_numpy(image_u8.astype(np.float32)/255.0).to(device)        # [H,W]
    img = img[None,None,...]                                                     # [1,1,H,W]
    s_gx = torch.from_numpy(seed_gx.astype(np.float32)).to(device)[None,None,...]
    s_gy = torch.from_numpy(seed_gy.astype(np.float32)).to(device)[None,None,...]

    # Image gradients (for edge gate and force)
    kx, ky = _sobel_kernels(device)
    im_gx = F.conv2d(img, kx, padding=1)
    im_gy = F.conv2d(img, ky, padding=1)
    im_mag = torch.sqrt(im_gx*im_gx + im_gy*im_gy)

    # Edge gate q
    q = (im_mag > (edge_thresh/255.0)).float()

    # Init diffused as |seed gradient|
    diffused = torch.sqrt(s_gx*s_gx + s_gy*s_gy).clone()

    lap_k = _laplace_kernel(device)

    # Iterative update
    for _ in range(int(iterations)):
        lap = F.conv2d(diffused, lap_k, padding=1)
        # A simple divergence proxy: finite-diff along x (like np.gradient(...)[0])
        # Use Sobel ky on diffused as a smooth derivative in y
        div_v = F.conv2d(diffused, ky, padding=1)

        ftx = im_gx - s_gx
        fty = im_gy - s_gy
        ftm = torch.sqrt(ftx*ftx + fty*fty)

        diffused = diffused + diffusion_rate * (mu * lap + (lam + mu) * div_v + q * (ftm - diffused))

    # Normalize to uint8
    eps = 1e-8
    mx = diffused.max().clamp_min(eps)
    out = (diffused / mx * 255.0).squeeze().detach().cpu().numpy().astype(np.uint8)
    return out

# ----------------------------- Objective -------------------------------------
@dataclass
class EvalResult:
    score: float
    n_contours: int
    area_px: int
    area_frac: float
    small_count: int

def evaluate_params_cpu_seg(image_u8: np.ndarray,
                            energy_u8: np.ndarray,
                            its_snake: int,
                            alpha: float, beta: float, gamma: float,
                            l_size: int = 5,
                            u_size: Optional[int] = None,
                            energy_threshold: int = 50) -> EvalResult:
    """CPU snakes + objective."""
    h, w = image_u8.shape
    if u_size is None:
        u_size = int(0.25 * h * w)

    labeled = label(energy_u8 > energy_threshold)
    props = regionprops(labeled)
    out_mask = np.zeros((h, w), dtype=bool)

    for rgn in props:
        if not (l_size < rgn.area < u_size):  # filter ROIs
            continue
        minr, minc, maxr, maxc = rgn.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5:
            continue

        crop_u8 = image_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop_u8), 3)

        # 200-point circle init
        s = np.linspace(0, 2*np.pi, 200, endpoint=False)
        rr = (maxr - minr) / 2.0
        cc = (maxc - minc) / 2.0
        init = np.vstack([rr * np.sin(s) + rr, cc * np.cos(s) + cc]).T

        try:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_num_iter=its_snake
            )
        except TypeError:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_iterations=its_snake
            )

        snake_i = np.round(snake).astype(int)
        snake_i[:, 0] = np.clip(snake_i[:, 0], 0, maxr - minr - 1)
        snake_i[:, 1] = np.clip(snake_i[:, 1], 0, maxc - minc - 1)

        pr, pc = snake_i[:, 0], snake_i[:, 1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop_u8.shape)
        out_mask[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    area_px = int(np.sum(out_mask))
    area_frac = area_px / float(h * w + 1e-8)

    if area_frac > MAX_AREA_FRAC_HARD or area_frac < MIN_AREA_FRAC_HARD:
        return EvalResult(score=-1e9, n_contours=0, area_px=area_px, area_frac=area_frac, small_count=0)

    seg_u8 = (out_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_contours = len(contours)

    lab = label(out_mask)
    props = regionprops(lab)
    small_count = sum(1 for p in props if p.area < SMALL_MIN_AREA)

    score = (n_contours
             - W_SMALL * small_count
             - W_AREA * abs(area_frac - TARGET_AREA_FRAC))

    return EvalResult(score=score, n_contours=n_contours, area_px=area_px,
                      area_frac=area_frac, small_count=small_count)

# ----------------------------- Param utils -----------------------------------
def sample_candidate(rng: random.Random) -> Dict:
    return {
        "mu": rng.uniform(*BOUNDS["mu"]),
        "lambda": rng.uniform(*BOUNDS["lambda"]),
        "diffusion_rate": rng.uniform(*BOUNDS["diffusion_rate"]),
        "alpha": rng.uniform(*BOUNDS["alpha"]),
        "beta": rng.uniform(*BOUNDS["beta"]),
        "gamma": rng.uniform(*BOUNDS["gamma"]),
        "energy_threshold": int(round(rng.uniform(*BOUNDS["energy_threshold"]))),
    }

def clip_params(p: Dict) -> Dict:
    q = dict(p)
    for k, (lo, hi) in BOUNDS.items():
        if k == "energy_threshold":
            q[k] = int(min(max(int(round(q[k])), int(lo)), int(hi)))
        else:
            q[k] = float(min(max(q[k], lo), hi))
    return q

def perturb(rng: random.Random, p: Dict, scale: float) -> Dict:
    q = dict(p)
    for k in ["mu","lambda","diffusion_rate","alpha","beta","gamma"]:
        span = BOUNDS[k][1] - BOUNDS[k][0]
        q[k] += rng.uniform(-1,1) * scale * span
    q["energy_threshold"] += int(round(rng.uniform(-1,1) * scale * (BOUNDS["energy_threshold"][1] - BOUNDS["energy_threshold"][0]) * 0.2))
    return clip_params(q)

# ----------------------------- Per-image Worker ------------------------------
def process_one_image(filename: str,
                      input_dir: str,
                      output_dir: str,
                      mode: str,
                      budget_random: int,
                      topk_refine: int,
                      refine_steps: int,
                      iters_coarse: int,
                      snake_iters_coarse: int,
                      iters_final: int,
                      snake_iters_final: int,
                      downscale: float,
                      per_image_seconds: Optional[int],
                      seed: int,
                      device: str,
                      gpu_batch: int) -> Dict:
    """
    device: "cuda" or "cpu"
    gpu_batch: batch size for GPU energy evaluation during random search (>=1).
    """
    rng = random.Random((hash(filename) ^ seed) & 0xFFFFFFFF)

    in_path = os.path.join(input_dir, filename)
    raw, bgr8_full, gray8_full = read_color_anydepth(in_path)
    if raw is None:
        return {"filename": filename, "status": "load_error"}

    # Select working channel
    if mode == "bestch":
        b8, g8, r8 = cv2.split(bgr8_full)
        leaf_mask = binary_threshold_mask(gray8_full, threshold=127)
        work_full = apply_mask(g8, leaf_mask)
    elif mode == "fused":
        leaf_mask = binary_threshold_mask(gray8_full, threshold=127)
        work_full = fused_channel_u8(bgr8_full, leaf_mask)
    else:
        return {"filename": filename, "status": f"bad_mode_{mode}"}

    # Downscale during search
    if 0.0 < downscale < 1.0:
        work = cv2.resize(work_full, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    else:
        work = work_full

    # GPU gradients for search image
    _, gx, gy = gpu_gradients(work, device=device)

    h, w = work.shape
    u_size = int(0.25 * h * w)

    t0 = time.time()
    best: Optional[Tuple[EvalResult, Dict]] = None
    tried = 0

    # -------- Random exploration (batched energy on GPU) --------
    while tried < budget_random:
        batch = min(gpu_batch, budget_random - tried)
        cand_list = []
        for _ in range(batch):
            c = sample_candidate(rng); c["iters"] = iters_coarse
            cand_list.append(c)

        # compute energies in batch on GPU, then CPU snakes per candidate
        # (Batch the "q, lap, div, updates" part; snakes remain CPU)
        energies = []
        for c in cand_list:
            en = gpu_energy_map(work, gx, gy,
                                iterations=c["iters"],
                                diffusion_rate=c["diffusion_rate"],
                                mu=c["mu"], lam=c["lambda"],
                                edge_thresh=c["energy_threshold"],
                                device=device)
            energies.append(en)

        for c, en in zip(cand_list, energies):
            res = evaluate_params_cpu_seg(work, en, snake_iters_coarse,
                                          alpha=c["alpha"], beta=c["beta"], gamma=c["gamma"],
                                          l_size=5, u_size=u_size, energy_threshold=int(c["energy_threshold"]))
            tried += 1
            if (best is None) or (res.score > best[0].score):
                best = (res, c)
            if per_image_seconds and (time.time() - t0 > per_image_seconds):
                return _finalize(filename, raw, work_full, best, mode, output_dir,
                                 iters_final, snake_iters_final, device, timeout=True)

    # -------- Local refinement around best --------
    seeds = [best[1]]
    for _ in range(max(0, topk_refine - 1)):
        seeds.append(perturb(rng, best[1], scale=0.25))

    for s_idx, seed_p in enumerate(seeds):
        current = (best[0], dict(seed_p))
        patience = 0
        for step in range(refine_steps):
            scale = 0.25 * (0.85 ** step)
            c = perturb(rng, current[1], scale=scale)
            c["iters"] = iters_coarse

            # GPU energy for candidate
            en = gpu_energy_map(work, gx, gy,
                                iterations=c["iters"],
                                diffusion_rate=c["diffusion_rate"],
                                mu=c["mu"], lam=c["lambda"],
                                edge_thresh=c["energy_threshold"],
                                device=device)

            res = evaluate_params_cpu_seg(work, en, snake_iters_coarse,
                                          alpha=c["alpha"], beta=c["beta"], gamma=c["gamma"],
                                          l_size=5, u_size=u_size, energy_threshold=int(c["energy_threshold"]))
            if res.score > current[0].score + 1e-6:
                current = (res, c)
                patience = 0
                if res.score > best[0].score + 1e-6:
                    best = (res, c)
            else:
                patience += 1

            if per_image_seconds and (time.time() - t0 > per_image_seconds):
                return _finalize(filename, raw, work_full, best, mode, output_dir,
                                 iters_final, snake_iters_final, device, timeout=True)
            if patience >= 5:
                break

    # -------- Finalize at full resolution --------
    return _finalize(filename, raw, work_full, best, mode, output_dir,
                     iters_final, snake_iters_final, device, timeout=False)

def _finalize(filename, raw, full_img_u8, best, mode, output_dir,
              iters_final, snake_iters_final, device, timeout: bool):
    if best is None:
        return {"filename": filename, "status": "no_candidates"}

    # full-res GPU gradients + GPU energy
    _, gx_f, gy_f = gpu_gradients(full_img_u8, device=device)

    params = dict(best[1]); params["iters"] = iters_final
    energy = gpu_energy_map(full_img_u8, gx_f, gy_f,
                            iterations=params["iters"],
                            diffusion_rate=params["diffusion_rate"],
                            mu=params["mu"], lam=params["lambda"],
                            edge_thresh=int(params["energy_threshold"]),
                            device=device)

    # CPU snakes on full-res energy
    res_final = evaluate_params_cpu_seg(full_img_u8, energy, snake_iters_final,
                                        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
                                        l_size=5, u_size=int(0.25 * full_img_u8.size),
                                        energy_threshold=int(params["energy_threshold"]))

    # write outputs
    base = os.path.splitext(filename)[0]
    ensure_dir(output_dir)
    overlay16 = _prepare_overlay16(raw)

    seg_bin = np.zeros_like(full_img_u8, dtype=np.uint8)
    # reconstruct mask for drawing (uses same energy & params)
    # We already computed everything; re-run segmentation to build mask:
    # (reuse evaluate routine lightly)
    h, w = full_img_u8.shape
    labeled = label(energy > int(params["energy_threshold"]))
    props = regionprops(labeled)
    out_mask = np.zeros((h, w), dtype=bool)
    for rgn in props:
        if not (5 < rgn.area < int(0.25*h*w)):
            continue
        minr, minc, maxr, maxc = rgn.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5:
            continue
        crop_u8 = full_img_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop_u8), 3)
        s = np.linspace(0, 2*np.pi, 200, endpoint=False)
        rr = (maxr - minr) / 2.0
        cc = (maxc - minc) / 2.0
        init = np.vstack([rr * np.sin(s) + rr, cc * np.cos(s) + cc]).T
        try:
            snake = active_contour(crop_f, init,
                                   alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
                                   max_num_iter=snake_iters_final)
        except TypeError:
            snake = active_contour(crop_f, init,
                                   alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
                                   max_iterations=snake_iters_final)
        snake_i = np.round(snake).astype(int)
        snake_i[:, 0] = np.clip(snake_i[:, 0], 0, maxr - minr - 1)
        snake_i[:, 1] = np.clip(snake_i[:, 1], 0, maxc - minc - 1)
        pr, pc = snake_i[:, 0], snake_i[:, 1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop_u8.shape)
        out_mask[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    seg_u8 = (out_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_overlay16 = overlay16.copy()
    cv2.drawContours(out_overlay16, contours, -1, PINK_16, thickness=2)
    mask16 = (out_mask.astype(np.uint16)) * 65535

    tag = f"{mode}_opt_mu{fmt(params['mu'])}_lam{fmt(params['lambda'])}_d{fmt(params['diffusion_rate'])}" \
          f"_a{fmt(params['alpha'])}_b{fmt(params['beta'])}_g{fmt(params['gamma'])}_e{int(params['energy_threshold'])}"

    overlay_path = os.path.join(output_dir, f"{base}__{tag}__overlay16.tif")
    mask_path    = os.path.join(output_dir, f"{base}__{tag}__mask16.tif")
    cv2.imwrite(overlay_path, out_overlay16)
    cv2.imwrite(mask_path,    mask16)

    return {
        "filename": filename,
        "status": "timeout" if timeout else "ok",
        "mode": mode,
        "params": params,
        "score": res_final.score,
        "n_contours": res_final.n_contours,
        "area_px": res_final.area_px,
        "area_frac": res_final.area_frac,
        "small_count": res_final.small_count,
        "overlay16_path": overlay_path,
        "mask16_path": mask_path,
    }

# ----------------------------- CLI / Driver ----------------------------------
def main():
    ap = argparse.ArgumentParser(description="GPU-accelerated per-image parameter optimization for Navier+Snake (bestch | fused).")
    ap.add_argument("--input",  required=True, help="Input directory of images")
    ap.add_argument("--output", required=True, help="Output directory for masks/overlays")
    ap.add_argument("--mode", choices=["bestch","fused"], default="bestch")

    ap.add_argument("--budget-random", type=int, default=64, help="Random candidates")
    ap.add_argument("--topk-refine", type=int, default=4, help="Top seeds to locally refine")
    ap.add_argument("--refine-steps", type=int, default=12, help="Steps per seed")

    ap.add_argument("--iters-coarse", type=int, default=16, help="Energy iterations during search")
    ap.add_argument("--snake-iters-coarse", type=int, default=60, help="Snake iterations during search")
    ap.add_argument("--iters-final", type=int, default=30, help="Energy iterations for final full-res")
    ap.add_argument("--snake-iters-final", type=int, default=100, help="Snake iterations for final full-res")

    ap.add_argument("--downscale", type=float, default=0.75, help="Search resolution scale (0,1]; final is full-res)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of images to process")

    ap.add_argument("--workers", type=int, default=None, help="ProcessPool size (default: CPU-1). Use 1 when --device cuda unless you shard GPUs.")
    ap.add_argument("--per-image-seconds", type=int, default=None, help="Watchdog timeout per image")

    ap.add_argument("--log", default="opt_summary.csv", help="CSV summary filename")
    ap.add_argument("--seed", type=int, default=1337, help="Global RNG seed")

    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="Where to run diffusion/gradients")
    ap.add_argument("--gpu-batch", type=int, default=8, help="How many random candidates to evaluate per GPU batch")

    args = ap.parse_args()

    # Avoid oversubscription of BLAS in CPU parts
    for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but no CUDA visible; falling back to CPU.", flush=True)
        args.device = "cpu"

    input_dir = args.input
    output_dir = args.output
    ensure_dir(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(ALLOWED_EXT)]
    files.sort()
    total = len(files)
    if args.limit is not None:
        files = files[:args.limit]

    # IMPORTANT: GPUs + multiprocessing can contend for the same device.
    # Prefer workers=1 per GPU process unless you shard with CUDA_VISIBLE_DEVICES.
    max_workers = args.workers if args.workers else (1 if args.device=="cuda" else max(1, (os.cpu_count() or 2) - 1))

    print("========== nv_optimize_robust_v3_gpu ==========")
    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Mode:     {args.mode}")
    print(f"Images:   {len(files)} / {total} total")
    print(f"Workers:  {max_workers}")
    print(f"Device:   {args.device}   (gpu-batch={args.gpu_batch})")
    print(f"Search:   rand={args.budget_random}, topk={args.topk_refine}, steps={args.refine_steps}, downscale={args.downscale}")
    print(f"Coarse:   iters={args.iters_coarse}, snake={args.snake_iters_coarse}")
    print(f"Final:    iters={args.iters_final}, snake={args.snake_iters_final}")
    print(f"Timeout:  {args.per_image_seonds}s per image" if args.per_image_seconds else "Timeout:  none")
    print("===========================================", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for fn in files:
            fut = ex.submit(
                process_one_image, fn, input_dir, output_dir, args.mode,
                args.budget_random, args.topk_refine, args.refine_steps,
                args.iters_coarse, args.snake_iters_coarse,
                args.iters_final, args.snake_iters_final,
                args.downscale, args.per_image_seconds, args.seed,
                args.device, args.gpu_batch
            )
            futs.append((fn, fut))

        for i, (fn, fut) in enumerate(futs, 1):
            try:
                out = fut.result()
            except Exception as e:
                out = {"filename": fn, "status": f"worker_error:{e}"}
            rows.append(out)
            print(f"[{i}/{len(futs)}] {fn}: {out.get('status','?')}  score={out.get('score','-')}", flush=True)

    # Write summary CSV
    log_path = os.path.join(output_dir, args.log)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "filename","status","mode","score","n_contours","area_px","area_frac","small_count",
                "mu","lambda","diffusion_rate","alpha","beta","gamma","energy_threshold",
                "overlay16_path","mask16_path"
            ])
        for r in rows:
            p = r.get("params", {})
            w.writerow([
                r.get("filename",""),
                r.get("status",""),
                r.get("mode",""),
                r.get("score",""),
                r.get("n_contours",""),
                r.get("area_px",""),
                r.get("area_frac",""),
                r.get("small_count",""),
                p.get("mu",""),
                p.get("lambda",""),
                p.get("diffusion_rate",""),
                p.get("alpha",""),
                p.get("beta",""),
                p.get("gamma",""),
                p.get("energy_threshold",""),
                r.get("overlay16_path",""),
                r.get("mask16_path",""),
            ])

    print(f"[DONE] wrote summary -> {log_path}")

if __name__ == "__main__":
    main()
