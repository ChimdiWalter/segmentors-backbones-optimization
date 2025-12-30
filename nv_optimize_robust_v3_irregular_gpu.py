#!/usr/bin/env python3
# nv_optimize_robust_v3_irregular_gpu.py
# Irregular lesion optimizer (Edge Alignment + Local Contrast), with CUDA-batched
# Navier-style diffusion energy for *coarse* search (fast). Only the final winner
# per image runs snakes at full resolution (accurate).
#
# Usage example:
# python3 nv_optimize_robust_v3_irregular_gpu.py \
#   --input  /path/to/leaves \
#   --output /path/to/out_irregular_gpu \
#   --mode fused \
#   --device cuda --gpu-batch 8 --workers 1 \
#   --downscale 0.6 \
#   --budget-random 96 --topk-refine 6 --refine-steps 12 \
#   --iters-coarse 16 --iters-final 30 \
#   --snake-iters-final 120 \
#   --per-image-seconds 240

import os, sys, csv, time, math, random, argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# Optional Torch
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None

# ----------------------------- Constants / Config -----------------------------
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# 16-bit pink (B, G, R) for overlay
PINK_16 = (65535, 5140, 37779)

# Search bounds
BOUNDS = {
    "mu":              (0.03, 0.80),
    "lambda":          (0.03, 0.80),
    "diffusion_rate":  (0.03, 0.60),
    "alpha":           (0.05, 0.70),
    "beta":            (0.05, 1.20),
    "gamma":           (0.01, 0.60),
    "energy_threshold": (15, 160),  # int
}

# Irregular-shape scoring weights
W_EDGE_ALIGN = 5.0
W_CONTRAST   = 3.0
W_SMALL      = 1.0
SMALL_MIN_AREA = 25

# Soft/Hard area constraints
MIN_AREA_FRAC_SOFT = 0.001
MAX_AREA_FRAC_SOFT = 0.50
MAX_AREA_FRAC_HARD = 0.95

# ----------------------------- Utilities -------------------------------------
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def fmt(v):
    if isinstance(v, float):
        return str(v).replace('.', 'p')
    return str(v)

def _to_uint8(img):
    if img is None: return None
    if img.dtype == np.uint8: return img
    if img.dtype == np.uint16: return cv2.convertScaleAbs(img, alpha=255.0/65535.0)
    if img.dtype in (np.float32, np.float64):
        im = np.clip(img, 0.0, 1.0); return (im * 255.0).astype(np.uint8)
    imin, imax = float(np.min(img)), float(np.max(img))
    if imax <= imin + 1e-12: return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * (img - imin) / (imax - imin)).astype(np.uint8)

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

def gradient_magnitude_cpu(u8):
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    eps = 1e-8
    mag_u8 = np.uint8(255 * mag / max(np.max(mag), eps))
    return mag_u8, gx, gy

def _prepare_overlay16(raw):
    if raw.dtype == np.uint16:
        if raw.ndim == 2:
            return cv2.merge([raw, raw, raw])
        if raw.ndim == 3 and raw.shape[2] == 3:
            return raw.copy()
        if raw.ndim == 3 and raw.shape[2] == 1:
            g = raw[..., 0]
            return cv2.merge([g, g, g])

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

# ----------------------------- Torch helpers ----------------------------------
def _torch_device_choice(user_choice: str) -> Tuple[str, bool]:
    if torch is None:
        return ("cpu", False)
    if user_choice == "cpu":
        return ("cpu", False)
    if user_choice == "cuda":
        return ("cuda" if torch.cuda.is_available() else "cpu", torch.cuda.is_available())
    # auto
    return ("cuda", True) if torch.cuda.is_available() else ("cpu", False)

def _to_torch_u8(img_u8: np.ndarray, device: str):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0)  # HxW
    t = t.unsqueeze(0).unsqueeze(0).to(device)               # 1x1xHxW
    return t

def _sobel_kernels(device: str, dtype=torch.float32):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=dtype, device=device) / 8.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=dtype, device=device) / 8.0
    return kx.view(1,1,3,3), ky.view(1,1,3,3)

@torch.no_grad()
def gradient_mag_torch(img_u8: np.ndarray, device: str):
    t = _to_torch_u8(img_u8, device)
    kx, ky = _sobel_kernels(device, t.dtype)
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy)
    mmax = torch.clamp(mag.max(), min=1e-8)
    mag_u8 = (mag / mmax * 255.0).squeeze().detach().cpu().numpy().astype(np.uint8)
    return mag_u8, gx.squeeze().cpu().numpy().astype(np.float64), gy.squeeze().cpu().numpy().astype(np.float64)

# Depthwise conv over batch safely (fixes groups==B shape error)
def _dw_conv_per_batch(x, w, padding=1):
    """
    x: (B, 1, H, W)
    w: (B, 1, kh, kw) with groups=B
    return: (B, 1, H, W)
    """
    x_ = x.transpose(0, 1)  # (1, B, H, W)
    y_ = F.conv2d(x_, w, padding=padding, groups=x.shape[0])
    return y_.transpose(0, 1)  # (B, 1, H, W)

@torch.no_grad()
def navier_energy_torch_batched(
    img_u8: np.ndarray,
    base_gx_np: np.ndarray,
    base_gy_np: np.ndarray,
    iters: int,
    dr_vec: np.ndarray,
    mu_vec: np.ndarray,
    lam_vec: np.ndarray,
    ethr_vec: np.ndarray,
    device: str
) -> np.ndarray:
    """
    Batched GPU energy for B parameter sets on a single image.
    Returns uint8 energy maps: (B, H, W)
    """
    img = _to_torch_u8(img_u8, device)      # 1x1xHxW
    H, W = img.shape[-2:]
    B = int(len(dr_vec))

    # Sobel on image (shared)
    kx, ky = _sobel_kernels(device, img.dtype)
    im_gx = F.conv2d(img, kx, padding=1).repeat(B, 1, 1, 1)  # Bx1xHxW
    im_gy = F.conv2d(img, ky, padding=1).repeat(B, 1, 1, 1)
    im_mag = torch.sqrt(im_gx*im_gx + im_gy*im_gy)

    # base grads (shared) expanded to B
    bgx = torch.from_numpy(base_gx_np).to(device=device, dtype=img.dtype).view(1,1,H,W).repeat(B,1,1,1)
    bgy = torch.from_numpy(base_gy_np).to(device=device, dtype=img.dtype).view(1,1,H,W).repeat(B,1,1,1)

    # init diffused = ||base_grad||
    diffused = torch.sqrt(bgx*bgx + bgy*bgy)  # Bx1xHxW

    # edge gates per sample
    # thresholds are uint8-space; normalize to [0,1]
    q = (im_mag > torch.tensor(ethr_vec, device=device, dtype=img.dtype).view(B,1,1,1)/255.0).to(img.dtype)

    # Per-sample scalars as (B,1,1,1)
    mu  = torch.tensor(mu_vec,  device=device, dtype=img.dtype).view(B,1,1,1)
    lam = torch.tensor(lam_vec, device=device, dtype=img.dtype).view(B,1,1,1)
    dr  = torch.tensor(dr_vec,  device=device, dtype=img.dtype).view(B,1,1,1)

    # Depthwise kernels per-sample
    lap_k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=img.dtype, device=device).view(1,1,3,3).repeat(B,1,1,1)
    kx_dw = kx.repeat(B,1,1,1)

    for _ in range(int(iters)):
        lap  = _dw_conv_per_batch(diffused, lap_k, padding=1)         # Bx1xHxW
        div_v = _dw_conv_per_batch(diffused, kx_dw, padding=1)        # Bx1xHxW
        ftx = im_gx - bgx
        fty = im_gy - bgy
        ftm = torch.sqrt(ftx*ftx + fty*fty)
        diffused = diffused + dr * (mu * lap + (lam + mu) * div_v + q * (ftm - diffused))

    # normalize each map to uint8
    out = []
    for b in range(B):
        mmax = torch.clamp(diffused[b:b+1].max(), min=1e-8)
        e = (diffused[b:b+1] / mmax * 255.0).squeeze().detach().cpu().numpy().astype(np.uint8)
        out.append(e)
    return np.stack(out, axis=0)  # (B,H,W)

# ----------------------------- Objective (proxy) ------------------------------
@dataclass
class ProxyEval:
    score: float
    edge_score: float
    contrast_score: float
    area_frac: float
    n_components: int
    small_count: int

def _proxy_score_from_binary(img_u8: np.ndarray, mask_u8: np.ndarray) -> ProxyEval:
    """
    Compute edge alignment + local contrast from a *binary mask* (no snakes).
    """
    h, w = img_u8.shape
    total = float(h*w + 1e-8)
    area_px = int(cv2.countNonZero(mask_u8))
    area_frac = area_px / total
    if area_px < 5 or area_frac > MAX_AREA_FRAC_HARD:
        return ProxyEval(-1e9, 0.0, 0.0, area_frac, 0, 0)

    # gradient for edge strength
    mag_u8, _, _ = gradient_magnitude_cpu(img_u8)

    # perimeter 1px
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    contour = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    edge_strength = cv2.mean(mag_u8, mask=contour)[0] / 255.0 if cv2.countNonZero(contour) > 0 else 0.0

    # local contrast: inside vs thin outside ring
    dil = cv2.dilate(mask_u8, kernel, iterations=3)
    ring = cv2.subtract(dil, mask_u8)
    mean_in = cv2.mean(img_u8, mask=mask_u8)[0]
    mean_ring = cv2.mean(img_u8, mask=ring)[0] if cv2.countNonZero(ring) > 0 else mean_in
    contrast = abs(mean_in - mean_ring) / 255.0

    # components
    lab = label(mask_u8 > 0)
    props = regionprops(lab)
    n_comp = len(props)
    small_count = sum(1 for p in props if p.area < SMALL_MIN_AREA)

    score = 0.0
    score += W_EDGE_ALIGN * edge_strength
    score += W_CONTRAST   * contrast
    score -= W_SMALL      * small_count
    if area_frac < MIN_AREA_FRAC_SOFT or area_frac > MAX_AREA_FRAC_SOFT:
        score -= 5.0

    return ProxyEval(score, edge_strength, contrast, area_frac, n_comp, small_count)

# ----------------------------- Snakes (final only) ---------------------------
from skimage.segmentation import active_contour  # (import kept local above)
def snake_seg(image_u8, energy_u8, its, alpha, beta, gamma, l_size, u_size, energy_threshold):
    h, w = image_u8.shape
    labeled = label(energy_u8 > energy_threshold)
    props = regionprops(labeled)
    out_mask = np.zeros((h, w), dtype=bool)

    for rgn in props:
        if not (l_size < rgn.area < u_size): continue
        minr, minc, maxr, maxc = rgn.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5: continue

        crop = image_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop), 3)
        s = np.linspace(0, 2*np.pi, 200, endpoint=False)
        rr = (maxr - minr)/2.0
        cc = (maxc - minc)/2.0
        init = np.vstack([rr*np.sin(s) + rr, cc*np.cos(s) + cc]).T

        try:
            snk = active_contour(image=crop_f, snake=init,
                                 alpha=alpha, beta=beta, gamma=gamma,
                                 max_num_iter=its)
        except TypeError:
            snk = active_contour(image=crop_f, snake=init,
                                 alpha=alpha, beta=beta, gamma=gamma,
                                 max_iterations=its)

        si = np.round(snk).astype(int)
        si[:,0] = np.clip(si[:,0], 0, maxr-minr-1)
        si[:,1] = np.clip(si[:,1], 0, maxc-minc-1)
        pr, pc = si[:,0], si[:,1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop.shape)
        out_mask[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    return out_mask

# ----------------------------- Sampling utils --------------------------------
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
    q["energy_threshold"] += int(round(
        rng.uniform(-1,1) * scale * (BOUNDS["energy_threshold"][1] - BOUNDS["energy_threshold"][0]) * 0.2
    ))
    return clip_params(q)

# ----------------------------- Coarse (GPU batched) --------------------------
def coarse_proxy_scores_gpu(
    work: np.ndarray,
    base_gx: np.ndarray,
    base_gy: np.ndarray,
    param_batch: List[Dict],
    iters_coarse: int,
    device: str
):
    """
    Given a batch of param dicts (len = B), return ProxyEval list and
    the per-sample binary masks computed as (energy > threshold).
    """
    B = len(param_batch)
    mu_vec  = np.array([p["mu"] for p in param_batch], dtype=np.float32)
    lam_vec = np.array([p["lambda"] for p in param_batch], dtype=np.float32)
    dr_vec  = np.array([p["diffusion_rate"] for p in param_batch], dtype=np.float32)
    et_vec  = np.array([int(p["energy_threshold"]) for p in param_batch], dtype=np.int32)

    # CUDA-batched energy maps
    energies = navier_energy_torch_batched(
        work, base_gx, base_gy,
        iters=iters_coarse,
        dr_vec=dr_vec, mu_vec=mu_vec, lam_vec=lam_vec, ethr_vec=et_vec,
        device=device
    )  # (B,H,W) uint8

    results = []
    masks = []
    for b in range(B):
        thr = int(param_batch[b]["energy_threshold"])
        mask_u8 = (energies[b] > thr).astype(np.uint8) * 255
        masks.append(mask_u8)
        results.append(_proxy_score_from_binary(work, mask_u8))
    return results, masks

# ----------------------------- One-image pipeline ----------------------------
def process_one_image(
    filename: str,
    input_dir: str,
    output_dir: str,
    mode: str,
    budget_random: int,
    topk_refine: int,
    refine_steps: int,
    iters_coarse: int,
    iters_final: int,
    snake_iters_final: int,
    downscale: float,
    per_image_seconds: Optional[int],
    seed: int,
    device: str,
    use_gpu: bool,
    gpu_batch: int
) -> Dict:

    rng = random.Random((hash(filename) ^ seed) & 0xFFFFFFFF)

    in_path = os.path.join(input_dir, filename)
    raw, bgr8_full, gray8_full = read_color_anydepth(in_path)
    if raw is None:
        return {"filename": filename, "status": "load_error"}

    # channel/mode
    if mode == "bestch":
        b8, g8, r8 = cv2.split(bgr8_full)
        leaf_mask = binary_threshold_mask(gray8_full, threshold=127)
        work_full = apply_mask(g8, leaf_mask)
    elif mode == "fused":
        leaf_mask = binary_threshold_mask(gray8_full, threshold=127)
        work_full = fused_channel_u8(bgr8_full, leaf_mask)
    else:
        return {"filename": filename, "status": f"bad_mode_{mode}"}

    # downscale for search
    if 0.0 < downscale < 1.0:
        work = cv2.resize(work_full, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    else:
        work = work_full

    # base gradients once (GPU or CPU ok; we need float arrays for batched)
    if use_gpu and torch is not None:
        _, gx, gy = gradient_mag_torch(work, device)
    else:
        _, gx, gy = gradient_magnitude_cpu(work)

    h, w = work.shape
    t0 = time.time()

    # -------- Random exploration (batched) --------
    tried = 0
    all_candidates: List[Tuple[ProxyEval, Dict]] = []
    while tried < budget_random:
        # assemble a mini-batch
        B = min(gpu_batch if use_gpu else 8, budget_random - tried)
        pb = [clip_params(sample_candidate(rng)) for _ in range(B)]
        if use_gpu and torch is not None:
            proxy_list, _ = coarse_proxy_scores_gpu(work, gx, gy, pb, iters_coarse, device)
        else:
            # CPU fallback (compute energies one-by-one)
            proxy_list = []
            for p in pb:
                # cheap CPU energy (single)
                diffused = np.sqrt(gx**2 + gy**2).astype(np.float64)
                im_gx = cv2.Sobel(work, cv2.CV_64F, 1, 0, ksize=3)
                im_gy = cv2.Sobel(work, cv2.CV_64F, 0, 1, ksize=3)
                im_mag = np.sqrt(im_gx**2 + im_gy**2)
                q = (im_mag > int(p["energy_threshold"])).astype(np.float64)
                for _ in range(int(iters_coarse)):
                    lap = laplace(diffused)
                    div_v = np.gradient(diffused)[0]
                    ftx = im_gx - gx
                    fty = im_gy - gy
                    ftm = np.sqrt(ftx**2 + fty**2)
                    diffused += p["diffusion_rate"] * (p["mu"]*lap + (p["lambda"]+p["mu"])*div_v + q*(ftm - diffused))
                e = (diffused / max(np.max(diffused), 1e-8) * 255.0).astype(np.uint8)
                mask_u8 = (e > int(p["energy_threshold"])).astype(np.uint8)*255
                proxy_list.append(_proxy_score_from_binary(work, mask_u8))

        tried += B
        all_candidates.extend(list(zip(proxy_list, pb)))
        # watchdog
        if per_image_seconds and (time.time() - t0 > per_image_seconds):
            break

    # take top-K seeds by proxy score
    all_candidates.sort(key=lambda t: t[0].score, reverse=True)
    seeds = [c[1] for c in all_candidates[:max(1, topk_refine)]]
    if len(seeds) == 0:
        return {"filename": filename, "status": "no_candidates"}

    # -------- Local refinement (still proxy, batched if GPU) --------
    best_score = all_candidates[0][0].score
    best_params = dict(seeds[0])
    for s_idx in range(len(seeds)):
        current = dict(seeds[s_idx])
        patience = 0
        for step in range(refine_steps):
            # produce a small batch around "current"
            scale = 0.25 * (0.85 ** step)
            pb = [clip_params(perturb(rng, current, scale)) for _ in range(min(gpu_batch if use_gpu else 8, 16))]
            if use_gpu and torch is not None:
                proxy_list, _ = coarse_proxy_scores_gpu(work, gx, gy, pb, iters_coarse, device)
            else:
                proxy_list = []
                for p in pb:
                    diffused = np.sqrt(gx**2 + gy**2).astype(np.float64)
                    im_gx = cv2.Sobel(work, cv2.CV_64F, 1, 0, ksize=3)
                    im_gy = cv2.Sobel(work, cv2.CV_64F, 0, 1, ksize=3)
                    im_mag = np.sqrt(im_gx**2 + im_gy**2)
                    q = (im_mag > int(p["energy_threshold"])).astype(np.float64)
                    for _ in range(int(iters_coarse)):
                        lap = laplace(diffused)
                        div_v = np.gradient(diffused)[0]
                        ftx = im_gx - gx
                        fty = im_gy - gy
                        ftm = np.sqrt(ftx**2 + fty**2)
                        diffused += p["diffusion_rate"] * (p["mu"]*lap + (p["lambda"]+p["mu"])*div_v + q*(ftm - diffused))
                    e = (diffused / max(np.max(diffused), 1e-8) * 255.0).astype(np.uint8)
                    mask_u8 = (e > int(p["energy_threshold"])).astype(np.uint8)*255
                    proxy_list.append(_proxy_score_from_binary(work, mask_u8))

            # pick the best in this local batch
            idx = int(np.argmax([pe.score for pe in proxy_list]))
            cand_best = pb[idx]
            cand_score = proxy_list[idx].score
            if cand_score > best_score + 1e-6:
                best_score = cand_score
                best_params = dict(cand_best)
                current = dict(cand_best)
                patience = 0
            else:
                patience += 1

            if per_image_seconds and (time.time() - t0 > per_image_seconds):
                break
            if patience >= 5:
                break
        if per_image_seconds and (time.time() - t0 > per_image_seconds):
            break

    # -------- Finalize at full resolution with snakes (slow, but once) --------
    # full-res base grads
    if use_gpu and torch is not None:
        _, gx_f, gy_f = gradient_mag_torch(work_full, device)
        # energy at full-res (single)
        e_full = navier_energy_torch_batched(
            work_full, gx_f, gy_f, iters=iters_final,
            dr_vec=np.array([best_params["diffusion_rate"]], np.float32),
            mu_vec=np.array([best_params["mu"]], np.float32),
            lam_vec=np.array([best_params["lambda"]], np.float32),
            ethr_vec=np.array([int(best_params["energy_threshold"])], np.int32),
            device=device
        )[0]
    else:
        _, gx_f, gy_f = gradient_magnitude_cpu(work_full)
        diffused = np.sqrt(gx_f**2 + gy_f**2).astype(np.float64)
        im_gx = cv2.Sobel(work_full, cv2.CV_64F, 1, 0, ksize=3)
        im_gy = cv2.Sobel(work_full, cv2.CV_64F, 0, 1, ksize=3)
        im_mag = np.sqrt(im_gx**2 + im_gy**2)
        q = (im_mag > int(best_params["energy_threshold"])).astype(np.float64)
        for _ in range(int(iters_final)):
            lap = laplace(diffused)
            div_v = np.gradient(diffused)[0]
            ftx = im_gx - gx_f
            fty = im_gy - gy_f
            ftm = np.sqrt(ftx**2 + fty**2)
            diffused += best_params["diffusion_rate"] * (best_params["mu"]*lap + (best_params["lambda"]+best_params["mu"])*div_v + q*(ftm - diffused))
        e_full = (diffused / max(np.max(diffused), 1e-8) * 255.0).astype(np.uint8)

    # snakes
    seg_mask_bool = snake_seg(
        work_full, e_full, its=snake_iters_final,
        alpha=best_params["alpha"], beta=best_params["beta"], gamma=best_params["gamma"],
        l_size=5, u_size=int(0.40 * work_full.size), energy_threshold=int(best_params["energy_threshold"])
    )
    seg_u8 = (seg_mask_bool.astype(np.uint8))*255
    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # overlay/mask outputs
    base = os.path.splitext(filename)[0]
    ensure_dir(output_dir)
    overlay16 = _prepare_overlay16(raw)
    out_overlay16 = overlay16.copy()
    cv2.drawContours(out_overlay16, contours, -1, PINK_16, thickness=2)
    mask16 = (seg_mask_bool.astype(np.uint16)) * 65535

    tag = f"{mode}_irr_gpu_score{int(best_score)}"
    overlay_path = os.path.join(output_dir, f"{base}__{tag}__overlay16.tif")
    mask_path    = os.path.join(output_dir, f"{base}__{tag}__mask16.tif")
    cv2.imwrite(overlay_path, out_overlay16)
    cv2.imwrite(mask_path,    mask16)

    # compute final proxy on full-res for logging (fast)
    proxy_final = _proxy_score_from_binary(work_full, seg_u8)

    return {
        "filename": filename,
        "status": "ok" if (not per_image_seconds or (time.time()-t0) <= per_image_seconds) else "timeout",
        "mode": mode,
        "params": best_params,
        "score": float(best_score),
        "edge_score": float(proxy_final.edge_score),
        "contrast_score": float(proxy_final.contrast_score),
        "n_contours": int(len(contours)),
        "area_px": int(np.sum(seg_mask_bool)),
        "area_frac": float(proxy_final.area_frac),
        "overlay16_path": overlay_path,
        "mask16_path": mask_path,
    }

# ----------------------------- CLI / Driver ----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Irregular Lesion Optimizer (Edge+Contrast) with *batched* CUDA energy")
    ap.add_argument("--input",  required=True, help="Input directory of images")
    ap.add_argument("--output", required=True, help="Output directory for masks/overlays")
    ap.add_argument("--mode", choices=["bestch","fused"], default="bestch")

    ap.add_argument("--budget-random", type=int, default=64)
    ap.add_argument("--topk-refine", type=int, default=4)
    ap.add_argument("--refine-steps", type=int, default=12)

    ap.add_argument("--iters-coarse", type=int, default=16)
    ap.add_argument("--iters-final", type=int, default=30)
    ap.add_argument("--snake-iters-final", type=int, default=120)

    ap.add_argument("--downscale", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1, help="Force 1 on CUDA (single context)")
    ap.add_argument("--per-image-seconds", type=int, default=240)
    ap.add_argument("--log", default="opt_summary.csv")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto",
                    help="Energy step device (CUDA accelerates).")
    ap.add_argument("--gpu-batch", type=int, default=8, help="Candidates evaluated per GPU batch (coarse + refine)")

    args = ap.parse_args()

    # Thread oversubscription guards
    for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # Device
    device, cuda_ok = _torch_device_choice(args.device)
    use_gpu = bool(cuda_ok and (torch is not None))
    if args.device == "cuda" and not use_gpu:
        print("[warn] --device cuda requested but CUDA not available; using CPU.", flush=True)

    # Force workers=1 with GPU
    max_workers = int(args.workers or 1)
    if use_gpu and max_workers != 1:
        print("[info] Forcing --workers 1 on CUDA to avoid context issues.", flush=True)
        max_workers = 1

    input_dir = args.input
    output_dir = args.output
    ensure_dir(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(ALLOWED_EXT)]
    files.sort()
    total = len(files)
    if args.limit is not None:
        files = files[:args.limit]

    print("========== nv_optimize_robust_v3_irregular_gpu ==========")
    print(f"Device:   {device} (GPU energy={'yes' if use_gpu else 'no'})")
    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Mode:     {args.mode}")
    print(f"Images:   {len(files)} / {total} total")
    print(f"GPU batch:{args.gpu_batch}")
    print(f"Search:   rand={args.budget_random}, topk={args.topk_refine}, steps={args.refine_steps}, downscale={args.downscale}")
    print(f"Coarse:   iters={args.iters_coarse}")
    print(f"Final:    iters={args.iters_final}, snake={args.snake_iters_final}")
    print(f"Timeout:  {args.per_image_seconds}s per image" if args.per_image_seconds else "Timeout: none")
    print("===============================================", flush=True)

    rows = []
    # Sequential per-image loop (GPU-friendly). For cluster scale-out, use SLURM array or multiple nodes.
    for i, fn in enumerate(files, 1):
        t0 = time.time()
        try:
            out = process_one_image(
                fn, input_dir, output_dir, args.mode,
                args.budget_random, args.topk_refine, args.refine_steps,
                args.iters_coarse, args.iters_final, args.snake_iters_final,
                args.downscale, args.per_image_seconds, args.seed,
                device, use_gpu, args.gpu_batch
            )
        except Exception as e:
            out = {"filename": fn, "status": f"worker_error:{e}"}
        rows.append(out)
        dt = time.time() - t0
        print(f"[{i}/{len(files)}] {fn}: {out.get('status','?')}  score={out.get('score','-')}  ({dt:.1f}s)", flush=True)

    # CSV
    log_path = os.path.join(output_dir, args.log)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "filename","status","mode","score","edge_score","contrast_score",
            "n_contours","area_px","area_frac",
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
                r.get("edge_score",""),
                r.get("contrast_score",""),
                r.get("n_contours",""),
                r.get("area_px",""),
                r.get("area_frac",""),
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

    print(f"[DONE] Summary -> {log_path}")

if __name__ == "__main__":
    main()
