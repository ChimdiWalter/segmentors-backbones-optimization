#!/usr/bin/env python3
# nv_optimize_irregular_gpu.py
# Irregular lesion optimizer (Edge Alignment + Local Contrast) with GPU-accelerated
# Navier-style diffusion energy (PyTorch CUDA). Falls back to CPU if needed.

import os, sys, csv, time, math, json, random, argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# ----------------------------- Optional Torch -----------------------------
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None

# ----------------------------- Constants / Config -----------------------------
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# 16-bit pink (B, G, R) for overlay
PINK_16 = (65535, 5140, 37779)

# Parameter Search Bounds
BOUNDS = {
    "mu":              (0.03, 0.80),
    "lambda":          (0.03, 0.80),
    "diffusion_rate":  (0.03, 0.60),
    "alpha":           (0.05, 0.70),
    "beta":            (0.05, 1.20),
    "gamma":           (0.01, 0.60),
    "energy_threshold": (15, 160),  # integer; widened a bit
}

# --- SCORING WEIGHTS (Optimized for Irregular Shapes) ---
W_EDGE_ALIGN = 5.0    # Outline must match image edges
W_CONTRAST   = 3.0    # Lesion must be distinct from local background
W_SMALL      = 1.0    # Penalty for tiny specks
SMALL_MIN_AREA = 25   # px

# Soft Area Constraints (wide accept range)
MIN_AREA_FRAC_SOFT = 0.001  # 0.1% of image
MAX_AREA_FRAC_SOFT = 0.50   # 50% of image
MAX_AREA_FRAC_HARD = 0.95   # Hard reject if almost whole image

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

# ----------------------------- GPU kernels ------------------------------------

def _torch_device_choice(user_choice: str) -> Tuple[str, bool]:
    """
    Returns (device_str, using_cuda)
    """
    if torch is None:
        return ("cpu", False)
    if user_choice == "cpu":
        return ("cpu", False)
    if user_choice == "cuda":
        if torch.cuda.is_available():
            return ("cuda", True)
        else:
            return ("cpu", False)
    # auto
    if torch.cuda.is_available():
        return ("cuda", True)
    return ("cpu", False)

def _to_torch_u8(img_u8: np.ndarray, device: str):
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0)  # HxW float in [0,1]
    t = t.unsqueeze(0).unsqueeze(0).to(device)                # 1x1xHxW
    return t

def _sobel_kernels(device: str, dtype=torch.float32):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=dtype, device=device) / 8.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=dtype, device=device) / 8.0
    kx = kx.view(1,1,3,3)
    ky = ky.view(1,1,3,3)
    return kx, ky

@torch.no_grad()
def gradient_mag_torch(img_u8: np.ndarray, device: str):
    """Return (mag_u8_np, gx_np, gy_np) using torch conv2d."""
    t = _to_torch_u8(img_u8, device)
    kx, ky = _sobel_kernels(device, t.dtype)
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy)
    # normalize to uint8
    mmax = torch.clamp(mag.max(), min=1e-8)
    mag_u8 = (mag / mmax * 255.0).squeeze().detach().cpu().numpy().astype(np.uint8)
    gx_np = gx.squeeze().detach().cpu().numpy().astype(np.float64)
    gy_np = gy.squeeze().detach().cpu().numpy().astype(np.float64)
    return mag_u8, gx_np, gy_np

@torch.no_grad()
def navier_energy_torch(
    img_u8: np.ndarray,
    base_gx_np: np.ndarray,
    base_gy_np: np.ndarray,
    iterations: int,
    diffusion_rate: float,
    mu: float,
    lambda_param: float,
    edge_thresh: int,
    device: str
) -> np.ndarray:
    """
    GPU version of the iterative diffusion used to produce the 'energy' map.
    Inputs are numpy; computation is torch; output is uint8 numpy HxW.
    """
    # tensors
    img = _to_torch_u8(img_u8, device)        # 1x1xHxW in [0,1]
    H, W = img.shape[-2:]

    # precompute Sobel on image
    kx, ky = _sobel_kernels(device, img.dtype)
    im_gx = F.conv2d(img, kx, padding=1)
    im_gy = F.conv2d(img, ky, padding=1)
    im_mag = torch.sqrt(im_gx*im_gx + im_gy*im_gy)

    # base grads to device
    bgx = torch.from_numpy(base_gx_np).to(device=device, dtype=img.dtype).view(1,1,H,W)
    bgy = torch.from_numpy(base_gy_np).to(device=device, dtype=img.dtype).view(1,1,H,W)

    # init diffused = ||base_grad||
    diffused = torch.sqrt(bgx*bgx + bgy*bgy)

    # edge gate
    q = (im_mag > (float(edge_thresh)/255.0)).to(img.dtype)

    # Laplacian kernel
    lap_k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=img.dtype, device=device).view(1,1,3,3)

    # crude divergence proxy: grad along one axis
    # We'll approximate div(v) ≈ d/dx (diffused)
    for _ in range(int(iterations)):
        lap = F.conv2d(diffused, lap_k, padding=1)
        # gradient along x (using Sobel kx)
        div_v = F.conv2d(diffused, kx, padding=1)

        ftx = im_gx - bgx
        fty = im_gy - bgy
        ftm = torch.sqrt(ftx*ftx + fty*fty)
        diffused = diffused + diffusion_rate * (mu * lap + (lambda_param + mu) * div_v + q * (ftm - diffused))

    # normalize to uint8
    mmax = torch.clamp(diffused.max(), min=1e-8)
    out = (diffused / mmax * 255.0).squeeze().detach().cpu().numpy().astype(np.uint8)
    return out

# ----------------------------- Snakes & Objective -----------------------------
def snake_seg(image_u8, energy_u8, its, alpha, beta, gamma, l_size, u_size, energy_threshold):
    h, w = image_u8.shape
    labeled = label(energy_u8 > energy_threshold)
    props = regionprops(labeled)
    out_mask = np.zeros((h, w), dtype=bool)

    for rgn in props:
        if not (l_size < rgn.area < u_size):
            continue
        minr, minc, maxr, maxc = rgn.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5:
            continue

        crop_u8 = image_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop_u8), 3)

        # 200-point circular init
        s = np.linspace(0, 2*np.pi, 200, endpoint=False)
        rr = (maxr - minr) / 2.0
        cc = (maxc - minc) / 2.0
        init = np.vstack([rr * np.sin(s) + rr, cc * np.cos(s) + cc]).T

        try:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_num_iter=its
            )
        except TypeError:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_iterations=its
            )

        snake_i = np.round(snake).astype(int)
        snake_i[:, 0] = np.clip(snake_i[:, 0], 0, maxr - minr - 1)
        snake_i[:, 1] = np.clip(snake_i[:, 1], 0, maxc - minc - 1)

        pr, pc = snake_i[:, 0], snake_i[:, 1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop_u8.shape)
        out_mask[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    return out_mask

@dataclass
class EvalResult:
    score: float
    n_contours: int
    area_px: int
    area_frac: float
    small_count: int
    edge_score: float
    contrast_score: float

def evaluate_params_irregular(
    img_u8: np.ndarray,
    base_gx: np.ndarray,
    base_gy: np.ndarray,
    its_snake: int,
    params: Dict,
    l_size: int,
    u_size: int,
    device: str,
    use_gpu: bool
) -> EvalResult:
    """
    Objective: Edge Alignment + Local Contrast (no compactness prior).
    Uses GPU for the energy step when available.
    """
    h, w = img_u8.shape
    total_pixels = h * w

    mu = float(params["mu"])
    lam = float(params["lambda"])
    dr = float(params["diffusion_rate"])
    alpha = float(params["alpha"])
    beta  = float(params["beta"])
    gamma = float(params["gamma"])
    ethr  = int(params["energy_threshold"])
    iters = int(params.get("iters", 16))

    # Energy (GPU or CPU)
    if use_gpu and torch is not None:
        energy = navier_energy_torch(
            img_u8, base_gx, base_gy, iters, dr, mu, lam, ethr, device
        )
    else:
        # CPU fallback (original)
        diffused = np.sqrt(base_gx**2 + base_gy**2).astype(np.float64)
        im_gx = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
        im_gy = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
        im_mag = np.sqrt(im_gx**2 + im_gy**2)
        q = (im_mag > ethr).astype(np.float64)  # note: CPU uses raw threshold in [0,255]

        for _ in range(iters):
            lap = laplace(diffused)
            div_v = np.gradient(diffused)[0]
            ftx = im_gx - base_gx
            fty = im_gy - base_gy
            ftm = np.sqrt(ftx**2 + fty**2)
            diffused += dr * (mu * lap + (lam + mu) * div_v + q * (ftm - diffused))
        eps = 1e-8
        energy = np.uint8(255 * diffused / max(np.max(diffused), eps))

    # Segmentation
    seg_mask_bool = snake_seg(
        img_u8, energy, its=its_snake,
        alpha=alpha, beta=beta, gamma=gamma,
        l_size=l_size, u_size=u_size, energy_threshold=ethr
    )

    # Basic props
    area_px = int(np.sum(seg_mask_bool))
    area_frac = area_px / float(total_pixels + 1e-8)

    if area_px < 5 or area_frac > MAX_AREA_FRAC_HARD:
        return EvalResult(-1e9, 0, area_px, area_frac, 0, 0.0, 0.0)

    seg_u8 = (seg_mask_bool.astype(np.uint8)) * 255

    # Edge Alignment
    base_mag_u8, bgx_vis, bgy_vis = gradient_magnitude_cpu(img_u8)  # for mask-weighted mean
    # perimeter 1px
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    contour_edge = cv2.morphologyEx(seg_u8, cv2.MORPH_GRADIENT, kernel)
    edge_pixels = np.sum(contour_edge > 0)
    avg_edge_strength = 0.0
    if edge_pixels > 0:
        avg_edge_strength = cv2.mean(base_mag_u8, mask=contour_edge)[0] / 255.0

    # Local Contrast: inside vs thin outside ring
    dilated = cv2.dilate(seg_u8, kernel, iterations=3)
    ring_mask = cv2.subtract(dilated, seg_u8)
    mean_in = cv2.mean(img_u8, mask=seg_u8)[0]
    mean_ring = cv2.mean(img_u8, mask=ring_mask)[0] if cv2.countNonZero(ring_mask) > 0 else mean_in
    contrast_score = abs(mean_in - mean_ring) / 255.0

    # noise / comps
    label_img = label(seg_mask_bool)
    props = regionprops(label_img)
    n_contours = len(props)
    small_count = sum(1 for p in props if p.area < SMALL_MIN_AREA)

    # composite
    score = 0.0
    score += W_EDGE_ALIGN * avg_edge_strength
    score += W_CONTRAST   * contrast_score
    score -= W_SMALL      * small_count

    # soft area penalties
    if area_frac < MIN_AREA_FRAC_SOFT:
        score -= 5.0
    elif area_frac > MAX_AREA_FRAC_SOFT:
        score -= 5.0

    return EvalResult(score, n_contours, area_px, area_frac, small_count, avg_edge_strength, contrast_score)

# ----------------------------- Search Utils -----------------------------------
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

# ----------------------------- Per-image Worker (single-thread) --------------
def process_one_image(
    filename: str,
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
    use_gpu: bool
) -> Dict:

    rng = random.Random((hash(filename) ^ seed) & 0xFFFFFFFF)

    in_path = os.path.join(input_dir, filename)
    raw, bgr8_full, gray8_full = read_color_anydepth(in_path)
    if raw is None:
        return {"filename": filename, "status": "load_error"}

    if mode == "bestch":
        b8, g8, r8 = cv2.split(bgr8_full)
        leaf_mask = binary_threshold_mask(gray8_full, threshold=127)
        work_full = apply_mask(g8, leaf_mask)
    elif mode == "fused":
        leaf_mask = binary_threshold_mask(gray8_full, threshold=127)
        work_full = fused_channel_u8(bgr8_full, leaf_mask)
    else:
        return {"filename": filename, "status": f"bad_mode_{mode}"}

    # Downscale for search
    if 0.0 < downscale < 1.0:
        work = cv2.resize(work_full, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    else:
        work = work_full

    # Gradients once (base)
    if use_gpu and torch is not None:
        _, gx, gy = gradient_mag_torch(work, device)
    else:
        _, gx, gy = gradient_magnitude_cpu(work)

    h, w = work.shape
    u_size = int(0.40 * h * w)

    t0 = time.time()
    best: Optional[Tuple[EvalResult, Dict]] = None
    tried = 0

    # -------- Random exploration --------
    while tried < budget_random:
        cand = sample_candidate(rng)
        cand["iters"] = iters_coarse
        res = evaluate_params_irregular(work, gx, gy, snake_iters_coarse, cand, l_size=5, u_size=u_size, device=device, use_gpu=use_gpu)
        tried += 1
        if (best is None) or (res.score > best[0].score):
            best = (res, cand)
        if per_image_seconds and (time.time() - t0 > per_image_seconds):
            return _finalize_irregular(filename, raw, work_full, best, mode, output_dir, iters_final, snake_iters_final, device, use_gpu, timeout=True)

    # seeds around best
    seeds = [best[1]]
    for _ in range(max(0, topk_refine - 1)):
        seeds.append(perturb(rng, best[1], scale=0.25))

    # -------- Local refinement --------
    for seed_p in seeds:
        current = (best[0], dict(seed_p))
        patience = 0
        for step in range(refine_steps):
            scale = 0.25 * (0.85 ** step)
            cand = perturb(rng, current[1], scale=scale)
            cand["iters"] = iters_coarse
            res = evaluate_params_irregular(work, gx, gy, snake_iters_coarse, cand, l_size=5, u_size=u_size, device=device, use_gpu=use_gpu)
            tried += 1
            if res.score > current[0].score + 1e-6:
                current = (res, cand)
                patience = 0
                if res.score > best[0].score + 1e-6:
                    best = (res, cand)
            else:
                patience += 1
            if per_image_seconds and (time.time() - t0 > per_image_seconds):
                return _finalize_irregular(filename, raw, work_full, best, mode, output_dir, iters_final, snake_iters_final, device, use_gpu, timeout=True)
            if patience >= 5:
                break

    return _finalize_irregular(filename, raw, work_full, best, mode, output_dir, iters_final, snake_iters_final, device, use_gpu, timeout=False)

def _finalize_irregular(
    filename, raw, full_img_u8, best, mode, output_dir, iters_final, snake_iters_final, device, use_gpu, timeout: bool
):
    if best is None:
        return {"filename": filename, "status": "no_candidates"}

    # full-res base grads
    if use_gpu and torch is not None:
        _, gx_f, gy_f = gradient_mag_torch(full_img_u8, device)
    else:
        _, gx_f, gy_f = gradient_magnitude_cpu(full_img_u8)

    params = dict(best[1])
    params["iters"] = iters_final
    res_final = evaluate_params_irregular(
        full_img_u8, gx_f, gy_f, snake_iters_final, params,
        l_size=5, u_size=int(0.40 * full_img_u8.size), device=device, use_gpu=use_gpu
    )

    # write overlays
    base = os.path.splitext(filename)[0]
    ensure_dir(output_dir)
    overlay16 = _prepare_overlay16(raw)

    # regenerate final mask (already computed inside evaluate, but we need contours)
    # compute energy again to get mask for drawing:
    if use_gpu and torch is not None:
        energy = navier_energy_torch(
            full_img_u8, gx_f, gy_f, params["iters"], params["diffusion_rate"],
            params["mu"], params["lambda"], int(params["energy_threshold"]), device
        )
    else:
        # CPU fallback (duplicated for correctness)
        diffused = np.sqrt(gx_f**2 + gy_f**2).astype(np.float64)
        im_gx = cv2.Sobel(full_img_u8, cv2.CV_64F, 1, 0, ksize=3)
        im_gy = cv2.Sobel(full_img_u8, cv2.CV_64F, 0, 1, ksize=3)
        im_mag = np.sqrt(im_gx**2 + im_gy**2)
        q = (im_mag > int(params["energy_threshold"])).astype(np.float64)
        for _ in range(int(params["iters"])):
            lap = laplace(diffused)
            div_v = np.gradient(diffused)[0]
            ftx = im_gx - gx_f
            fty = im_gy - gy_f
            ftm = np.sqrt(ftx**2 + fty**2)
            diffused += params["diffusion_rate"] * (params["mu"] * lap + (params["lambda"] + params["mu"]) * div_v + q * (ftm - diffused))
        eps = 1e-8
        energy = np.uint8(255 * diffused / max(np.max(diffused), eps))

    seg_mask_bool = snake_seg(
        full_img_u8, energy, its=snake_iters_final,
        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
        l_size=5, u_size=int(0.40 * full_img_u8.size), energy_threshold=int(params["energy_threshold"])
    )
    seg_u8 = (seg_mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_overlay16 = overlay16.copy()
    cv2.drawContours(out_overlay16, contours, -1, PINK_16, thickness=2)
    mask16 = (seg_mask_bool.astype(np.uint16)) * 65535

    tag = f"{mode}_irr_score{int(res_final.score)}"
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
        "edge_score": res_final.edge_score,
        "contrast_score": res_final.contrast_score,
        "n_contours": res_final.n_contours,
        "area_px": res_final.area_px,
        "area_frac": res_final.area_frac,
        "overlay16_path": overlay_path,
        "mask16_path": mask_path,
    }

# ----------------------------- CLI / Driver ----------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Irregular Lesion Optimizer (Edge+Contrast) with GPU energy")
    ap.add_argument("--input",  required=True, help="Input directory of images")
    ap.add_argument("--output", required=True, help="Output directory for masks/overlays")
    ap.add_argument("--mode", choices=["bestch","fused"], default="bestch")

    ap.add_argument("--budget-random", type=int, default=64)
    ap.add_argument("--topk-refine", type=int, default=4)
    ap.add_argument("--refine-steps", type=int, default=12)

    ap.add_argument("--iters-coarse", type=int, default=16)
    ap.add_argument("--snake-iters-coarse", type=int, default=70)
    ap.add_argument("--iters-final", type=int, default=30)
    ap.add_argument("--snake-iters-final", type=int, default=120)

    ap.add_argument("--downscale", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1, help="Forced to 1 on CUDA for stability")
    ap.add_argument("--per-image-seconds", type=int, default=240)
    ap.add_argument("--log", default="opt_summary.csv")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto",
                    help="Where to run energy step (CUDA accelerates); snakes remain CPU")

    args = ap.parse_args()

    # Avoid oversubscription
    for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # Device setup
    device, use_cuda = _torch_device_choice(args.device)
    use_gpu = bool(use_cuda and (torch is not None))
    if args.device == "cuda" and not use_gpu:
        print("[warn] --device cuda requested but CUDA not available; falling back to CPU.", flush=True)

    # Force workers=1 on GPU (CUDA context safety)
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

    print("========== nv_optimize_irregular_gpu ==========")
    print(f"Device:   {device}  (GPU energy={'yes' if use_gpu else 'no'})")
    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Mode:     {args.mode}")
    print(f"Images:   {len(files)} / {total} total")
    print(f"Workers:  {max_workers}")
    print(f"Search:   rand={args.budget_random}, topk={args.topk_refine}, steps={args.refine_steps}, downscale={args.downscale}")
    print(f"Coarse:   iters={args.iters_coarse}, snake={args.snake_iters_coarse}")
    print(f"Final:    iters={args.iters_final}, snake={args.snake_iters_final}")
    print(f"Timeout:  {args.per_image_seconds}s per image" if args.per_image_seconds else "Timeout: none")
    print("===============================================", flush=True)

    rows = []
    # Sequential loop (GPU-friendly). If you need CPU parallelism, launch multiple processes via SLURM.
    for i, fn in enumerate(files, 1):
        t0 = time.time()
        try:
            out = process_one_image(
                fn, input_dir, output_dir, args.mode,
                args.budget_random, args.topk_refine, args.refine_steps,
                args.iters_coarse, args.snake_iters_coarse,
                args.iters_final, args.snake_iters_final,
                args.downscale, args.per_image_seconds, args.seed,
                device, use_gpu
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
            "filename","status","mode","score","edge_score","contrast_score","n_contours","area_px","area_frac",
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
