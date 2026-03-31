#!/usr/bin/env python3
# nv_optimize_wysiwyg.py
# -------------------------------------------------------------------
# "WHAT YOU SEE IS WHAT YOU GET"
# 1. VISUALIZATION: Pink Snake Outline.
# 2. MASK: The exact area INSIDE that Pink Snake (Filled White).
# 3. CSV: Saves all parameters (mu, alpha...) for Neural Net.
# -------------------------------------------------------------------

import os, sys, csv, time, math, json, random, argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# ================= CONFIGURATION =================
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# Color: Light Pink (Magenta + Green mix)
COLOR_INNER = (65535, 32000, 65535)

# Search Bounds
BOUNDS = {
    "mu":              (0.05, 0.60),
    "lambda":          (0.05, 0.60),
    "diffusion_rate":  (0.05, 0.35),
    "alpha":           (0.001, 0.05),
    "beta":            (0.01, 0.25),
    "gamma":           (0.10, 0.60),
    "energy_threshold": (5, 80),      # Wide range to catch both faint and dark
}

# Objectives
TARGET_AREA_FRAC    = 0.05      
AREA_FRAC_TOL       = 0.05      
W_AREA              = 0.05      
W_SMALL_BASE        = 0.1       
SMALL_MIN_AREA      = 16        
MAX_AREA_FRAC_HARD  = 0.85      
MIN_AREA_FRAC_HARD  = 1e-7      

W_GRAD_ALIGN        = 0.5       
W_COLOR_DIST        = 0.0       
W_CONTOUR           = 2.0       
W_OUTLIER           = 0.0       

# ================= UTILITIES =================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def fmt(v):
    if isinstance(v, float): return str(v).replace('.', 'p')
    return str(v)

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None: return None
    if img.dtype == np.uint8: return img
    if img.dtype == np.uint16: return cv2.convertScaleAbs(img, alpha=255.0 / 65535.0)
    if img.dtype in (np.float32, np.float64):
        im = np.clip(img, 0.0, 1.0)
        return (im * 255.0).astype(np.uint8)
    imin, imax = float(np.min(img)), float(np.max(img))
    if imax <= imin + 1e-12: return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * (img - imin) / (imax - imin)).astype(np.uint8)

def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8: return (img.astype(np.float32)) / 255.0
    if img.dtype == np.uint16: return (img.astype(np.float32)) / 65535.0
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx <= mn + 1e-12: return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)

def read_color_anydepth(path: str):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None: return None, None, None
    if raw.ndim == 2:
        raw_gray = raw
        bgr8 = _to_uint8(raw_gray)
        if bgr8.ndim == 2: bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_GRAY2BGR)
        gray8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        return raw_gray, bgr8, gray8
    else:
        bgr8 = _to_uint8(raw)
        if bgr8.ndim == 2: bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_GRAY2BGR)
        gray8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        return raw, bgr8, gray8

def binary_threshold_mask(gray8: np.ndarray, threshold: int = 127) -> np.ndarray:
    _, mask = cv2.threshold(gray8, threshold, 255, cv2.THRESH_BINARY)
    return mask

def apply_mask(image_u8: np.ndarray, mask_u8: Optional[np.ndarray]) -> np.ndarray:
    if mask_u8 is None: return image_u8
    return cv2.bitwise_and(image_u8, image_u8, mask=mask_u8)

def compute_image_gradients(u8: np.ndarray):
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, mag

def build_seed_from_image_gradients(img_grad_x, img_grad_y, sigma=1.5):
    seed_gx = cv2.GaussianBlur(img_grad_x, (0, 0), sigmaX=sigma, sigmaY=sigma)
    seed_gy = cv2.GaussianBlur(img_grad_y, (0, 0), sigmaX=sigma, sigmaY=sigma)
    seed_mag = np.sqrt(seed_gx ** 2 + seed_gy ** 2)
    return seed_gx, seed_gy, seed_mag

def elastic_deformation_diffusion_refined(seed_mag, img_mag, img_grad_x, img_grad_y, seed_grad_x, seed_grad_y, iterations, diffusion_rate, mu, lambda_param, edge_thresh):
    diffused = seed_mag.astype(np.float64).copy()
    edge_mask = (img_mag > edge_thresh).astype(np.float64)
    for _ in range(iterations):
        lap_img = laplace(diffused)
        grad_y, grad_x = np.gradient(diffused)
        div_v = grad_x + grad_y
        ftx = img_grad_x - seed_grad_x
        fty = img_grad_y - seed_grad_y
        ftm = np.sqrt(ftx ** 2 + fty ** 2)
        diffused += diffusion_rate * (mu * lap_img + (lambda_param + mu) * div_v + edge_mask * (ftm - diffused))
    diffused = np.maximum(diffused, 0.0)
    mx = float(diffused.max())
    if mx < 1e-8: return np.zeros_like(diffused, dtype=np.uint8)
    return np.uint8(255.0 * diffused / mx)

def snake_seg(image_u8, energy_u8, its, alpha, beta, gamma, min_blob_frac, max_blob_frac, energy_threshold):
    h, w = image_u8.shape
    img_area = h * w
    l_size = max(SMALL_MIN_AREA, int(min_blob_frac * img_area))
    u_size = int(max_blob_frac * img_area)
    labeled = label(energy_u8 > energy_threshold)
    props = regionprops(labeled)
    out_mask = np.zeros((h, w), dtype=bool)
    for rgn in props:
        if not (l_size < rgn.area < u_size): continue
        minr, minc, maxr, maxc = rgn.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5: continue
        crop_u8 = image_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop_u8), 3)
        s = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        rr = (maxr - minr) / 2.0
        cc = (maxc - minc) / 2.0
        init = np.vstack([rr * np.sin(s) + rr, cc * np.cos(s) + cc]).T
        try: snake = active_contour(image=crop_f, snake=init, alpha=alpha, beta=beta, gamma=gamma, max_num_iter=its)
        except TypeError: snake = active_contour(image=crop_f, snake=init, alpha=alpha, beta=beta, gamma=gamma, max_iterations=its)
        snake_i = np.round(snake).astype(int)
        snake_i[:, 0] = np.clip(snake_i[:, 0], 0, maxr - minr - 1)
        snake_i[:, 1] = np.clip(snake_i[:, 1], 0, maxc - minc - 1)
        pr, pc = snake_i[:, 0], snake_i[:, 1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop_u8.shape)
        out_mask[minr:maxr, minc:maxc][rr_fill, cc_fill] = True
    return out_mask

def _prepare_overlay16(raw: np.ndarray) -> np.ndarray:
    if raw.dtype == np.uint16:
        if raw.ndim == 2: return cv2.merge([raw, raw, raw])
        if raw.ndim == 3 and raw.shape[2] == 3: return raw.copy()
        if raw.ndim == 3 and raw.shape[2] == 1: g = raw[..., 0]; return cv2.merge([g, g, g])
    raw8 = raw if raw.dtype == np.uint8 else _to_uint8(raw)
    raw16 = (raw8.astype(np.uint16)) * 257
    if raw8.ndim == 2: return cv2.merge([raw16, raw16, raw16])
    return raw16

def fused_channel_u8(bgr8, binary_mask=None):
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)
    candidates = [bgr8[..., 1], gray, lab[..., 1], lab[..., 2]]
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

def grad_alignment_score(seg_mask_bool, grad_mag_u8):
    seg_u8 = (seg_mask_bool.astype(np.uint8)) * 255
    edges = cv2.Canny(seg_u8, 50, 150)
    edge_vals = grad_mag_u8[edges > 0]
    if edge_vals.size == 0: return 0.0
    edge_mean = float(edge_vals.mean()) / 255.0
    global_mean = float(grad_mag_u8.mean()) / 255.0 + 1e-6
    return edge_mean / global_mean

def color_distance_score(seg_mask_bool, img_lab, img_u8, fg_min_pixels=50):
    leaf_mask = img_u8 > 0
    fg_mask = seg_mask_bool & leaf_mask
    bg_mask = (~seg_mask_bool) & leaf_mask
    if fg_mask.sum() < fg_min_pixels or bg_mask.sum() < fg_min_pixels: return 0.0
    lesion_ab = img_lab[..., 1:3][fg_mask]
    bg_ab     = img_lab[..., 1:3][bg_mask]
    lesion_mean = lesion_ab.mean(axis=0)
    bg_mean     = bg_ab.mean(axis=0)
    dist = float(np.linalg.norm(lesion_mean - bg_mean))
    return dist / (math.sqrt(2.0) * 255.0)

@dataclass
class EvalResult:
    score: float
    n_contours: int
    area_px: int
    area_frac: float
    small_count: int
    grad_align: float
    color_dist: float

def evaluate_params_refined(img_u8, img_mag, img_grad_x, img_grad_y, seed_mag, seed_grad_x, seed_grad_y, grad_mag_u8, lab_img, its_snake, params, min_blob_frac=1e-7, max_blob_frac=0.85):
    h, w = img_u8.shape
    img_area = h * w
    mu, lam, dr = float(params["mu"]), float(params["lambda"]), float(params["diffusion_rate"])
    alpha, beta, gamma = float(params["alpha"]), float(params["beta"]), float(params["gamma"])
    ethr = int(params["energy_threshold"])

    energy = elastic_deformation_diffusion_refined(seed_mag, img_mag, img_grad_x, img_grad_y, seed_grad_x, seed_grad_y, params.get("iters", 16), dr, mu, lam, ethr)
    seg_mask_bool = snake_seg(img_u8, energy, its_snake, alpha, beta, gamma, min_blob_frac, max_blob_frac, ethr)
    
    area_px = int(np.sum(seg_mask_bool))
    area_frac = area_px / float(img_area + 1e-8)
    if area_frac > MAX_AREA_FRAC_HARD or area_frac < MIN_AREA_FRAC_HARD: return EvalResult(-1e9, 0, area_px, area_frac, 0, 0.0, 0.0)

    lab_masks = label(seg_mask_bool)
    props = regionprops(lab_masks)
    areas = [p.area for p in props]
    if len(areas) == 0: return EvalResult(-1e9, 0, area_px, area_frac, 0, 0.0, 0.0)

    areas = np.asarray(areas, dtype=np.float32)
    n_contours = int(len(areas))
    small_thresh = max(SMALL_MIN_AREA, int(1e-7 * img_area))
    small_count  = int(np.sum(areas < small_thresh))
    median_area = float(np.median(areas))
    size_norm   = areas / (median_area + 1e-8)
    outlier_count = int(np.sum((size_norm < 0.25) | (size_norm > 4.0)))
    valid_count = int(np.sum(areas >= small_thresh))

    grad_align = grad_alignment_score(seg_mask_bool, grad_mag_u8)
    color_dist = color_distance_score(seg_mask_bool, lab_img, img_u8)
    delta = abs(area_frac - TARGET_AREA_FRAC)
    pen_area = max(0.0, delta - AREA_FRAC_TOL)

    score = (W_GRAD_ALIGN * grad_align + W_COLOR_DIST * color_dist + W_CONTOUR * math.log1p(valid_count) - W_SMALL_BASE * small_count - W_OUTLIER * outlier_count - W_AREA * pen_area)
    return EvalResult(score, n_contours, area_px, area_frac, small_count, grad_align, color_dist)

def sample_candidate(rng):
    return {
        "mu": rng.uniform(*BOUNDS["mu"]), "lambda": rng.uniform(*BOUNDS["lambda"]),
        "diffusion_rate": rng.uniform(*BOUNDS["diffusion_rate"]), "alpha": rng.uniform(*BOUNDS["alpha"]),
        "beta": rng.uniform(*BOUNDS["beta"]), "gamma": rng.uniform(*BOUNDS["gamma"]),
        "energy_threshold": int(round(rng.uniform(*BOUNDS["energy_threshold"])))
    }

def clip_params(p):
    q = dict(p)
    for k, (lo, hi) in BOUNDS.items():
        if k == "energy_threshold": q[k] = int(min(max(int(round(q[k])), int(lo)), int(hi)))
        else: q[k] = float(min(max(q[k], lo), hi))
    return q

def perturb(rng, p, scale):
    q = dict(p)
    for k in ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma"]:
        span = BOUNDS[k][1] - BOUNDS[k][0]
        q[k] += rng.uniform(-1, 1) * scale * span
    q["energy_threshold"] += int(round(rng.uniform(-1, 1) * scale * (BOUNDS["energy_threshold"][1] - BOUNDS["energy_threshold"][0]) * 0.2))
    return clip_params(q)

def process_one_image(filename, input_dir, output_dir, mode, budget_random, topk_refine, refine_steps, iters_coarse, snake_iters_coarse, iters_final, snake_iters_final, downscale, per_image_seconds, seed):
    rng = random.Random((hash(filename) ^ seed) & 0xFFFFFFFF)
    in_path = os.path.join(input_dir, filename)
    raw, bgr8_full, gray8_full = read_color_anydepth(in_path)
    if raw is None: return {"filename": filename, "status": "load_error"}

    if mode == "bestch":
        b8, g8, r8 = cv2.split(bgr8_full)
        leaf_mask = binary_threshold_mask(gray8_full, threshold=127)
        work_full = apply_mask(g8, leaf_mask)
    elif mode == "fused":
        leaf_mask = binary_threshold_mask(gray8_full, threshold=127)
        work_full = fused_channel_u8(bgr8_full, leaf_mask)
    else: return {"filename": filename, "status": f"bad_mode_{mode}"}

    if 0.0 < downscale < 1.0:
        work = cv2.resize(work_full, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        bgr_work = cv2.resize(bgr8_full, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_AREA)
    else: work = work_full; bgr_work = bgr8_full

    img_gx, img_gy, img_mag = compute_image_gradients(work)
    seed_gx, seed_gy, seed_mag = build_seed_from_image_gradients(img_gx, img_gy, sigma=1.5)
    eps = 1e-8
    grad_mag_u8 = np.uint8(255.0 * img_mag / max(float(img_mag.max()), eps))
    lab_work = cv2.cvtColor(bgr_work, cv2.COLOR_BGR2Lab)

    t0 = time.time()
    best = None
    tried = 0

    while tried < budget_random:
        cand = sample_candidate(rng)
        cand["iters"] = iters_coarse
        res = evaluate_params_refined(work, img_mag, img_gx, img_gy, seed_mag, seed_gx, seed_gy, grad_mag_u8, lab_work, snake_iters_coarse, cand)
        tried += 1
        if (best is None) or (res.score > best[0].score): best = (res, cand)
        if per_image_seconds and (time.time() - t0 > per_image_seconds): return _finalize(filename, raw, bgr8_full, work_full, best, mode, output_dir, iters_final, snake_iters_final, True, time.time() - t0)

    seeds = [best[1]] if best is not None else []
    for _ in range(max(0, topk_refine - 1)):
        if best is not None: seeds.append(perturb(rng, best[1], scale=0.25))

    for s_idx, seed_p in enumerate(seeds):
        current = (best[0], dict(seed_p))
        patience = 0
        for step in range(refine_steps):
            scale = 0.25 * (0.85 ** step)
            cand = perturb(rng, current[1], scale=scale)
            cand["iters"] = iters_coarse
            res = evaluate_params_refined(work, img_mag, img_gx, img_gy, seed_mag, seed_gx, seed_gy, grad_mag_u8, lab_work, snake_iters_coarse, cand)
            tried += 1
            if res.score > current[0].score + 1e-6:
                current = (res, cand)
                patience = 0
                if best is None or res.score > best[0].score + 1e-6: best = (res, cand)
            else: patience += 1
            if per_image_seconds and (time.time() - t0 > per_image_seconds): return _finalize(filename, raw, bgr8_full, work_full, best, mode, output_dir, iters_final, snake_iters_final, True, time.time() - t0)
            if patience >= 5: break

    return _finalize(filename, raw, bgr8_full, work_full, best, mode, output_dir, iters_final, snake_iters_final, False, time.time() - t0)

# ================= FINALIZATION: WYSIWYG (PINK SNAKE = MASK) =================
def _finalize(filename, raw, bgr8_full, full_img_u8, best, mode, output_dir, iters_final, snake_iters_final, timeout, elapsed_seconds):
    if best is None: return {"filename": filename, "status": "no_candidates", "mode": mode, "overlay16_path": "", "mask16_path": ""}
    
    img_gx_f, img_gy_f, img_mag_f = compute_image_gradients(full_img_u8)
    seed_gx_f, seed_gy_f, seed_mag_f = build_seed_from_image_gradients(img_gx_f, img_gy_f, sigma=1.5)
    params = dict(best[1])
    params["iters"] = iters_final
    
    energy = elastic_deformation_diffusion_refined(seed_mag_f, img_mag_f, img_gx_f, img_gy_f, seed_gx_f, seed_gy_f, params["iters"], params["diffusion_rate"], params["mu"], params["lambda"], int(params["energy_threshold"]))
    base_thresh = int(params["energy_threshold"])
    
    # 2. GET THE SNAKE
    seg_mask_bool = snake_seg(full_img_u8, energy, snake_iters_final, params["alpha"], params["beta"], params["gamma"], 1e-7, 0.85, base_thresh)
    
    # 3. CREATE SOLID MASK (White Inside)
    mask_filled = np.zeros_like(full_img_u8, dtype=np.uint8)
    mask_filled[seg_mask_bool] = 255
    
    # 4. Save
    base = os.path.splitext(filename)[0]
    ensure_dir(output_dir)
    
    # --- OVERLAY: DRAW PINK LINE around the Snake ---
    overlay16 = _prepare_overlay16(raw)
    out_overlay16 = overlay16.copy()
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_overlay16, contours, -1, COLOR_INNER, thickness=2)
    
    # --- MASK: SAVE THE FILLED AREA ---
    mask16 = (mask_filled.astype(np.uint16)) * 257

    tag = f"{mode}_INNER_mu{fmt(params['mu'])}_b{fmt(params['beta'])}_e{base_thresh}"
    overlay_path = os.path.join(output_dir, f"{base}__{tag}__overlay16.tif")
    mask_path    = os.path.join(output_dir, f"{base}__{tag}__mask16.tif")
    cv2.imwrite(overlay_path, out_overlay16)
    cv2.imwrite(mask_path,    mask16)

    return {
        "filename": filename, "status": "timeout" if timeout else "ok", "mode": mode, "params": params,
        "score": best[0].score, "n_contours": best[0].n_contours, "area_px": int(np.sum(mask_filled > 0)),
        "overlay16_path": overlay_path, "mask16_path": mask_path, "elapsed_seconds": elapsed_seconds
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["bestch", "fused"], default="fused")
    ap.add_argument("--budget-random", type=int, default=64)
    ap.add_argument("--topk-refine", type=int, default=4)
    ap.add_argument("--refine-steps", type=int, default=12)
    ap.add_argument("--iters-coarse", type=int, default=16)
    ap.add_argument("--snake-iters-coarse", type=int, default=60)
    ap.add_argument("--iters-final", type=int, default=30)
    ap.add_argument("--snake-iters-final", type=int, default=100)
    ap.add_argument("--downscale", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--per-image-seconds", type=int, default=None)
    ap.add_argument("--log", default="opt_summary.csv")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(var, "1")
    try: cv2.setNumThreads(1)
    except: pass

    ensure_dir(args.output)
    files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(ALLOWED_EXT)])
    if args.limit: files = files[:args.limit]
    max_workers = args.workers if args.workers else max(1, (os.cpu_count() or 2) - 1)

    print(f"========== nv_optimize_wysiwyg (PINK SNAKE = WHITE MASK) ==========")
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_one_image, fn, args.input, args.output, args.mode, args.budget_random, args.topk_refine, args.refine_steps, args.iters_coarse, args.snake_iters_coarse, args.iters_final, args.snake_iters_final, args.downscale, args.per_image_seconds, args.seed) for fn in files]
        for i, fut in enumerate(futs, 1):
            try:
                out = fut.result()
                print(f"[{i}/{len(files)}] {out['filename']}: {out['status']} score={out.get('score', '-')}", flush=True)
                rows.append(out)
            except Exception as e: print(f"Error: {e}")

    log_path = os.path.join(args.output, args.log)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        # --- FIXED HEADER WITH PHYSICS PARAMS ---
        w.writerow(["filename", "status", "mode", "score", "n_contours", "area_px", 
                    "mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "energy_threshold",
                    "overlay16_path", "mask16_path", "elapsed_seconds"])
        for r in rows:
            p = r.get("params", {})
            w.writerow([
                r.get("filename"), r.get("status"), r.get("mode"), r.get("score"), r.get("n_contours"), r.get("area_px"),
                p.get("mu"), p.get("lambda"), p.get("diffusion_rate"), p.get("alpha"), p.get("beta"), p.get("gamma"), p.get("energy_threshold"),
                r.get("overlay16_path"), r.get("mask16_path"), r.get("elapsed_seconds")
            ])
    print(f"Done. Log: {log_path}")

if __name__ == "__main__":
    main()