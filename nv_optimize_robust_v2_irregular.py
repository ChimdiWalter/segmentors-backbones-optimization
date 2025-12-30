#!/usr/bin/env python3
# nv_optimize_irregular.py
# Optimized for IRREGULAR lesions (e.g. plant diseases, spreading blobs)
# Scoring based on: Edge Alignment + Local Contrast.

import os, sys, csv, time, math, json, random, argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

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
    "energy_threshold": (15, 140),  # integer
}

# --- SCORING WEIGHTS (Optimized for Irregular Shapes) ---
# We do NOT penalize non-circular shapes.
# We Reward:
# 1. The outline sitting on a sharp gradient (Edge Alignment)
# 2. The inside looking different from the immediate outside (Local Contrast)

W_EDGE_ALIGN = 5.0    # Critical: Outline must match image edges
W_CONTRAST   = 3.0    # Critical: Lesion must be distinct from local background
W_SMALL      = 1.0    # Penalty: Discourage tiny noise specks
SMALL_MIN_AREA = 25   # px (dust threshold)

# Soft Area Constraints (We accept a wide range of sizes)
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
    if img is None:
        return None
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

def gradient_magnitude(u8):
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    eps = 1e-8
    mag_u8 = np.uint8(255 * mag / max(np.max(mag), eps))
    return mag_u8, gx, gy

def elastic_deformation_diffusion(image_u8, grad_x, grad_y,
                                  iterations=30, diffusion_rate=0.2,
                                  mu=0.5, lambda_param=0.5,
                                  edge_thresh=50):
    # Start from vector magnitude between image gradient and seed gradients
    diffused = np.sqrt(grad_x**2 + grad_y**2).astype(np.float64)

    im_gx = cv2.Sobel(image_u8, cv2.CV_64F, 1, 0, ksize=3)
    im_gy = cv2.Sobel(image_u8, cv2.CV_64F, 0, 1, ksize=3)
    im_mag = np.sqrt(im_gx**2 + im_gy**2)

    q = (im_mag > edge_thresh).astype(np.float64)  # edges gate

    for _ in range(iterations):
        lap = laplace(diffused)
        div_v = np.gradient(diffused)[0]  # coarse divergence proxy
        ftx = im_gx - grad_x
        fty = im_gy - grad_y
        ftm = np.sqrt(ftx**2 + fty**2)
        diffused += diffusion_rate * (mu * lap + (lambda_param + mu) * div_v + q * (ftm - diffused))

    eps = 1e-8
    return np.uint8(255 * diffused / max(np.max(diffused), eps))

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

# ----------------------------- Objective -------------------------------------
@dataclass
class EvalResult:
    score: float
    n_contours: int
    area_px: int
    area_frac: float
    small_count: int
    edge_score: float     # Added: how well boundaries align with gradients
    contrast_score: float # Added: how different lesion is from local bg

def evaluate_params(img_u8: np.ndarray,
                    base_gx: np.ndarray,
                    base_gy: np.ndarray,
                    its_snake: int,
                    params: Dict,
                    l_size: int = 5,
                    u_size: Optional[int] = None) -> EvalResult:
    """
    New scoring logic for IRREGULAR shapes.
    Prioritizes Edge Alignment and Local Contrast.
    """
    h, w = img_u8.shape
    total_pixels = h * w
    if u_size is None:
        u_size = int(0.40 * total_pixels) 

    mu = float(params["mu"])
    lam = float(params["lambda"])
    dr = float(params["diffusion_rate"])
    alpha = float(params["alpha"])
    beta  = float(params["beta"])
    gamma = float(params["gamma"])
    ethr  = int(params["energy_threshold"])

    # 1. Energy
    energy = elastic_deformation_diffusion(
        img_u8, base_gx, base_gy,
        iterations=params.get("iters", 16),
        diffusion_rate=dr, mu=mu, lambda_param=lam,
        edge_thresh=ethr
    )

    # 2. Segmentation
    seg_mask_bool = snake_seg(
        img_u8, energy, its=its_snake,
        alpha=alpha, beta=beta, gamma=gamma,
        l_size=l_size, u_size=u_size, energy_threshold=ethr
    )

    # 3. Basic Props
    area_px = int(np.sum(seg_mask_bool))
    area_frac = area_px / float(total_pixels + 1e-8)

    # --- Hard Rejection ---
    # Reject if empty or taking up almost entire image
    if area_px < 5 or area_frac > MAX_AREA_FRAC_HARD:
        return EvalResult(-1e9, 0, area_px, area_frac, 0, 0.0, 0.0)

    seg_u8 = (seg_mask_bool.astype(np.uint8)) * 255

    # 4. --- SCORING METRICS ---

    # A. Edge Alignment (The "Truth" metric)
    # Does the mask boundary sit on high-gradient pixels?
    base_mag = np.sqrt(base_gx**2 + base_gy**2)
    
    # Get mask perimeter (1px thick)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    contour_edge = cv2.morphologyEx(seg_u8, cv2.MORPH_GRADIENT, kernel)
    
    edge_pixels = np.sum(contour_edge > 0)
    avg_edge_strength = 0.0
    if edge_pixels > 0:
        # Normalized 0.0 to 1.0
        avg_edge_strength = cv2.mean(base_mag, mask=contour_edge)[0] / 255.0

    # B. Local Contrast (The "Visibility" metric)
    # Compare inside of lesion to a thin ring immediately outside.
    # Better for irregular shapes than global contrast.
    dilated = cv2.dilate(seg_u8, kernel, iterations=3)
    ring_mask = cv2.subtract(dilated, seg_u8) # The ring outside
    
    mean_in = cv2.mean(img_u8, mask=seg_u8)[0]
    
    # Handle edge case where mask touches image border (ring might be empty)
    if cv2.countNonZero(ring_mask) > 0:
        mean_ring = cv2.mean(img_u8, mask=ring_mask)[0]
    else:
        mean_ring = 0 # Fallback
    
    contrast_score = abs(mean_in - mean_ring) / 255.0

    # C. Noise / Components
    label_img = label(seg_mask_bool)
    props = regionprops(label_img)
    n_contours = len(props)
    small_count = sum(1 for p in props if p.area < SMALL_MIN_AREA)

    # --- COMPOSITE SCORE ---
    score = 0.0
    
    # Reward Edge Alignment (High Importance)
    score += W_EDGE_ALIGN * avg_edge_strength
    
    # Reward Contrast (Medium-High Importance)
    score += W_CONTRAST * contrast_score
    
    # Penalize Noise (Low Importance, but keeps things clean)
    score -= W_SMALL * small_count

    # Soft Area Penalty
    # Gently discourage extreme sizes without hard caps
    if area_frac < MIN_AREA_FRAC_SOFT:
        score -= 5.0
    elif area_frac > MAX_AREA_FRAC_SOFT:
        score -= 5.0

    return EvalResult(
        score=score, 
        n_contours=n_contours, 
        area_px=area_px,
        area_frac=area_frac, 
        small_count=small_count,
        edge_score=avg_edge_strength,
        contrast_score=contrast_score
    )

# ----------------------------- Sampling / Refinement --------------------------
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
                      seed: int) -> Dict:
    
    rng = random.Random((hash(filename) ^ seed) & 0xFFFFFFFF)

    in_path = os.path.join(input_dir, filename)
    raw, bgr8_full, gray8_full = read_color_anydepth(in_path)
    if raw is None:
        return {"filename": filename, "status": "load_error"}

    # Choose working channel by mode
    if mode == "bestch":
        # use G channel with binary leaf mask
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

    # Gradients once
    _, gx, gy = gradient_magnitude(work)

    h, w = work.shape
    u_size = int(0.40 * h * w)

    t0 = time.time()
    best: Optional[Tuple[EvalResult, Dict]] = None
    tried = 0

    # -------- Random exploration --------
    while tried < budget_random:
        cand = sample_candidate(rng)
        cand["iters"] = iters_coarse
        res = evaluate_params(work, gx, gy, snake_iters_coarse, cand, l_size=5, u_size=u_size)
        tried += 1
        if (best is None) or (res.score > best[0].score):
            best = (res, cand)
        # watchdog
        if per_image_seconds and (time.time() - t0 > per_image_seconds):
            return _finalize(filename, raw, work_full, best, mode, output_dir, iters_final, snake_iters_final, timeout=True)

    # gather top seeds
    seeds = [best[1]]
    for _ in range(max(0, topk_refine - 1)):
        seeds.append(perturb(rng, best[1], scale=0.25))

    # -------- Local refinement --------
    for s_idx, seed_p in enumerate(seeds):
        current = (best[0], dict(seed_p))
        patience = 0
        for step in range(refine_steps):
            scale = 0.25 * (0.85 ** step)  # cooling
            cand = perturb(rng, current[1], scale=scale)
            cand["iters"] = iters_coarse
            res = evaluate_params(work, gx, gy, snake_iters_coarse, cand, l_size=5, u_size=u_size)
            tried += 1
            if res.score > current[0].score + 1e-6:
                current = (res, cand)
                patience = 0
                if res.score > best[0].score + 1e-6:
                    best = (res, cand)
            else:
                patience += 1
            if per_image_seconds and (time.time() - t0 > per_image_seconds):
                return _finalize(filename, raw, work_full, best, mode, output_dir, iters_final, snake_iters_final, timeout=True)
            if patience >= 5:
                break

    # -------- Finalize at full resolution --------
    return _finalize(filename, raw, work_full, best, mode, output_dir, iters_final, snake_iters_final, timeout=False)

def _finalize(filename, raw, full_img_u8, best, mode, output_dir, iters_final, snake_iters_final, timeout: bool):
    if best is None:
        return {"filename": filename, "status": "no_candidates"}

    # full-res gradients
    _, gx_f, gy_f = gradient_magnitude(full_img_u8)

    # run at full res
    params = dict(best[1])
    params["iters"] = iters_final
    res_final = evaluate_params(full_img_u8, gx_f, gy_f, snake_iters_final, params)

    # write outputs
    base = os.path.splitext(filename)[0]
    ensure_dir(output_dir)
    overlay16 = _prepare_overlay16(raw)

    seg_u8 = np.zeros_like(full_img_u8, dtype=np.uint8)
    energy = elastic_deformation_diffusion(
        full_img_u8, gx_f, gy_f,
        iterations=params["iters"],
        diffusion_rate=params["diffusion_rate"],
        mu=params["mu"],
        lambda_param=params["lambda"],
        edge_thresh=int(params["energy_threshold"])
    )
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

    tag = f"{mode}_score{int(res_final.score)}_irr"

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
        "area_frac": res_final.area_frac,
        "overlay16_path": overlay_path,
    }

# ----------------------------- CLI / Driver ----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Irregular Lesion Optimizer (Edge+Contrast)")
    ap.add_argument("--input",  required=True, help="Input directory of images")
    ap.add_argument("--output", required=True, help="Output directory for masks/overlays")
    ap.add_argument("--mode", choices=["bestch","fused"], default="bestch")

    ap.add_argument("--budget-random", type=int, default=50)
    ap.add_argument("--topk-refine", type=int, default=3)
    ap.add_argument("--refine-steps", type=int, default=10)

    ap.add_argument("--iters-coarse", type=int, default=16)
    ap.add_argument("--snake-iters-coarse", type=int, default=50)
    ap.add_argument("--iters-final", type=int, default=30)
    ap.add_argument("--snake-iters-final", type=int, default=100)

    ap.add_argument("--downscale", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--per-image-seconds", type=int, default=None)
    ap.add_argument("--log", default="opt_summary.csv")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    try: cv2.setNumThreads(1)
    except: pass

    input_dir = args.input
    output_dir = args.output
    ensure_dir(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(ALLOWED_EXT)]
    files.sort()
    if args.limit is not None:
        files = files[:args.limit]

    max_workers = args.workers if args.workers else max(1, (os.cpu_count() or 2) - 1)

    print(f"Processing {len(files)} images. Scoring: Edge Alignment + Local Contrast.")

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for fn in files:
            fut = ex.submit(
                process_one_image, fn, input_dir, output_dir, args.mode,
                args.budget_random, args.topk_refine, args.refine_steps,
                args.iters_coarse, args.snake_iters_coarse,
                args.iters_final, args.snake_iters_final,
                args.downscale, args.per_image_seconds, args.seed
            )
            futs.append((fn, fut))

        for i, (fn, fut) in enumerate(futs, 1):
            try:
                out = fut.result()
            except Exception as e:
                out = {"filename": fn, "status": f"worker_error:{e}"}
            rows.append(out)
            print(f"[{i}/{len(futs)}] {fn}: {out.get('status','?')} score={fmt(out.get('score',0))} edge={fmt(out.get('edge_score',0))} contrast={fmt(out.get('contrast_score',0))}", flush=True)

    # Write CSV
    log_path = os.path.join(output_dir, args.log)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "filename","status","score","edge_score","contrast_score","n_contours","area_frac",
            "mu","lambda","diffusion_rate","alpha","beta","gamma","energy_threshold"
        ])
        for r in rows:
            p = r.get("params", {})
            w.writerow([
                r.get("filename",""),
                r.get("status",""),
                fmt(r.get("score","")),
                fmt(r.get("edge_score","")),
                fmt(r.get("contrast_score","")),
                r.get("n_contours",""),
                fmt(r.get("area_frac","")),
                fmt(p.get("mu","")),
                fmt(p.get("lambda","")),
                fmt(p.get("diffusion_rate","")),
                fmt(p.get("alpha","")),
                fmt(p.get("beta","")),
                fmt(p.get("gamma","")),
                p.get("energy_threshold",""),
            ])

    print(f"[DONE] Summary -> {log_path}")

if __name__ == "__main__":
    main()