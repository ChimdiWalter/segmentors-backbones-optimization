#!/usr/bin/env python3
# nv_optimize_robust_v2.py (refactored)
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

# ----------------------------- Constants / Config -----------------------------
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# 16-bit pink (B, G, R) for overlays
PINK_16 = (65535, 5140, 37779)

# Search bounds
BOUNDS = {
    "mu":              (0.03, 0.80),
    "lambda":          (0.03, 0.80),
    "diffusion_rate":  (0.03, 0.60),
    "alpha":           (0.05, 0.70),
    "beta":            (0.05, 1.20),
    "gamma":           (0.01, 0.60),
    "energy_threshold": (15, 140),  # integer
}

# Objective weights
TARGET_AREA_FRAC    = 0.05     # prefer ~5% lesion pixels
W_AREA              = 2.0      # penalty weight for area deviation
W_SMALL             = 0.5      # penalty per small component
SMALL_MIN_AREA      = 25       # px
MAX_AREA_FRAC_HARD  = 0.60     # hard cap (reject if above)
MIN_AREA_FRAC_HARD  = 0.0005   # hard cap (reject if below)

# ----------------------------- Utilities -------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def fmt(v):
    if isinstance(v, float):
        return str(v).replace('.', 'p')
    return str(v)

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return cv2.convertScaleAbs(img, alpha=255.0 / 65535.0)
    if img.dtype in (np.float32, np.float64):
        im = np.clip(img, 0.0, 1.0)
        return (im * 255.0).astype(np.uint8)
    imin, imax = float(np.min(img)), float(np.max(img))
    if imax <= imin + 1e-12:
        return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * (img - imin) / (imax - imin)).astype(np.uint8)

def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return (img.astype(np.float32)) / 255.0
    if img.dtype == np.uint16:
        return (img.astype(np.float32)) / 65535.0
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx <= mn + 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)

def read_color_anydepth(path: str):
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

def binary_threshold_mask(gray8: np.ndarray, threshold: int = 127) -> np.ndarray:
    _, mask = cv2.threshold(gray8, threshold, 255, cv2.THRESH_BINARY)
    return mask

def apply_mask(image_u8: np.ndarray, mask_u8: Optional[np.ndarray]) -> np.ndarray:
    if mask_u8 is None:
        return image_u8
    return cv2.bitwise_and(image_u8, image_u8, mask=mask_u8)

# Old helper retained (not core to PDE now, but harmless to keep)
def gradient_magnitude(u8: np.ndarray):
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    eps = 1e-8
    mag_u8 = np.uint8(255 * mag / max(np.max(mag), eps))
    return mag_u8, gx, gy

# ----------------------------- Gradient + Seed Helpers ------------------------
def compute_image_gradients(u8: np.ndarray):
    """
    Compute image gradients and magnitude (Sobel).
    Returns (gx, gy, mag) as float64 arrays.
    """
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, mag

def build_seed_from_image_gradients(
    img_grad_x: np.ndarray,
    img_grad_y: np.ndarray,
    sigma: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a smoothed seed gradient field and its magnitude from the image gradients.
    Seed = low-pass / coarse gradient; image = fine gradient.
    """
    seed_gx = cv2.GaussianBlur(img_grad_x, (0, 0), sigmaX=sigma, sigmaY=sigma)
    seed_gy = cv2.GaussianBlur(img_grad_y, (0, 0), sigmaX=sigma, sigmaY=sigma)
    seed_mag = np.sqrt(seed_gx ** 2 + seed_gy ** 2)
    return seed_gx, seed_gy, seed_mag

# ----------------------------- Navier/Elastic PDE ----------------------------
def elastic_deformation_diffusion_refined(
    seed_mag: np.ndarray,
    img_mag: np.ndarray,
    img_grad_x: np.ndarray,
    img_grad_y: np.ndarray,
    seed_grad_x: np.ndarray,
    seed_grad_y: np.ndarray,
    iterations: int = 30,
    diffusion_rate: float = 0.2,
    mu: float = 0.5,
    lambda_param: float = 0.5,
    edge_thresh: float = 50.0,
) -> np.ndarray:
    """
    Navier-Stokes/elastic-inspired scalar diffusion field.

    Parameters
    ----------
    seed_mag      : initial scalar field, typically |seed_grad|.
    img_mag       : magnitude of image gradient |∇I|.
    img_grad_x/y  : image gradient components.
    seed_grad_x/y : seed/prior gradient components.
    iterations    : PDE steps.
    diffusion_rate, mu, lambda_param : PDE hyperparams.
    edge_thresh   : threshold on |∇I| to define an edge mask q.

    Returns
    -------
    energy_u8 : uint8 energy image (0-255).
    """
    diffused = seed_mag.astype(np.float64).copy()

    img_grad_x = img_grad_x.astype(np.float64)
    img_grad_y = img_grad_y.astype(np.float64)
    seed_grad_x = seed_grad_x.astype(np.float64)
    seed_grad_y = seed_grad_y.astype(np.float64)

    # Edge mask
    edge_mask = (img_mag > edge_thresh).astype(np.float64)

    for _ in range(iterations):
        # isotropic diffusion
        lap = laplace(diffused)

        # simple divergence proxy of diffused field
        grad_y, grad_x = np.gradient(diffused)
        div_v = grad_x + grad_y

        # difference between image and seed gradient fields
        ftx = img_grad_x - seed_grad_x
        fty = img_grad_y - seed_grad_y
        ftm = np.sqrt(ftx ** 2 + fty ** 2)

        diffused += diffusion_rate * (
            mu * lap +
            (lambda_param + mu) * div_v +
            edge_mask * (ftm - diffused)
        )

    diffused = np.maximum(diffused, 0.0)
    mx = float(diffused.max())
    if mx < 1e-8:
        return np.zeros_like(diffused, dtype=np.uint8)
    return np.uint8(255.0 * diffused / mx)

# ----------------------------- Snakes & Overlays -----------------------------
def snake_seg(
    image_u8: np.ndarray,
    energy_u8: np.ndarray,
    its: int,
    alpha: float,
    beta: float,
    gamma: float,
    l_size: int,
    u_size: Optional[int],
    energy_threshold: int
) -> np.ndarray:
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
        s = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        rr = (maxr - minr) / 2.0
        cc = (maxc - minc) / 2.0
        init = np.vstack([rr * np.sin(s) + rr, cc * np.cos(s) + cc]).T

        try:
            snake = active_contour(
                image=crop_f,
                snake=init,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                max_num_iter=its
            )
        except TypeError:
            snake = active_contour(
                image=crop_f,
                snake=init,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                max_iterations=its
            )

        snake_i = np.round(snake).astype(int)
        snake_i[:, 0] = np.clip(snake_i[:, 0], 0, maxr - minr - 1)
        snake_i[:, 1] = np.clip(snake_i[:, 1], 0, maxc - minc - 1)

        pr, pc = snake_i[:, 0], snake_i[:, 1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop_u8.shape)
        out_mask[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    return out_mask

def _prepare_overlay16(raw: np.ndarray) -> np.ndarray:
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

def fused_channel_u8(bgr8: np.ndarray, binary_mask: Optional[np.ndarray] = None) -> np.ndarray:
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

def evaluate_params_refined(
    img_u8: np.ndarray,
    img_mag: np.ndarray,
    img_grad_x: np.ndarray,
    img_grad_y: np.ndarray,
    seed_mag: np.ndarray,
    seed_grad_x: np.ndarray,
    seed_grad_y: np.ndarray,
    its_snake: int,
    params: Dict,
    l_size: int = 5,
    u_size: Optional[int] = None
) -> EvalResult:
    """
    Run refined energy + snake; return composite score.
    Uses precomputed image + seed gradients.
    """
    h, w = img_u8.shape
    if u_size is None:
        u_size = int(0.25 * h * w)

    mu    = float(params["mu"])
    lam   = float(params["lambda"])
    dr    = float(params["diffusion_rate"])
    alpha = float(params["alpha"])
    beta  = float(params["beta"])
    gamma = float(params["gamma"])
    ethr  = int(params["energy_threshold"])

    energy = elastic_deformation_diffusion_refined(
        seed_mag=seed_mag,
        img_mag=img_mag,
        img_grad_x=img_grad_x,
        img_grad_y=img_grad_y,
        seed_grad_x=seed_grad_x,
        seed_grad_y=seed_grad_y,
        iterations=params.get("iters", 16),
        diffusion_rate=dr,
        mu=mu,
        lambda_param=lam,
        edge_thresh=ethr,
    )

    seg_mask_bool = snake_seg(
        img_u8,
        energy,
        its=its_snake,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        l_size=l_size,
        u_size=u_size,
        energy_threshold=ethr,
    )

    area_px = int(np.sum(seg_mask_bool))
    area_frac = area_px / float(h * w + 1e-8)

    # reject absurd areas
    if area_frac > MAX_AREA_FRAC_HARD or area_frac < MIN_AREA_FRAC_HARD:
        return EvalResult(
            score=-1e9,
            n_contours=0,
            area_px=area_px,
            area_frac=area_frac,
            small_count=0,
        )

    seg_u8 = (seg_mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(
        seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    n_contours = len(contours)

    lab = label(seg_mask_bool)
    props = regionprops(lab)
    small_count = sum(1 for p in props if p.area < SMALL_MIN_AREA)

    score = (
        n_contours
        - W_SMALL * small_count
        - W_AREA * abs(area_frac - TARGET_AREA_FRAC)
    )

    return EvalResult(
        score=score,
        n_contours=n_contours,
        area_px=area_px,
        area_frac=area_frac,
        small_count=small_count,
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
    for k in ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma"]:
        span = BOUNDS[k][1] - BOUNDS[k][0]
        q[k] += rng.uniform(-1, 1) * scale * span
    q["energy_threshold"] += int(
        round(
            rng.uniform(-1, 1)
            * scale
            * (BOUNDS["energy_threshold"][1] - BOUNDS["energy_threshold"][0])
            * 0.2
        )
    )
    return clip_params(q)

# ----------------------------- Per-image Worker ------------------------------
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
) -> Dict:
    """
    Returns a dict with status + best params + metrics; also writes mask/overlay.
    """
    rng = random.Random((hash(filename) ^ seed) & 0xFFFFFFFF)

    in_path = os.path.join(input_dir, filename)
    raw, bgr8_full, gray8_full = read_color_anydepth(in_path)
    if raw is None:
        return {"filename": filename, "status": "load_error"}

    # Choose working channel by mode
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
        work = cv2.resize(
            work_full,
            None,
            fx=downscale,
            fy=downscale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = work_full

    # Precompute image + seed gradients once at search resolution
    img_gx, img_gy, img_mag = compute_image_gradients(work)
    seed_gx, seed_gy, seed_mag = build_seed_from_image_gradients(img_gx, img_gy, sigma=1.5)

    h, w = work.shape
    u_size = int(0.25 * h * w)

    t0 = time.time()
    best: Optional[Tuple[EvalResult, Dict]] = None
    tried = 0

    # -------- Random exploration --------
    while tried < budget_random:
        cand = sample_candidate(rng)
        cand["iters"] = iters_coarse
        res = evaluate_params_refined(
            work,
            img_mag,
            img_gx,
            img_gy,
            seed_mag,
            seed_gx,
            seed_gy,
            snake_iters_coarse,
            cand,
            l_size=5,
            u_size=u_size,
        )
        tried += 1
        if (best is None) or (res.score > best[0].score):
            best = (res, cand)
        if per_image_seconds and (time.time() - t0 > per_image_seconds):
            elapsed = time.time() - t0
            return _finalize(
                filename,
                raw,
                work_full,
                best,
                mode,
                output_dir,
                iters_final,
                snake_iters_final,
                timeout=True,
                elapsed_seconds=elapsed,
            )

    # -------- Local refinement around best --------
    seeds = [best[1]] if best is not None else []
    for _ in range(max(0, topk_refine - 1)):
        if best is not None:
            seeds.append(perturb(rng, best[1], scale=0.25))

    for s_idx, seed_p in enumerate(seeds):
        current = (best[0], dict(seed_p))
        patience = 0
        for step in range(refine_steps):
            scale = 0.25 * (0.85 ** step)
            cand = perturb(rng, current[1], scale=scale)
            cand["iters"] = iters_coarse
            res = evaluate_params_refined(
                work,
                img_mag,
                img_gx,
                img_gy,
                seed_mag,
                seed_gx,
                seed_gy,
                snake_iters_coarse,
                cand,
                l_size=5,
                u_size=u_size,
            )
            tried += 1
            if res.score > current[0].score + 1e-6:
                current = (res, cand)
                patience = 0
                if best is None or res.score > best[0].score + 1e-6:
                    best = (res, cand)
            else:
                patience += 1

            if per_image_seconds and (time.time() - t0 > per_image_seconds):
                elapsed = time.time() - t0
                return _finalize(
                    filename,
                    raw,
                    work_full,
                    best,
                    mode,
                    output_dir,
                    iters_final,
                    snake_iters_final,
                    timeout=True,
                    elapsed_seconds=elapsed,
                )
            if patience >= 5:
                break

    elapsed = time.time() - t0
    return _finalize(
        filename,
        raw,
        work_full,
        best,
        mode,
        output_dir,
        iters_final,
        snake_iters_final,
        timeout=False,
        elapsed_seconds=elapsed,
    )

# ----------------------------- Finalization ----------------------------------
def _finalize(
    filename: str,
    raw: np.ndarray,
    full_img_u8: np.ndarray,
    best: Optional[Tuple[EvalResult, Dict]],
    mode: str,
    output_dir: str,
    iters_final: int,
    snake_iters_final: int,
    timeout: bool,
    elapsed_seconds: Optional[float] = None,
) -> Dict:
    if best is None:
        return {
            "filename": filename,
            "status": "no_candidates",
            "mode": mode,
            "overlay16_path": "",
            "mask16_path": "",
            "elapsed_seconds": elapsed_seconds if elapsed_seconds is not None else "",
        }

    # full-res gradients + seeds
    img_gx_f, img_gy_f, img_mag_f = compute_image_gradients(full_img_u8)
    seed_gx_f, seed_gy_f, seed_mag_f = build_seed_from_image_gradients(img_gx_f, img_gy_f, sigma=1.5)

    params = dict(best[1])
    params["iters"] = iters_final

    res_final = evaluate_params_refined(
        full_img_u8,
        img_mag_f,
        img_gx_f,
        img_gy_f,
        seed_mag_f,
        seed_gx_f,
        seed_gy_f,
        snake_iters_final,
        params,
    )

    base = os.path.splitext(filename)[0]
    ensure_dir(output_dir)
    overlay16 = _prepare_overlay16(raw)

    # recompute final energy + mask at full res
    energy = elastic_deformation_diffusion_refined(
        seed_mag=seed_mag_f,
        img_mag=img_mag_f,
        img_grad_x=img_gx_f,
        img_grad_y=img_gy_f,
        seed_grad_x=seed_gx_f,
        seed_grad_y=seed_gy_f,
        iterations=params["iters"],
        diffusion_rate=params["diffusion_rate"],
        mu=params["mu"],
        lambda_param=params["lambda"],
        edge_thresh=int(params["energy_threshold"]),
    )
    seg_mask_bool = snake_seg(
        full_img_u8,
        energy,
        its=snake_iters_final,
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=params["gamma"],
        l_size=5,
        u_size=int(0.25 * full_img_u8.size),
        energy_threshold=int(params["energy_threshold"]),
    )
    seg_u8 = (seg_mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(
        seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    out_overlay16 = overlay16.copy()
    cv2.drawContours(out_overlay16, contours, -1, PINK_16, thickness=2)
    mask16 = (seg_mask_bool.astype(np.uint16)) * 65535

    tag = (
        f"{mode}_opt_mu{fmt(params['mu'])}_lam{fmt(params['lambda'])}_d{fmt(params['diffusion_rate'])}"
        f"_a{fmt(params['alpha'])}_b{fmt(params['beta'])}_g{fmt(params['gamma'])}_e{int(params['energy_threshold'])}"
    )

    overlay_path = os.path.join(output_dir, f"{base}__{tag}__overlay16.tif")
    mask_path    = os.path.join(output_dir, f"{base}__{tag}__mask16.tif")
    cv2.imwrite(overlay_path, out_overlay16)
    cv2.imwrite(mask_path,    mask16)

    result = {
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
        "elapsed_seconds": elapsed_seconds if elapsed_seconds is not None else "",
    }
    return result

# ----------------------------- CLI / Driver ----------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Robust per-image parameter optimization for Navier+Snake (bestch | fused)."
    )
    ap.add_argument("--input", required=True, help="Input directory of images")
    ap.add_argument("--output", required=True, help="Output directory for masks/overlays")
    ap.add_argument("--mode", choices=["bestch", "fused"], default="bestch")

    ap.add_argument("--budget-random", type=int, default=64, help="Random candidates")
    ap.add_argument("--topk-refine", type=int, default=4, help="Top seeds to locally refine")
    ap.add_argument("--refine-steps", type=int, default=12, help="Steps per seed")

    ap.add_argument("--iters-coarse", type=int, default=16, help="Energy iterations during search")
    ap.add_argument("--snake-iters-coarse", type=int, default=60, help="Snake iterations during search")
    ap.add_argument("--iters-final", type=int, default=30, help="Energy iterations for final full-res")
    ap.add_argument("--snake-iters-final", type=int, default=100, help="Snake iterations for final full-res")

    ap.add_argument(
        "--downscale",
        type=float,
        default=0.75,
        help="Search resolution scale (0,1]; final is full-res)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max number of images to process")
    ap.add_argument("--workers", type=int, default=None, help="ProcessPool size (default: CPU-1)")
    ap.add_argument("--per-image-seconds", type=int, default=None, help="Watchdog timeout per image")

    ap.add_argument("--log", default="opt_summary.csv", help="CSV summary filename")
    ap.add_argument("--seed", type=int, default=1337, help="Global RNG seed")

    args = ap.parse_args()

    # Avoid oversubscription
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    input_dir = args.input
    output_dir = args.output
    ensure_dir(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(ALLOWED_EXT)]
    files.sort()
    total = len(files)
    if args.limit is not None:
        files = files[: args.limit]

    max_workers = args.workers if args.workers else max(1, (os.cpu_count() or 2) - 1)

    print("========== nv_optimize_robust_v2 (refined) ==========")
    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Mode:     {args.mode}")
    print(f"Images:   {len(files)} / {total} total")
    print(f"Workers:  {max_workers}")
    print(
        f"Search:   rand={args.budget_random}, topk={args.topk_refine}, steps={args.refine_steps}, downscale={args.downscale}"
    )
    print(f"Coarse:   iters={args.iters_coarse}, snake={args.snake_iters_coarse}")
    print(f"Final:    iters={args.iters_final}, snake={args.snake_iters_final}")
    print(
        f"Timeout:  {args.per_image_seconds}s per image"
        if args.per_image_seconds
        else "Timeout:  none"
    )
    print("====================================================", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for fn in files:
            fut = ex.submit(
                process_one_image,
                fn,
                input_dir,
                output_dir,
                args.mode,
                args.budget_random,
                args.topk_refine,
                args.refine_steps,
                args.iters_coarse,
                args.snake_iters_coarse,
                args.iters_final,
                args.snake_iters_final,
                args.downscale,
                args.per_image_seconds,
                args.seed,
            )
            futs.append((fn, fut))

        for i, (fn, fut) in enumerate(futs, 1):
            try:
                out = fut.result()
            except Exception as e:
                out = {"filename": fn, "status": f"worker_error:{e}"}
            rows.append(out)
            print(
                f"[{i}/{len(futs)}] {fn}: {out.get('status','?')}  score={out.get('score','-')}",
                flush=True,
            )

    log_path = os.path.join(output_dir, args.log)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "filename",
                    "status",
                    "mode",
                    "score",
                    "n_contours",
                    "area_px",
                    "area_frac",
                    "small_count",
                    "mu",
                    "lambda",
                    "diffusion_rate",
                    "alpha",
                    "beta",
                    "gamma",
                    "energy_threshold",
                    "overlay16_path",
                    "mask16_path",
                    "elapsed_seconds",
                ]
            )
        for r in rows:
            p = r.get("params", {})
            w.writerow(
                [
                    r.get("filename", ""),
                    r.get("status", ""),
                    r.get("mode", ""),
                    r.get("score", ""),
                    r.get("n_contours", ""),
                    r.get("area_px", ""),
                    r.get("area_frac", ""),
                    r.get("small_count", ""),
                    p.get("mu", ""),
                    p.get("lambda", ""),
                    p.get("diffusion_rate", ""),
                    p.get("alpha", ""),
                    p.get("beta", ""),
                    p.get("gamma", ""),
                    p.get("energy_threshold", ""),
                    r.get("overlay16_path", ""),
                    r.get("mask16_path", ""),
                    r.get("elapsed_seconds", ""),
                ]
            )

    print(f"[DONE] wrote summary -> {log_path}")

if __name__ == "__main__":
    main()
