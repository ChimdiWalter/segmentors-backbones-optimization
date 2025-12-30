#!/usr/bin/env python3
"""
navier_bestch_opt.py
- Best-channel = GREEN only (like your original navier_bestch)
- Per-image parameter optimization (unsupervised by default, Dice if GT provided)
- Saves 16-bit overlays + mask16 and logs best params & objective parts

Usage (defaults match your originals):
  python3 navier_bestch_opt.py \
    --input  /deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves \
    --output /deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/navier_output \
    --log    results_log.csv

Optional:
  --gt-dir /path/to/gt_masks          # if you have binary GT masks (same filename stem)
  --weights 0.45,0.25,0.15,0.15       # edge, compactness, area_prior, stability
  --random 24 --refine 12 --seed 0    # optimizer settings
"""

import os
import sys
import csv
import math
import argparse
import random
import itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Tuple

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# ----------------------------- Defaults -------------------------------------
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# 16-bit pink (B, G, R)
PINK_16 = (65535, 5140, 37779)

# Energy thresholds to try during optimization (discrete)
ENERGY_THRESHOLDS = (30, 50, 70)

# Curated TEN combos (for warmstart) (mu, lam, d, alpha, beta, gamma, eth)
WARM_COMBOS = [
    (0.05, 0.05, 0.1, 0.01, 0.1, 0.01, 30),
    (0.05, 0.05, 0.4, 0.01, 0.1, 0.10, 30),
    (0.05, 0.50, 0.1, 0.01, 0.5, 0.01, 30),
    (0.05, 0.50, 0.4, 0.01, 0.5, 0.10, 30),

    (0.50, 0.05, 0.1, 0.10, 0.1, 0.01, 30),
    (0.50, 0.05, 0.4, 0.10, 0.1, 0.10, 30),
    (0.50, 0.50, 0.1, 0.10, 0.5, 0.01, 30),
    (0.50, 0.50, 0.4, 0.10, 0.5, 0.10, 30),

    (0.05, 0.05, 0.1, 0.50, 0.1, 0.01, 70),
    (0.50, 0.50, 0.4, 0.50, 0.5, 0.10, 70),
]

CONTOUR_MODE = cv2.RETR_EXTERNAL
APPROX_MODE  = cv2.CHAIN_APPROX_SIMPLE

# ----------------------------- Utils ----------------------------------------
def ensure_dir(path): os.makedirs(path, exist_ok=True)
def fmt(v): return str(v).replace('.', 'p')

def read_color_anydepth(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None: return None, None, None

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

def _to_uint8(img):
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

# ----------------------------- Core ops -------------------------------------
def binary_threshold_mask(gray8, threshold=127):
    _, mask = cv2.threshold(gray8, threshold, 255, cv2.THRESH_BINARY)
    return mask

def apply_mask(image_u8, mask_u8):
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
    diffused = np.sqrt(grad_x**2 + grad_y**2).astype(np.float64)

    im_gx = cv2.Sobel(image_u8, cv2.CV_64F, 1, 0, ksize=3)
    im_gy = cv2.Sobel(image_u8, cv2.CV_64F, 0, 1, ksize=3)
    im_mag = np.sqrt(im_gx**2 + im_gy**2)

    q = (im_mag > edge_thresh).astype(np.float64)

    for _ in range(iterations):
        lap = laplace(diffused)
        div_v = np.gradient(diffused)[0]
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
                max_num_iter=100
            )
        except TypeError:
            snake = active_contour(
                image=crop_f, snake=init,
                alpha=alpha, beta=beta, gamma=gamma,
                max_iterations=100
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

# ---------------------- Objective & Optimizer -------------------------------
def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return (2.0*inter / s) if s > 0 else 0.0

def boundary_hit_fraction(mask: np.ndarray, grad_mag_u8: np.ndarray, thr: int=40) -> float:
    m = mask.astype(np.uint8)
    edges = cv2.Canny(m*255, 50, 150)
    strong = (grad_mag_u8 >= thr).astype(np.uint8)
    hit = np.logical_and(edges>0, strong>0).sum()
    tot = (edges>0).sum()
    return (hit / tot) if tot > 0 else 0.0

def mean_compactness(mask: np.ndarray) -> float:
    m = mask.astype(np.uint8)
    nlab, lab = cv2.connectedComponents(m)
    if nlab <= 1:
        return 0.0
    vals = []
    for lbl in range(1, nlab):
        comp = (lab==lbl).astype(np.uint8)
        area = int(comp.sum())
        if area < 5:
            continue
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        per = float(cv2.arcLength(contours[0], True))
        if per <= 1e-6:
            continue
        vals.append((4.0*math.pi*area)/(per*per))
    return float(np.mean(vals)) if vals else 0.0

def gaussian_area_prior(area: int, target: float, sigma: float) -> float:
    z = (area - target) / (sigma + 1e-8)
    return math.exp(-float(z*z))

def segment_with_params(image_u8, gx, gy, params):
    mu, lam, d, alpha, beta, gamma, eth = params
    energy = elastic_deformation_diffusion(
        image_u8, gx, gy,
        iterations=30, diffusion_rate=d, mu=mu, lambda_param=lam, edge_thresh=eth
    )
    mask = snake_seg(
        image_u8, energy, its=100,
        alpha=alpha, beta=beta, gamma=gamma,
        l_size=5, u_size=50000, energy_threshold=eth
    )
    return mask

def stability_score(mask_base: np.ndarray, make_mask_fn, jitters: int=2) -> float:
    base = mask_base.astype(bool)
    ious = []
    for _ in range(jitters):
        m = make_mask_fn(jitter=True)
        if m is None:
            continue
        other = m.astype(bool)
        inter = np.logical_and(base, other).sum()
        uni = np.logical_or(base, other).sum()
        ious.append((inter/uni) if uni>0 else 1.0)
    return float(np.mean(ious)) if ious else 0.0

def optimize_params_for_image(
    image_u8: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    grad_mag_u8: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    w_edge: float = 0.45,
    w_compact: float = 0.25,
    w_area: float = 0.15,
    w_stab: float = 0.15,
    area_target: Optional[float] = None,
    area_sigma: float = 3e5,
    energy_threshold_choices=ENERGY_THRESHOLDS,
    warmstart_params=WARM_COMBOS,
    n_random=24,
    n_refine=12,
    seed: int = 0
) -> Dict:
    """Returns dict with best_params, best_score, best_mask, parts."""
    rng = random.Random(seed)
    H, W = image_u8.shape
    total_pixels = H*W
    if area_target is None:
        area_target = 0.02 * total_pixels  # default 2% of image

    # bounds
    MU = (0.01, 1.0)
    LA = (0.01, 1.0)
    D  = (0.05, 0.6)
    A  = (0.005, 0.6)
    B  = (0.05,  1.0)
    G  = (0.005, 0.5)

    def clip(v, lo, hi): return max(lo, min(hi, v))

    def rand_params():
        mu = 10**rng.uniform(np.log10(MU[0]), np.log10(MU[1]))
        la = 10**rng.uniform(np.log10(LA[0]), np.log10(LA[1]))
        d  = rng.uniform(*D)
        a  = 10**rng.uniform(np.log10(A[0]), np.log10(A[1]))
        b  = rng.uniform(*B)
        g  = 10**rng.uniform(np.log10(G[0]), np.log10(G[1]))
        eth = rng.choice(list(energy_threshold_choices))
        return (mu, la, d, a, b, g, eth)

    def objective(params):
        mask = segment_with_params(image_u8, grad_x, grad_y, params)
        if mask is None:
            return -1e9, None, {}
        area = int(mask.sum())

        parts = {}
        if gt_mask is not None:
            parts["dice"] = dice_coefficient(mask, gt_mask.astype(bool))
            score = parts["dice"]
        else:
            parts["edge"] = boundary_hit_fraction(mask, grad_mag_u8, thr=40)
            parts["compact"] = mean_compactness(mask)
            parts["area_prior"] = gaussian_area_prior(area, area_target, area_sigma)
            def mk(jitter=False):
                if not jitter:
                    return mask
                mu, la, d, a, b, g, eth = params
                j = lambda x, lo, hi: clip(x*(1.0 + rng.uniform(-0.08, 0.08)), lo, hi)
                pj = (j(mu,*MU), j(la,*LA), j(d,*D), j(a,*A), j(b,*B), j(g,*G), eth)
                return segment_with_params(image_u8, grad_x, grad_y, pj)
            parts["stability"] = stability_score(mask, mk, jitters=2)
            score = (w_edge*parts["edge"] + w_compact*parts["compact"] +
                     w_area*parts["area_prior"] + w_stab*parts["stability"])
        parts["area_px"] = area
        return float(score), mask, parts

    best = {"score": -1e9, "params": None, "mask": None, "parts": {}}

    def try_params(p):
        s, m, pr = objective(p)
        nonlocal best
        if s > best["score"]:
            best = {"score": s, "params": p, "mask": m, "parts": pr}

    for p in warmstart_params:
        try_params(p)
    for _ in range(n_random):
        try_params(rand_params())

    # coordinate-like refine around best
    for _ in range(n_refine):
        mu, la, d, a, b, g, eth = best["params"]
        def step_pair(x, lo, hi, frac=0.15):
            step = (hi-lo)*frac
            return [clip(x-step, lo, hi), clip(x+step, lo, hi)]
        for mu2 in step_pair(mu, *MU):
            try_params((mu2, la, d, a, b, g, eth))
        for la2 in step_pair(la, *LA):
            try_params((mu, la2, d, a, b, g, eth))
        for d2 in step_pair(d, *D):
            try_params((mu, la, d2, a, b, g, eth))
        for a2 in step_pair(a, *A):
            try_params((mu, la, d, a2, b, g, eth))
        for b2 in step_pair(b, *B):
            try_params((mu, la, d, a, b2, g, eth))
        for g2 in step_pair(g, *G):
            try_params((mu, la, d, a, b, g2, eth))

    return {
        "best_params": best["params"],
        "best_score":  best["score"],
        "best_mask":   best["mask"],
        "parts":       best["parts"]
    }

# ----------------------------- Worker ---------------------------------------
def load_gt_mask(gt_dir: Optional[str], filename: str, shape_hw: Tuple[int,int]):
    if not gt_dir:
        return None
    stem = os.path.splitext(filename)[0]
    # try common endings
    cands = [
        os.path.join(gt_dir, stem + ".tif"),
        os.path.join(gt_dir, stem + ".tiff"),
        os.path.join(gt_dir, stem + ".png"),
        os.path.join(gt_dir, stem + "_mask.png"),
        os.path.join(gt_dir, stem + "_mask.tif"),
    ]
    for p in cands:
        if os.path.exists(p):
            m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim == 3:
                m = cv2.cvtColor(_to_uint8(m), cv2.COLOR_BGR2GRAY)
            m = (m > 0).astype(np.uint8)
            if m.shape != shape_hw:
                m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
            return m.astype(bool)
    return None

def process_single_image(filename, args):
    rows = []
    try:
        in_path = os.path.join(args.input, filename)
        raw, bgr8, gray8 = read_color_anydepth(in_path)
        if raw is None:
            print(f"[skip] Could not load {filename}", flush=True)
            return rows

        # GREEN channel (bestch)
        b8, g8, r8 = cv2.split(bgr8)
        binary_mask = binary_threshold_mask(gray8, threshold=127)
        chan = apply_mask(g8, binary_mask)

        # Precompute image-driven terms
        grad_mag_u8, gx, gy = gradient_magnitude(chan)

        # Optional GT
        gt_mask = load_gt_mask(args.gt_dir, filename, chan.shape)

        # Optimize
        opt = optimize_params_for_image(
            image_u8=chan,
            grad_x=gx, grad_y=gy,
            grad_mag_u8=grad_mag_u8,
            gt_mask=gt_mask,
            w_edge=args.weights[0],
            w_compact=args.weights[1],
            w_area=args.weights[2],
            w_stab=args.weights[3],
            area_target=args.area_target if args.area_target > 0 else None,
            area_sigma=args.area_sigma,
            energy_threshold_choices=ENERGY_THRESHOLDS,
            warmstart_params=WARM_COMBOS,
            n_random=args.random,
            n_refine=args.refine,
            seed=args.seed + (abs(hash(filename)) % 1000003)
        )

        (mu, lam, d, alpha, beta, gamma, eth) = opt["best_params"]
        best_mask = opt["best_mask"].astype(bool)
        parts = opt["parts"]
        total_area = int(best_mask.sum())

        # Post-process & contours for overlay
        seg_u8 = (best_mask.astype(np.uint8)) * 255
        seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(seg_u8, CONTOUR_MODE, APPROX_MODE)
        n_contours = len(contours)

        # Save overlay16 and mask16
        overlay16 = _prepare_overlay16(raw)
        out_overlay16 = overlay16.copy()
        cv2.drawContours(out_overlay16, contours, -1, PINK_16, thickness=2)

        mask16 = (best_mask.astype(np.uint16)) * 65535

        base = os.path.splitext(filename)[0]
        param_str = f"mu{fmt(mu)}_lam{fmt(lam)}_d{fmt(d)}_a{fmt(alpha)}_b{fmt(beta)}_g{fmt(gamma)}_e{eth}"
        overlay_path = os.path.join(args.output, f"{base}__{param_str}__overlay16.tif")
        mask_path    = os.path.join(args.output, f"{base}__{param_str}__mask16.tif")
        ensure_dir(os.path.dirname(overlay_path))
        cv2.imwrite(overlay_path, out_overlay16)
        cv2.imwrite(mask_path,    mask16)

        row = [
            "bestch", filename,
            mu, lam, d, alpha, beta, gamma, eth,
            n_contours, total_area,
            parts.get("edge", np.nan),
            parts.get("compact", np.nan),
            parts.get("area_prior", np.nan),
            parts.get("stability", np.nan),
            parts.get("dice", np.nan),
            float(opt["best_score"]),
            overlay_path, mask_path
        ]
        rows.append(row)
    except Exception as ex:
        print(f"[worker error] {filename}: {ex}", flush=True)
    return rows

# ----------------------------- Driver ---------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Navier+Snake (GREEN channel) with per-image parameter optimization.")
    ap.add_argument("--input",  default="/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves")
    ap.add_argument("--output", default="/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/navier_output")
    ap.add_argument("--log",    default="results_log.csv")
    ap.add_argument("--gt-dir", default=None, help="Optional directory containing GT masks (binary).")
    ap.add_argument("--weights", default="0.45,0.25,0.15,0.15",
                    help="edge,compactness,area_prior,stability weights for unsupervised objective.")
    ap.add_argument("--area-target", type=float, default=-1.0,
                    help="Target lesion area in pixels (<=0 uses 2% of image).")
    ap.add_argument("--area-sigma", type=float, default=3e5)
    ap.add_argument("--random", type=int, default=24, help="# random samples in optimizer.")
    ap.add_argument("--refine", type=int, default=12, help="# local refinements in optimizer.")
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    # parse weights
    w = [float(x) for x in args.weights.split(",")]
    if len(w) != 4:
        raise ValueError("--weights must have 4 comma-separated floats")
    args.weights = w

    # Avoid thread oversubscription per worker
    for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input dir not found: {args.input}")
    ensure_dir(args.output)

    files = [f for f in os.listdir(args.input) if f.lower().endswith(ALLOWED_EXT)]
    if len(files) == 0:
        raise RuntimeError("Found 0 candidate images.")
    max_workers = max(1, (os.cpu_count() or 2) - 1)

    print("========== navier BESTCH (GREEN) — OPT per image ==========")
    print(f"Start:       {datetime.now().isoformat(timespec='seconds')}")
    print(f"Python:      {sys.executable}")
    print(f"OpenCV:      {cv2.__version__}")
    print(f"Input dir:   {args.input}")
    print(f"Output dir:  {args.output}")
    print(f"Found files: {len(files)}")
    print(f"Workers:     {max_workers}")
    print(f"Objective w: edge={args.weights[0]}, compact={args.weights[1]}, area={args.weights[2]}, stab={args.weights[3]}")
    print("===========================================================", flush=True)

    all_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_single_image, fn, args): fn for fn in files}
        for i, fut in enumerate(as_completed(futs), 1):
            fn = futs[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                print(f"[{i}/{len(files)}] done: {fn} -> {len(rows)} row(s)", flush=True)
            except Exception as e:
                print(f"[error] {fn}: {e}", flush=True)

    log_path = os.path.join(args.output, args.log)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "mode","filename","mu","lambda","diffusion_rate","alpha","beta","gamma","energy_threshold",
                "n_contours","total_area_px",
                "edge_hit","compactness","area_prior","stability","dice","objective_score",
                "overlay16_path","mask16_path"
            ])
        writer.writerows(all_rows)

    print(f"[DONE] wrote {len(all_rows)} rows -> {log_path}", flush=True)

if __name__ == "__main__":
    main()
