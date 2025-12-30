#!/usr/bin/env python3
"""
Robust per-image parameter optimization for Navier(+snake) segmentation.

Channel modes:
  - bestch : green channel (G)
  - fused  : max(G, GRAY, Lab a*) + CLAHE on masked leaf region

Objective (no GT):
  J = 0.7 * EdgeAlign + 0.2 * Stability - 0.1 * Complexity
where
  EdgeAlign   = mean gradient magnitude along mask boundary (0..1)
  Stability   = median IoU(M(theta), M(theta+tiny_jitter)) over 4 jitters
  Complexity  = (#components / area) + circularity = P^2/(4πA)

Search:
  Stage A: bounded random (Latin-ish) with curated warm starts
  Stage B: short local refine around the current best

Outputs (per image):
  - <base>__best__overlay16.tif
  - <base>__best__mask16.tif
  - <base>__best__params.json
  - Rows appended to results_robust_opt.csv
"""

import os, sys, json, time, math, random, argparse
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# ---------------------------- Defaults / config -----------------------------

ALLOWED_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

PINK_16 = (65535, 5140, 37779)   # (B,G,R) 16-bit pink for overlays

# Curated combos (mu, lam, dr, alpha, beta, gamma) — your proven basins
CURATED = [
    (0.05, 0.05, 0.1, 0.01, 0.1, 0.01),
    (0.05, 0.05, 0.4, 0.01, 0.1, 0.1),
    (0.05, 0.50, 0.1, 0.01, 0.5, 0.01),
    (0.05, 0.50, 0.4, 0.01, 0.5, 0.1),

    (0.50, 0.05, 0.1, 0.10, 0.1, 0.01),
    (0.50, 0.05, 0.4, 0.10, 0.1, 0.1),
    (0.50, 0.50, 0.1, 0.10, 0.5, 0.01),
    (0.50, 0.50, 0.4, 0.10, 0.5, 0.1),

    (0.05, 0.05, 0.1, 0.50, 0.1, 0.01),
    (0.50, 0.50, 0.4, 0.50, 0.5, 0.1),
]
CURATED_THRESH = [30, 70]

# Conservative bounds (cover curated but avoid dead zones)
BOUNDS = dict(
    mu=(0.02, 0.80),
    lam=(0.02, 0.80),
    diffusion_rate=(0.05, 0.60),
    alpha=(0.05, 0.60),
    beta=(0.05, 1.20),
    gamma=(0.02, 0.60),
    energy_threshold=(25, 95),
)

# Snake size gates as *fractions of leaf area*
LFRAC = 1e-4     # 0.01% of leaf area
UFRAC = 0.12     # 12% of leaf area

# Limits to keep things snappy & stable
MAX_PIXELS = 1_600_000   # e.g., 1600x1000; images above are downscaled
SNAKE_MAX_ITERS = 100
GAUSS_SIGMA = 1.25

# Objective weights
W_EDGE = 0.7
W_STAB = 0.2
W_COMP = 0.1

# ---------------------------- Utilities ------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_uint8(img):
    if img.dtype == np.uint8: return img
    if img.dtype == np.uint16: return cv2.convertScaleAbs(img, alpha=255.0/65535.0)
    if img.dtype in (np.float32, np.float64):
        im = np.clip(img, 0.0, 1.0); return (im * 255.0).astype(np.uint8)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn + 1e-12: return np.zeros_like(img, dtype=np.uint8)
    return (255.0 * (img - mn) / (mx - mn)).astype(np.uint8)

def to_float01(img):
    if img.dtype == np.uint8:  return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16: return img.astype(np.float32) / 65535.0
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn + 1e-12: return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)

def read_color_anydepth(path: Path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None: return None, None, None
    if raw.ndim == 2:
        raw_gray = raw
        bgr8 = to_uint8(raw_gray)
        if bgr8.ndim == 2:
            bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_GRAY2BGR)
        gray8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        return raw_gray, bgr8, gray8
    else:
        bgr8 = to_uint8(raw)
        if bgr8.ndim == 2:
            bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_GRAY2BGR)
        gray8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        return raw, bgr8, gray8

def resize_if_needed(bgr8):
    h, w = bgr8.shape[:2]
    if h * w <= MAX_PIXELS:
        return bgr8, 1.0
    scale = math.sqrt(MAX_PIXELS / float(h*w))
    bgr_small = cv2.resize(bgr8, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return bgr_small, scale

def binary_leaf_mask(gray8, thr=127):
    _, m = cv2.threshold(gray8, thr, 255, cv2.THRESH_BINARY)
    return m

def apply_mask(u8, mask_u8):
    return cv2.bitwise_and(u8, u8, mask=mask_u8)

def gradient_mag(u8):
    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = to_uint8(mag / (mag.max() + 1e-8))
    return mag_u8, gx, gy

def fused_channel_u8(bgr8, leaf_mask=None):
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)
    g = bgr8[...,1]
    candidates = [g, gray, lab[...,1]]
    fused = None
    for ch in candidates:
        chm = apply_mask(ch, leaf_mask) if leaf_mask is not None else ch
        chf = chm.astype(np.float32)
        mx  = float(chf.max())
        chn = (255.0 * chf / mx).astype(np.uint8) if mx > 1e-8 else np.zeros_like(chm, np.uint8)
        fused = chn if fused is None else np.maximum(fused, chn)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(fused)
    return apply_mask(enh, leaf_mask) if leaf_mask is not None else enh

def elastic_diffusion(image_u8, gx, gy, iterations, diffusion_rate, mu, lam, edge_thresh):
    diffused = np.sqrt(gx**2 + gy**2).astype(np.float64)
    im_gx = cv2.Sobel(image_u8, cv2.CV_64F, 1,0, ksize=3)
    im_gy = cv2.Sobel(image_u8, cv2.CV_64F, 0,1, ksize=3)
    im_mag = np.sqrt(im_gx**2 + im_gy**2)
    q = (im_mag > edge_thresh).astype(np.float64)
    for _ in range(iterations):
        lap = laplace(diffused)
        div_v = np.gradient(diffused)[0]  # simple proxy
        ftx = im_gx - gx
        fty = im_gy - gy
        ftm = np.sqrt(ftx**2 + fty**2)
        diffused += diffusion_rate * (mu * lap + (lam + mu) * div_v + q * (ftm - diffused))
    eps = 1e-8
    out = np.uint8(255 * diffused / max(np.max(diffused), eps))
    return out

def snake_seg(image_u8, energy_u8, alpha, beta, gamma, l_size, u_size, e_thr):
    h, w = image_u8.shape
    lbl = label(energy_u8 > e_thr)
    props = regionprops(lbl)
    out = np.zeros((h,w), dtype=bool)
    imgf = gaussian(to_float01(image_u8), GAUSS_SIGMA)

    for r in props:
        if not (l_size < r.area < u_size):
            continue
        minr, minc, maxr, maxc = r.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5:
            continue

        crop = image_u8[minr:maxr, minc:maxc]
        cropf = imgf[minr:maxr, minc:maxc]
        s = np.linspace(0, 2*np.pi, 200, endpoint=False)
        rr = (maxr - minr)/2.0
        cc = (maxc - minc)/2.0
        init = np.vstack([rr*np.sin(s)+rr, cc*np.cos(s)+cc]).T
        try:
            snake = active_contour(cropf, init, alpha=alpha, beta=beta, gamma=gamma, max_num_iter=SNAKE_MAX_ITERS)
        except TypeError:
            snake = active_contour(cropf, init, alpha=alpha, beta=beta, gamma=gamma, max_iterations=SNAKE_MAX_ITERS)

        sn = np.round(snake).astype(int)
        sn[:,0] = np.clip(sn[:,0], 0, maxr-minr-1)
        sn[:,1] = np.clip(sn[:,1], 0, maxc-minc-1)
        pr, pc = sn[:,0], sn[:,1]
        rr_fill, cc_fill = polygon(pr, pc, shape=crop.shape)
        out[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    return out

def iou_binary(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter)/max(1.0, float(union))

def boundary_map(mask):
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))

# ---------------------------- Objective -------------------------------------

def objective(mask_bool, chan_u8, base_params, jitter_eval_fn):
    # invalid?
    A = int(mask_bool.sum())
    if A == 0: return -1e9
    # Edge align
    by = boundary_map(mask_bool)
    if by.sum() > 0:
        gx = cv2.Sobel(chan_u8, cv2.CV_32F, 1,0, ksize=3)
        gy = cv2.Sobel(chan_u8, cv2.CV_32F, 0,1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        ys, xs = np.where(by > 0)
        edge_align = float(grad[ys, xs].mean()) / 255.0
    else:
        edge_align = 0.0

    # Complexity: components/area + circularity
    cc, _ = cv2.connectedComponents(mask_bool.astype(np.uint8))
    perim = float((by>0).sum())
    circ = (perim*perim) / (max(1.0, 4.0*np.pi*A))
    complexity = (cc-1)/max(1.0, A) + circ

    # Stability
    ious = []
    for _ in range(4):
        jparams = jitter_params(base_params, frac=0.05, bounds=BOUNDS)
        m2 = jitter_eval_fn(jparams)
        ious.append(iou_binary(mask_bool, m2) if m2.sum()>0 else 0.0)
    stability = float(np.median(ious)) if ious else 0.0

    return W_EDGE*edge_align + W_STAB*stability - W_COMP*complexity

def jitter_params(p, frac, bounds):
    out = {}
    for k, v in p.items():
        lo, hi = bounds[k]
        span = (hi - lo) * frac
        out[k] = float(np.clip(v + np.random.uniform(-span, span), lo, hi))
    # keep energy_threshold integer
    out["energy_threshold"] = int(round(out["energy_threshold"]))
    return out

# ---------------------------- Runner (one image) ----------------------------

def run_one_mask(chan_u8, leaf_mask, params):
    """Full diffusion+snake for one param set -> boolean mask."""
    h, w = chan_u8.shape
    leaf_area = int((leaf_mask>0).sum())
    if leaf_area == 0:
        # if no leaf mask, allow entire image area
        leaf_area = h*w
    l_size = max(5, int(LFRAC * leaf_area))
    u_size = max(50, int(UFRAC * leaf_area))

    # gradients (on lightly smoothed)
    ch_s = to_uint8(gaussian(chan_u8, 1.0))
    _, gx, gy = gradient_mag(ch_s)

    emap = elastic_diffusion(
        ch_s, gx, gy,
        iterations=30,
        diffusion_rate=params["diffusion_rate"],
        mu=params["mu"],
        lam=params["lam"],
        edge_thresh=params["energy_threshold"],
    )
    seg = snake_seg(
        ch_s, emap,
        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
        l_size=l_size, u_size=u_size,
        e_thr=params["energy_threshold"],
    )
    # constrain to leaf region if available
    if leaf_mask is not None:
        seg = np.logical_and(seg, leaf_mask>0)
    return seg

def evaluate_candidate(chan_u8, leaf_mask, params):
    m = run_one_mask(chan_u8, leaf_mask, params)
    return m

def optimize_for_image(bgr8, mode, base_name, out_dir, save_visual=True, rng=None,
                       budget_random=128, topk_refine=6, refine_steps=20):
    if rng is None: rng = np.random.default_rng()

    # downscale if necessary
    bgr8_small, scale = resize_if_needed(bgr8)
    gray8 = cv2.cvtColor(bgr8_small, cv2.COLOR_BGR2GRAY)
    leaf_mask = binary_leaf_mask(gray8, 127)

    # choose channel
    if mode == "bestch":
        chan = bgr8_small[...,1]  # G
    elif mode == "fused":
        chan = fused_channel_u8(bgr8_small, leaf_mask)
    else:
        raise ValueError("mode must be in {'bestch','fused'}")

    # Prepare candidate pool: curated seeds @ 30/70 + randoms
    cand_params = []
    for (mu,lam,dr,a,b,g) in CURATED:
        for e in CURATED_THRESH:
            cand_params.append(dict(mu=mu, lam=lam, diffusion_rate=dr,
                                    alpha=a, beta=b, gamma=g, energy_threshold=e))

    # Latin-ish random samples inside bounds
    keys = ["mu","lam","diffusion_rate","alpha","beta","gamma","energy_threshold"]
    for _ in range(budget_random):
        p = {}
        for k in keys:
            lo, hi = BOUNDS[k]
            if k == "energy_threshold":
                p[k] = int(rng.integers(lo, hi+1))
            else:
                p[k] = float(rng.uniform(lo, hi))
        cand_params.append(p)

    # Stage A: score all candidates (skip invalid quickly)
    scored = []
    t0 = time.time()
    for p in cand_params:
        def _jit_eval(jp):
            return run_one_mask(chan, leaf_mask, jp)
        mask = evaluate_candidate(chan, leaf_mask, p)
        if mask.sum() == 0 or mask.sum() > 0.5 * chan.size:
            score = -1e9
        else:
            score = objective(mask, chan, p, _jit_eval)
        scored.append((score, p, mask))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:topk_refine]

    # Stage B: local refine (simple coordinate search, tiny steps, few iters)
    def clampP(p):
        q = {}
        for k,v in p.items():
            lo, hi = BOUNDS[k]
            q[k] = float(np.clip(v, lo, hi))
        q["energy_threshold"] = int(round(np.clip(q["energy_threshold"], *BOUNDS["energy_threshold"])))
        return q

    def try_perturb(p, scale_frac=0.08):
        q = p.copy()
        for k in ["mu","lam","diffusion_rate","alpha","beta","gamma"]:
            span = (BOUNDS[k][1] - BOUNDS[k][0]) * scale_frac
            q[k] = p[k] + np.random.uniform(-span, span)
        q["energy_threshold"] = p["energy_threshold"] + int(np.random.randint(-4, 5))
        return clampP(q)

    best_score, best_p, best_m = top[0]
    for _, p0, _m0 in top:
        cur_s, cur_p, cur_m = _, p0, _m0
        for _ in range(refine_steps):
            cand = try_perturb(cur_p, scale_frac=0.06)
            m = evaluate_candidate(chan, leaf_mask, cand)
            if m.sum()==0 or m.sum()>0.5*chan.size:
                s = -1e9
            else:
                def _jit_eval(jp): return run_one_mask(chan, leaf_mask, jp)
                s = objective(m, chan, cand, _jit_eval)
            if s > cur_s:
                cur_s, cur_p, cur_m = s, cand, m
        if cur_s > best_score:
            best_score, best_p, best_m = cur_s, cur_p, cur_m

    # Save visual results (project mask back to original scale)
    out = {
        "image": base_name,
        "mode": mode,
        "score": float(best_score),
        **{k: float(v) if k!="energy_threshold" else int(v) for k,v in best_p.items()},
        "scale_used": float(scale),
        "pixels": int(chan.size),
        "mask_area": int(best_m.sum())
    }

    if save_visual:
        # Make overlay on original resolution
        # Build 16b overlay base from original
        overlay16 = _overlay16_from_bgr(bgr8)
        mask_up = best_m.astype(np.uint8)
        if scale != 1.0:
            H, W = bgr8.shape[:2]
            mask_up = cv2.resize(mask_up, (W, H), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay16, contours, -1, PINK_16, thickness=2)
        mask16 = (mask_up.astype(np.uint16)) * 65535
    else:
        overlay16 = None
        mask16 = None

    return out, overlay16, mask16, best_p

def _overlay16_from_bgr(bgr8):
    # turn 8-bit BGR into 16-bit 3ch
    return (bgr8.astype(np.uint16)) * 257

# ---------------------------- Main (batch) ----------------------------------

def main():
    p = argparse.ArgumentParser(description="Robust per-image Navier+Snake optimizer")
    p.add_argument("--input",  required=True, help="Folder of images")
    p.add_argument("--output", required=True, help="Output folder")
    p.add_argument("--mode", choices=["bestch","fused"], default="bestch",
                   help="Channel mode: bestch=G, fused=max(G,GRAY,Lab a*)+CLAHE")
    p.add_argument("--budget-random", type=int, default=128,
                   help="Random samples in Stage A (default 128)")
    p.add_argument("--topk-refine", type=int, default=6,
                   help="Top-K from Stage A to refine (default 6)")
    p.add_argument("--refine-steps", type=int, default=20,
                   help="Per-candidate coordinate steps (default 20)")
    p.add_argument("--log", default="results_robust_opt.csv",
                   help="CSV log filename")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # thread sanity
    for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try: cv2.setNumThreads(1)
    except Exception: pass

    rng = np.random.default_rng(args.seed)

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    ensure_dir(out_dir)

    rows = []
    images = [f for f in sorted(in_dir.iterdir()) if f.suffix.lower() in ALLOWED_EXT]
    print(f"[info] found {len(images)} images")

    for imgp in images:
        raw, bgr8, gray8 = read_color_anydepth(imgp)
        if raw is None:
            print(f"[skip] cannot read {imgp.name}")
            continue
        print(f"[run] {imgp.name} …", flush=True)

        try:
            out, overlay16, mask16, bestp = optimize_for_image(
                bgr8, args.mode, imgp.stem, out_dir, save_visual=True, rng=rng,
                budget_random=args.budget_random,
                topk_refine=args.topk_refine,
                refine_steps=args.refine_steps
            )
        except Exception as e:
            print(f"[error] {imgp.name}: {e}")
            continue

        # save files
        ov_path = out_dir / f"{imgp.stem}__best__overlay16.tif"
        mk_path = out_dir / f"{imgp.stem}__best__mask16.tif"
        js_path = out_dir / f"{imgp.stem}__best__params.json"
        if overlay16 is not None:
            cv2.imwrite(str(ov_path), overlay16)
            cv2.imwrite(str(mk_path), mask16)
        with open(js_path, "w") as f:
            json.dump(out, f, indent=2)

        # append CSV row
        rows.append(out)

    # write CSV
    if rows:
        df = pd.DataFrame(rows)
        csv_path = out_dir / args.log
        if csv_path.exists():
            old = pd.read_csv(csv_path)
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"[ok] wrote {csv_path}")
    else:
        print("[warn] no results to write")

if __name__ == "__main__":
    main()
