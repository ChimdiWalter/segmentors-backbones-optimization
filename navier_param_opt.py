#!/usr/bin/env python3
import os, sys, csv, time, math, signal, argparse, random, itertools, traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.ndimage import laplace
from skimage.draw import polygon

# ----------------------------- Safety / Threads -----------------------------
for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ----------------------------- Defaults -------------------------------------
ALLOWED_EXT   = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
PINK_16       = (65535, 5140, 37779)     # 16-bit (B,G,R)
CONTOUR_MODE  = cv2.RETR_EXTERNAL
APPROX_MODE   = cv2.CHAIN_APPROX_SIMPLE

# Safe parameter bounds
BOUNDS = {
    "mu":              (0.03, 0.80),
    "lambda":          (0.03, 0.80),
    "diffusion_rate":  (0.05, 0.60),
    "alpha":           (0.005, 0.60),
    "beta":            (0.05,  1.20),
    "gamma":           (0.005, 0.60),
    "energy_threshold":(20, 120),  # integer
}

# ----------------------------- I/O utils ------------------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
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

def _prepare_overlay16(raw):
    if raw.dtype == np.uint16:
        if raw.ndim == 2:  return cv2.merge([raw, raw, raw])
        if raw.ndim == 3:  return raw.copy() if raw.shape[2]==3 else cv2.merge([raw[...,0]]*3)
    # fallback from 8-bit or other
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
    return (raw8.astype(np.uint16)) * 257

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
                                  mu=0.5, lambda_param=0.5, edge_thresh=50):
    # initialize from gradient magnitude of (image - pregrad)
    diffused = np.sqrt(grad_x**2 + grad_y**2).astype(np.float64)

    im_gx = cv2.Sobel(image_u8, cv2.CV_64F, 1, 0, ksize=3)
    im_gy = cv2.Sobel(image_u8, cv2.CV_64F, 0, 1, ksize=3)
    im_mag = np.sqrt(im_gx**2 + im_gy**2)
    q = (im_mag > edge_thresh).astype(np.float64)

    prev = diffused.copy()
    for _ in range(iterations):
        lap = laplace(diffused)
        div_v = np.gradient(diffused)[0]   # simple proxy divergence term
        ftx = im_gx - grad_x
        fty = im_gy - grad_y
        ftm = np.sqrt(ftx**2 + fty**2)
        diffused += diffusion_rate * (mu * lap + (lambda_param + mu) * div_v + q * (ftm - diffused))
        # simple residual check to avoid blow-ups
        if np.any(np.isnan(diffused)) or np.any(np.isinf(diffused)):
            return None, np.inf
        step_res = float(np.mean(np.abs(diffused - prev)))
        prev = diffused.copy()

    eps = 1e-8
    out = np.uint8(255 * diffused / max(np.max(diffused), eps))
    # residual (lower is better) — used as feasibility gate, not as objective
    return out, step_res

def snake_seg(image_u8, energy_u8, alpha, beta, gamma, l_size, u_size, energy_threshold):
    h, w = image_u8.shape
    labeled = label(energy_u8 > energy_threshold)
    props = regionprops(labeled)
    out_mask = np.zeros((h, w), dtype=bool)
    if len(props) == 0:
        return out_mask

    for rgn in props:
        if not (l_size < rgn.area < u_size):  # size gate
            continue
        minr, minc, maxr, maxc = rgn.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5:
            continue

        crop_u8 = image_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop_u8), 3)

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

def fused_channel_u8(bgr8, binary_mask=None):
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)
    candidates = [bgr8[..., 1], gray, lab[..., 1]]  # G, GRAY, a*
    fused = None
    for ch in candidates:
        chm = apply_mask(ch, binary_mask) if binary_mask is not None else ch
        chf = chm.astype(np.float32); mx = float(chf.max())
        chn = (255.0 * chf / mx).astype(np.uint8) if mx > 1e-8 else np.zeros_like(chm, dtype=np.uint8)
        fused = chn if fused is None else np.maximum(fused, chn)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(fused)
    return apply_mask(enh, binary_mask) if binary_mask is not None else enh

# ----------------------------- Objective ------------------------------------
def mask_perimeter(mask):
    # perimeter via contours
    u8 = (mask.astype(np.uint8))*255
    cnts, _ = cv2.findContours(u8, CONTOUR_MODE, APPROX_MODE)
    peri = 0.0
    for c in cnts: peri += cv2.arcLength(c, True)
    return peri

def objective_unsupervised(mask_bool, grad_mag_u8):
    """
    Larger is better.
    Components: edge adherence, compactness, size sanity, component penalty.
    """
    if mask_bool is None: return -1e9
    area = float(np.sum(mask_bool))
    if area < 10: return -1e9  # empty/near-empty
    h, w = mask_bool.shape
    if area > 0.6 * h * w:  # avoid swallowing background
        return -1e9

    # Edge adherence: average gradient at boundary pixels
    # compute boundary by erosion
    kernel = np.ones((3,3), np.uint8)
    er = cv2.erode((mask_bool*255).astype(np.uint8), kernel, iterations=1)
    boundary = ((mask_bool.astype(np.uint8)*255) & (cv2.bitwise_not(er))).astype(bool)
    if not np.any(boundary): return -1e9
    edge_score = float(np.mean(grad_mag_u8[boundary])) / 255.0  # 0..1

    # Compactness: 4πA/P^2 (0..1)
    P = mask_perimeter(mask_bool)
    if P <= 1e-6: return -1e9
    compact = float((4.0 * math.pi * area) / (P*P))
    compact = max(0.0, min(1.0, compact))

    # Component penalty
    lab = label(mask_bool)
    ncomp = int(lab.max())

    # Size prior: prefer moderate area fraction ~ 0.02..0.20 (tweakable)
    af = area / float(h*w)
    if   af < 0.02: size_bonus = -0.2*(0.02 - af)/0.02
    elif af > 0.20: size_bonus = -0.2*(af - 0.20)/0.20
    else:           size_bonus = +0.05

    # Combine
    # weights tuned for stability; adjust as needed
    J = (0.55 * edge_score) + (0.35 * compact) + (0.10 * size_bonus) - (0.03 * max(0, ncomp-1))
    return J

# ---------------------- Trial (with timeout guard) --------------------------
def one_trial(params, mode, bgr8, gray8, raw, scale=1.0):
    """
    Returns (score, seg_mask_bool, residual, meta_dict) or (-inf, None, inf, ...) on failure.
    """
    try:
        if scale != 1.0:
            bgr8s = cv2.resize(bgr8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray8s= cv2.resize(gray8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            bgr8s, gray8s = bgr8, gray8

        bin_mask = binary_threshold_mask(gray8s, threshold=127)

        if mode == "bestch":
            ch = bgr8s[...,1]  # G
            work = apply_mask(ch, bin_mask)
        elif mode == "fused":
            work = fused_channel_u8(bgr8s, bin_mask)
        else:
            raise ValueError("mode must be in {bestch, fused}")

        grad_u8, gx, gy = gradient_magnitude(work)

        # PDE
        diff, resid = elastic_deformation_diffusion(
            work, gx, gy,
            iterations=30,
            diffusion_rate=float(params["diffusion_rate"]),
            mu=float(params["mu"]),
            lambda_param=float(params["lambda"]),
            edge_thresh=int(params["energy_threshold"])
        )
        if diff is None or not np.isfinite(resid) or resid > 1e6:
            return -1e9, None, np.inf, {"reason":"pde_fail"}

        # Snake
        seg = snake_seg(
            work, diff,
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            gamma=float(params["gamma"]),
            l_size=5, u_size=50000,
            energy_threshold=int(params["energy_threshold"])
        )

        score = objective_unsupervised(seg, grad_u8)
        return score, seg, resid, {}
    except Exception as e:
        return -1e9, None, np.inf, {"reason": f"exception:{e}"}

def sample_params(rng):
    p = {}
    for k,(lo,hi) in BOUNDS.items():
        if k == "energy_threshold":
            p[k] = int(rng.integers(lo, hi+1))
        else:
            p[k] = float(rng.random()*(hi-lo) + lo)
    # small correlations to keep snakes from exploding at high curvature
    if p["alpha"] < 0.02 and p["beta"] > 0.8:
        p["beta"] = 0.8 * p["beta"]
    return p

# ------------------------- Optimizer loop -----------------------------------
def optimize_image(image_path, out_dir, mode,
                   seed=42,
                   coarse_trials=40, coarse_scale=0.5, coarse_timeout=6.0,
                   refine_k=6, refine_trials=12, refine_timeout=10.0,
                   no_improve_stop=20):
    """
    Returns dict with best params, score, and paths.
    """
    ensure_dir(out_dir)
    raw, bgr8, gray8 = read_color_anydepth(image_path)
    if raw is None:
        return {"status":"skip", "reason":"load_fail", "image": image_path}

    base = os.path.splitext(os.path.basename(image_path))[0]
    overlay16_base = _prepare_overlay16(raw)

    rng = np.random.default_rng(seed + (hash(base) % (10**6)))

    # Resume-safe CSV
    log_csv = os.path.join(out_dir, f"{base}__opt_log.csv")
    best_csv= os.path.join(out_dir, f"{base}__best.csv")

    best = {"score": -1e9, "params": None}
    stagnation = 0

    def timed_trial(params, scale, timeout_s):
        with ProcessPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(one_trial, params, mode, bgr8, gray8, raw, scale)
            try:
                return fut.result(timeout=timeout_s)
            except TimeoutError:
                return -1e9, None, np.inf, {"reason":"timeout"}

    # 1) Coarse screening on downscaled image
    for t in range(coarse_trials):
        params = sample_params(rng)
        score, seg, resid, meta = timed_trial(params, coarse_scale, coarse_timeout)

        with open(log_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["coarse", t, score, resid, params])

        if score > best["score"]:
            best = {"score": score, "params": params, "seg": seg}
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= no_improve_stop:
                break

    if best["params"] is None:
        return {"status":"fail", "reason":"no_feasible_params", "image": image_path}

    # 1b) Keep top-k from coarse for refinement
    # Re-read the coarse log to pick top K (robust if resumed).
    coarse_rows = []
    if os.path.exists(log_csv):
        with open(log_csv, "r") as f:
            for row in csv.reader(f):
                if len(row) >= 5 and row[0] == "coarse":
                    try:
                        coarse_rows.append((float(row[2]), row[4]))  # (score, params_repr)
                    except:
                        pass
    coarse_rows.sort(key=lambda x: x[0], reverse=True)
    seeds_for_refine = []
    for s, preg in coarse_rows[:max(1, refine_k)]:
        # eval-safe literal parser:
        try:
            p = eval(preg, {"__builtins__":None}, {})
            if isinstance(p, dict): seeds_for_refine.append(p)
        except Exception:
            continue
    if not seeds_for_refine:
        seeds_for_refine = [best["params"]]

    # 2) Refinement on full-res around the seeds
    def jitter(p, scale=0.2):
        q = dict(p)
        for k,(lo,hi) in BOUNDS.items():
            if k == "energy_threshold":
                span = hi-lo
                val  = int(round(p[k] + random.uniform(-0.2,0.2)*span))
                q[k] = max(lo, min(hi, val))
            else:
                span = hi-lo
                val  = p[k] + np.random.normal(0.0, scale*span)
                q[k] = float(max(lo, min(hi, val)))
        return q

    # evaluate seeds first
    for idx, sparams in enumerate(seeds_for_refine):
        score, seg, resid, meta = timed_trial(sparams, 1.0, refine_timeout)
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(["seed", idx, score, resid, sparams])
        if score > best["score"]:
            best = {"score": score, "params": sparams, "seg": seg}

    stagnation = 0
    for t in range(refine_trials):
        p0 = random.choice(seeds_for_refine)
        params = jitter(p0, scale=0.18)
        score, seg, resid, meta = timed_trial(params, 1.0, refine_timeout)

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(["refine", t, score, resid, params])

        if score > best["score"]:
            best = {"score": score, "params": params, "seg": seg}
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= no_improve_stop:
                break

    # Save best mask + overlay
    if best["params"] is None or best["seg"] is None:
        return {"status":"fail", "reason":"no_valid_mask", "image": image_path}

    seg_bool = best["seg"]
    seg_u8 = (seg_bool.astype(np.uint8))*255
    contours, _ = cv2.findContours(seg_u8, CONTOUR_MODE, APPROX_MODE)

    out_overlay16 = overlay16_base.copy()
    cv2.drawContours(out_overlay16, contours, -1, PINK_16, thickness=2)
    mask16 = (seg_bool.astype(np.uint16))*65535

    tag = f"opt_mu{fmt(best['params']['mu'])}_lam{fmt(best['params']['lambda'])}_d{fmt(best['params']['diffusion_rate'])}" \
          f"_a{fmt(best['params']['alpha'])}_b{fmt(best['params']['beta'])}_g{fmt(best['params']['gamma'])}_e{int(best['params']['energy_threshold'])}"

    overlay_path = os.path.join(out_dir, f"{base}__{tag}__overlay16.tif")
    mask_path    = os.path.join(out_dir, f"{base}__{tag}__mask16.tif")
    cv2.imwrite(overlay_path, out_overlay16)
    cv2.imwrite(mask_path,    mask16)

    with open(best_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image","mode","score","overlay16_path","mask16_path"])
        w.writerow([image_path, mode, best["score"], overlay_path, mask_path])
        w.writerow(["params"])
        for k in ["mu","lambda","diffusion_rate","alpha","beta","gamma","energy_threshold"]:
            w.writerow([k, best["params"][k]])

    return {
        "status":"ok",
        "image": image_path,
        "mode": mode,
        "score": best["score"],
        "overlay": overlay_path,
        "mask": mask_path,
        "params": best["params"]
    }

# ----------------------------- CLI ------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Robust per-image parameter optimization for Navier+Snake.")
    ap.add_argument("--mode", choices=["bestch","fused"], required=True,
                    help="bestch = green channel only; fused = max(G, Gray, Lab a*)+CLAHE")
    ap.add_argument("--input", required=True, help="Folder with input images")
    ap.add_argument("--output", required=True, help="Output folder for optimized masks/overlays/logs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coarse-trials", type=int, default=40)
    ap.add_argument("--coarse-scale", type=float, default=0.5)
    ap.add_argument("--coarse-timeout", type=float, default=6.0)
    ap.add_argument("--refine-k", type=int, default=6)
    ap.add_argument("--refine-trials", type=int, default=12)
    ap.add_argument("--refine-timeout", type=float, default=10.0)
    ap.add_argument("--no-improve-stop", type=int, default=20)
    ap.add_argument("--max-workers", type=int, default=0,
                    help="0/1 = sequential images; >1 = optimize multiple images in parallel (safe).")
    args = ap.parse_args()

    ensure_dir(args.output)
    files = [f for f in os.listdir(args.input) if f.lower().endswith(ALLOWED_EXT)]
    if not files:
        print("No input images found.")
        sys.exit(1)

    def run_one(fn):
        try:
            out_dir_img = os.path.join(args.output, os.path.splitext(fn)[0])
            return optimize_image(
                image_path=os.path.join(args.input, fn),
                out_dir=out_dir_img,
                mode=args.mode,
                seed=args.seed,
                coarse_trials=args.coarse_trials,
                coarse_scale=args.coarse_scale,
                coarse_timeout=args.coarse_timeout,
                refine_k=args.refine_k,
                refine_trials=args.refine_trials,
                refine_timeout=args.refine_timeout,
                no_improve_stop=args.no_improve_stop
            )
        except Exception as e:
            return {"status":"fail","image":fn,"reason":f"runner_exception:{e}"}

    # Parallelize across images if requested (each image has its own internal timeouts)
    if args.max_workers and args.max_workers > 1:
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futs = [ex.submit(run_one, fn) for fn in files]
            for i,f in enumerate(as_completed(futs),1):
                res = f.result()
                print(f"[{i}/{len(files)}] {res['status']} :: {res.get('image')}", flush=True)
    else:
        for i,fn in enumerate(files,1):
            res = run_one(fn)
            print(f"[{i}/{len(files)}] {res['status']} :: {res.get('image')}", flush=True)

if __name__ == "__main__":
    main()
