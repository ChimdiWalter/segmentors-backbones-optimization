#!/usr/bin/env python3
"""Patch-level local optimization experiment.

For each selected patch, produces three masks:
  M_teacher_patch  – saved teacher mask crop (binary, inner/necrotic)
  M_global_on_patch – run finalize on patch using leaf's global θ*
  M_local_opt_patch – run patch-level optimizer search → θ_patch*

Computes Dice/IoU of global and local masks against teacher.

Usage:
  source ~/.venvs/lesegenv/bin/activate
  python3 patch_local_optimize_compare.py --n_leaves 3 --patches_per_leaf 10 \
      --seed 0 --budget 32 --refine 6 \
      --out_csv experiments/exp_v1/outputs/patch_local_opt_test.csv
"""

import argparse, csv, math, os, random, sys, time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import laplace
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.draw import polygon

# ── Constants ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
OPT_CSV = ROOT / "experiments" / "exp_v1" / "outputs" / "opt_summary_local.csv"
PATCH_META = ROOT / "leaves_patches" / "patch_metadata.csv"
SMALL_MIN_AREA = 16

# Objective weights (same as optimizer)
W_GRAD_ALIGN = 0.5
W_COLOR_DIST = 4.0
W_CONTOUR    = 2.0
W_AREA       = 0.05
W_SMALL_BASE = 0.1

TARGET_AREA_FRAC = 0.05
AREA_FRAC_TOL    = 0.05
MIN_AREA_FRAC_HARD = 1e-7
MAX_AREA_FRAC_HARD = 0.85

BOUNDS = {
    "mu":              (0.05, 0.60),
    "lambda_param":    (0.05, 0.60),
    "diffusion_rate":  (0.05, 0.35),
    "alpha":           (0.001, 0.05),
    "beta":            (0.01, 0.25),
    "gamma":           (0.10, 0.60),
    "energy_threshold": (5, 60),
}

# ── Physics helpers (from check_leaf_params_transfer_to_patches.py) ──────────

def _to_float01(img):
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def apply_mask(u8, mask_u8):
    return cv2.bitwise_and(u8, u8, mask=mask_u8) if mask_u8 is not None else u8


def binary_threshold_mask(gray8, threshold=127):
    _, m = cv2.threshold(gray8, threshold, 255, cv2.THRESH_BINARY)
    return m


def fused_channel_u8(bgr8, binary_mask=None):
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab)
    candidates = [bgr8[..., 1], gray, lab[..., 1], lab[..., 2]]
    fused = None
    for ch in candidates:
        chm = apply_mask(ch, binary_mask) if binary_mask is not None else ch
        chf = chm.astype(np.float32)
        mx = float(chf.max())
        chn = (255.0 * chf / mx).astype(np.uint8) if mx > 1e-8 else np.zeros_like(chm, np.uint8)
        fused = chn if fused is None else np.maximum(fused, chn)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(fused)
    return apply_mask(enh, binary_mask) if binary_mask is not None else enh


def compute_grads(u8):
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, mag


def build_seed(gx, gy, sigma=1.5):
    sgx = cv2.GaussianBlur(gx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sgy = cv2.GaussianBlur(gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sm  = np.sqrt(sgx * sgx + sgy * sgy)
    return sgx, sgy, sm


def elastic_energy(seed_mag, img_mag, img_gx, img_gy, seed_gx, seed_gy,
                   iters, diffusion_rate, mu, lambda_param, edge_thr):
    diff = seed_mag.astype(np.float64).copy()
    edge_mask = (img_mag > edge_thr).astype(np.float64)
    for _ in range(int(iters)):
        lap = laplace(diff)
        gy, gx = np.gradient(diff)
        div = gx + gy
        ftx = img_gx - seed_gx
        fty = img_gy - seed_gy
        ftm = np.sqrt(ftx * ftx + fty * fty)
        diff += diffusion_rate * (mu * lap + (lambda_param + mu) * div + edge_mask * (ftm - diff))
    diff = np.maximum(diff, 0.0)
    mx = float(diff.max())
    if mx < 1e-8:
        return np.zeros_like(diff, dtype=np.uint8)
    return np.uint8(255.0 * diff / mx)


def snake_seg(img_u8, energy_u8, its, alpha, beta, gamma,
              min_blob_frac=1e-7, max_blob_frac=0.85, thr=50):
    h, w = img_u8.shape
    img_area = h * w
    l_size = max(SMALL_MIN_AREA, int(min_blob_frac * img_area))
    u_size = int(max_blob_frac * img_area)

    labeled = label(energy_u8 > thr)
    props = regionprops(labeled)
    out = np.zeros((h, w), dtype=bool)

    for r in props:
        if not (l_size < r.area < u_size):
            continue
        minr, minc, maxr, maxc = r.bbox
        if (maxr - minr) < 5 or (maxc - minc) < 5:
            continue
        crop = img_u8[minr:maxr, minc:maxc]
        crop_f = gaussian(_to_float01(crop), 3)

        s = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        rr = (maxr - minr) / 2.0
        cc = (maxc - minc) / 2.0
        init = np.vstack([rr * np.sin(s) + rr, cc * np.cos(s) + cc]).T

        try:
            snake = active_contour(crop_f, init, alpha=alpha, beta=beta,
                                   gamma=gamma, max_num_iter=its)
        except TypeError:
            snake = active_contour(crop_f, init, alpha=alpha, beta=beta,
                                   gamma=gamma, max_iterations=its)

        si = np.round(snake).astype(int)
        si[:, 0] = np.clip(si[:, 0], 0, crop.shape[0] - 1)
        si[:, 1] = np.clip(si[:, 1], 0, crop.shape[1] - 1)
        rr_fill, cc_fill = polygon(si[:, 0], si[:, 1], shape=crop.shape)
        out[minr:maxr, minc:maxc][rr_fill, cc_fill] = True

    return out


# ── Inner-mask logic (replicates navier_optimize_robust_mostrecent.py) ───────

def finalize_inner_mask(work_u8, bgr8, params, max_blob_frac=1.0):
    """Run full physics pipeline and produce INNER (necrotic) mask.

    This replicates the two-stage process from the original optimizer:
    1. Outer snake segmentation
    2. Mask energy to outer region, then re-threshold at base + 25 for inner
    """
    gx, gy, mag = compute_grads(work_u8)
    sgx, sgy, smag = build_seed(gx, gy, 1.5)

    energy = elastic_energy(
        smag, mag, gx, gy, sgx, sgy,
        iters=params.get("iters_final", 30),
        diffusion_rate=params["diffusion_rate"],
        mu=params["mu"],
        lambda_param=params["lambda_param"],
        edge_thr=int(params["energy_threshold"]),
    )

    base_thresh = int(params["energy_threshold"])

    # Stage 1: outer snake
    seg_outer = snake_seg(
        work_u8, energy,
        its=params.get("snake_iters_final", 100),
        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
        thr=base_thresh,
        max_blob_frac=max_blob_frac,
    )

    # Stage 2: inner mask (energy masked to outer region, higher threshold)
    outer_u8 = seg_outer.astype(np.uint8) * 255
    energy_masked = apply_mask(energy, outer_u8)
    thresh_inner = min(254, base_thresh + 25)
    mask_inner = (energy_masked > thresh_inner).astype(np.uint8)

    return mask_inner, seg_outer.astype(np.uint8)


# ── Objective scoring (for local optimization) ──────────────────────────────

def grad_alignment_score(seg_mask_bool, grad_mag_u8):
    seg_u8 = seg_mask_bool.astype(np.uint8) * 255
    edges = cv2.Canny(seg_u8, 50, 150)
    edge_vals = grad_mag_u8[edges > 0]
    if edge_vals.size == 0:
        return 0.0
    edge_mean = float(edge_vals.mean()) / 255.0
    global_mean = float(grad_mag_u8.mean()) / 255.0 + 1e-6
    return edge_mean / global_mean


def color_distance_score(seg_mask_bool, img_lab, img_u8, fg_min_pixels=50):
    leaf_mask = img_u8 > 0
    fg_mask = seg_mask_bool & leaf_mask
    bg_mask = (~seg_mask_bool) & leaf_mask
    if fg_mask.sum() < fg_min_pixels or bg_mask.sum() < fg_min_pixels:
        return 0.0
    lesion_ab = img_lab[..., 1:3][fg_mask].astype(np.float64)
    bg_ab = img_lab[..., 1:3][bg_mask].astype(np.float64)
    dist = float(np.linalg.norm(lesion_ab.mean(axis=0) - bg_ab.mean(axis=0)))
    return dist / (math.sqrt(2.0) * 255.0)


def evaluate_params(work_u8, bgr8, params, max_blob_frac=1.0):
    """Evaluate parameter set; returns (score, mask_inner, mask_outer)."""
    gx, gy, mag = compute_grads(work_u8)
    sgx, sgy, smag = build_seed(gx, gy, 1.5)
    base_thresh = int(params["energy_threshold"])

    energy = elastic_energy(
        smag, mag, gx, gy, sgx, sgy,
        iters=params.get("iters_final", 30),
        diffusion_rate=params["diffusion_rate"],
        mu=params["mu"],
        lambda_param=params["lambda_param"],
        edge_thr=base_thresh,
    )

    seg_outer = snake_seg(
        work_u8, energy,
        its=params.get("snake_iters_final", 100),
        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
        thr=base_thresh,
        max_blob_frac=max_blob_frac,
    )

    # Inner mask
    outer_u8 = seg_outer.astype(np.uint8) * 255
    energy_masked = apply_mask(energy, outer_u8)
    thresh_inner = min(254, base_thresh + 25)
    mask_inner = (energy_masked > thresh_inner).astype(np.uint8)
    mask_inner_bool = mask_inner > 0

    # Score using OUTER mask (same as original optimizer objective)
    h, w = work_u8.shape
    img_area = h * w
    area_px = int(seg_outer.sum())
    area_frac = area_px / img_area if img_area > 0 else 0.0

    if area_frac > MAX_AREA_FRAC_HARD or area_frac < MIN_AREA_FRAC_HARD:
        return -1e9, mask_inner, seg_outer.astype(np.uint8)

    # Grad alignment on outer boundary
    mag_u8 = np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    grad_align = grad_alignment_score(seg_outer, mag_u8)

    # Color distance
    lab_img = cv2.cvtColor(bgr8, cv2.COLOR_BGR2Lab) if bgr8.ndim == 3 else None
    gray_for_mask = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY) if bgr8.ndim == 3 else work_u8
    color_dist = color_distance_score(seg_outer, lab_img, gray_for_mask) if lab_img is not None else 0.0

    # Connected components on outer mask
    labeled_outer = label(seg_outer)
    props_outer = regionprops(labeled_outer)
    valid_count = sum(1 for p in props_outer if p.area >= SMALL_MIN_AREA)
    small_count = sum(1 for p in props_outer if p.area < SMALL_MIN_AREA)

    # Area penalty
    delta = abs(area_frac - TARGET_AREA_FRAC)
    pen_area = max(0.0, delta - AREA_FRAC_TOL)

    score = (
        W_GRAD_ALIGN * grad_align
        + W_COLOR_DIST * color_dist
        + W_CONTOUR * math.log1p(valid_count)
        - W_SMALL_BASE * small_count
        - W_AREA * pen_area
    )

    return score, mask_inner, seg_outer.astype(np.uint8)


# ── Patch-level optimizer ────────────────────────────────────────────────────

def sample_candidate(rng):
    return {
        "mu":              rng.uniform(*BOUNDS["mu"]),
        "lambda_param":    rng.uniform(*BOUNDS["lambda_param"]),
        "diffusion_rate":  rng.uniform(*BOUNDS["diffusion_rate"]),
        "alpha":           rng.uniform(*BOUNDS["alpha"]),
        "beta":            rng.uniform(*BOUNDS["beta"]),
        "gamma":           rng.uniform(*BOUNDS["gamma"]),
        "energy_threshold": int(round(rng.uniform(*BOUNDS["energy_threshold"]))),
        "iters_final":     30,
        "snake_iters_final": 100,
    }


def clip_params(p):
    q = dict(p)
    for k in ["mu", "lambda_param", "diffusion_rate", "alpha", "beta", "gamma"]:
        lo, hi = BOUNDS[k]
        q[k] = max(lo, min(hi, q[k]))
    lo, hi = BOUNDS["energy_threshold"]
    q["energy_threshold"] = max(lo, min(hi, int(round(q["energy_threshold"]))))
    return q


def perturb(rng, p, scale):
    q = dict(p)
    for k in ["mu", "lambda_param", "diffusion_rate", "alpha", "beta", "gamma"]:
        lo, hi = BOUNDS[k]
        q[k] += rng.uniform(-1, 1) * scale * (hi - lo)
    lo, hi = BOUNDS["energy_threshold"]
    q["energy_threshold"] += int(round(rng.uniform(-1, 1) * scale * (hi - lo) * 0.2))
    return clip_params(q)


def optimize_patch(work_u8, bgr8, budget, refine_steps, rng):
    """Random-restart + hill-climbing on a single patch."""
    best_score = -1e18
    best_params = None
    best_inner = None

    # Phase 1: random exploration
    for _ in range(budget):
        cand = sample_candidate(rng)
        score, m_inner, _ = evaluate_params(work_u8, bgr8, cand, max_blob_frac=1.0)
        if score > best_score:
            best_score = score
            best_params = cand
            best_inner = m_inner

    if best_params is None:
        return None, None, -1e18

    # Phase 2: local refinement around best
    current_params = dict(best_params)
    current_score = best_score
    patience = 0
    for step in range(refine_steps):
        scale = 0.25 * (0.85 ** step)
        cand = perturb(rng, current_params, scale)
        score, m_inner, _ = evaluate_params(work_u8, bgr8, cand, max_blob_frac=1.0)
        if score > current_score + 1e-6:
            current_score = score
            current_params = cand
            patience = 0
            if score > best_score + 1e-6:
                best_score = score
                best_params = cand
                best_inner = m_inner
        else:
            patience += 1
        if patience >= 5:
            break

    return best_params, best_inner, best_score


# ── Metrics ──────────────────────────────────────────────────────────────────

def dice(a, b, eps=1e-6):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = float((a & b).sum())
    return (2 * inter + eps) / (float(a.sum()) + float(b.sum()) + eps)


def iou(a, b, eps=1e-6):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return (inter + eps) / (union + eps)


# ── Montage generation ───────────────────────────────────────────────────────

def overlay_mask(bgr, mask_bool, color=(0, 0, 255), alpha_blend=0.4):
    """Overlay mask on BGR image with semi-transparent color."""
    vis = bgr.copy()
    if mask_bool.any():
        vis[mask_bool] = (
            (1 - alpha_blend) * vis[mask_bool].astype(np.float32)
            + alpha_blend * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
    return vis


def make_leaf_montage(patch_results, leaf_id, out_path):
    """Create montage for one leaf: 3 rows (improvement, similar, worse).

    Each row: raw | teacher overlay | global-on-patch overlay | local-opt overlay
    """
    if not patch_results:
        return

    # Sort by dice improvement (local - global)
    sorted_by_delta = sorted(patch_results, key=lambda r: r["dice_local_teacher"] - r["dice_global_teacher"])

    # Pick worst delta, median delta, best delta
    picks = []
    if len(sorted_by_delta) >= 3:
        picks = [sorted_by_delta[0], sorted_by_delta[len(sorted_by_delta) // 2], sorted_by_delta[-1]]
    else:
        picks = sorted_by_delta[:3]

    rows = []
    TH = 128  # thumbnail height
    TW = 128  # thumbnail width
    for pr in picks:
        bgr = cv2.imread(pr["patch_image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bgr_small = cv2.resize(bgr, (TW, TH))

        m_teacher = pr["m_teacher"]
        m_global = pr["m_global"]
        m_local = pr["m_local"]

        def resize_mask(m):
            return cv2.resize(m.astype(np.uint8) * 255, (TW, TH), interpolation=cv2.INTER_NEAREST) > 127

        mt = resize_mask(m_teacher)
        mg = resize_mask(m_global)
        ml = resize_mask(m_local)

        ov_teacher = overlay_mask(bgr_small, mt, color=(0, 255, 0))
        ov_global = overlay_mask(bgr_small, mg, color=(255, 165, 0))
        ov_local = overlay_mask(bgr_small, ml, color=(0, 0, 255))

        delta = pr["dice_local_teacher"] - pr["dice_global_teacher"]
        # Add text label
        lbl = f"dD={delta:+.3f}"
        cv2.putText(ov_local, lbl, (2, TH - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        row = np.hstack([bgr_small, ov_teacher, ov_global, ov_local])
        rows.append(row)

    if not rows:
        return

    # Column labels
    label_h = 20
    col_w = TW
    header = np.ones((label_h, col_w * 4, 3), dtype=np.uint8) * 255
    labels = ["Raw", "Teacher", "Global", "Local"]
    for i, txt in enumerate(labels):
        cv2.putText(header, txt, (i * col_w + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    montage = np.vstack([header] + rows)
    cv2.imwrite(str(out_path), montage)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Patch-level local optimization comparison")
    ap.add_argument("--n_leaves", type=int, default=3)
    ap.add_argument("--patches_per_leaf", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=32, help="Random candidates per patch")
    ap.add_argument("--refine", type=int, default=6, help="Refinement steps per patch")
    ap.add_argument("--out_csv", type=str,
                    default=str(ROOT / "experiments" / "exp_v1" / "outputs" / "patch_local_opt_test.csv"))
    ap.add_argument("--out_dir", type=str, default=None, help="Directory for montage PNGs")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if args.out_dir is None:
        args.out_dir = str(Path(args.out_csv).parent)

    # ── Load per-leaf global params ──────────────────────────────────────
    leaf_params = {}
    with open(OPT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "").strip().lower() != "ok":
                continue
            stem = Path(row["filename"]).stem
            leaf_params[stem] = {
                "mu":              float(row["mu"]),
                "lambda_param":    float(row["lambda"]),
                "diffusion_rate":  float(row["diffusion_rate"]),
                "alpha":           float(row["alpha"]),
                "beta":            float(row["beta"]),
                "gamma":           float(row["gamma"]),
                "energy_threshold": int(float(row["energy_threshold"])),
                "iters_final":     30,
                "snake_iters_final": 100,
                "mode":            row.get("mode", "fused").strip() or "fused",
            }

    # ── Load patch metadata, group by leaf ───────────────────────────────
    by_leaf = {}  # stem -> list of row dicts
    with open(PATCH_META, newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row["patch_image"]).name.split("__y")[0]
            by_leaf.setdefault(stem, []).append(row)

    # Filter to leaves present in both
    common_leaves = [k for k in by_leaf if k in leaf_params]
    if len(common_leaves) < args.n_leaves:
        print(f"WARNING: only {len(common_leaves)} common leaves (requested {args.n_leaves})")
    selected_leaves = rng.sample(common_leaves, min(args.n_leaves, len(common_leaves)))
    print(f"Selected {len(selected_leaves)} leaves: {selected_leaves}")

    # ── Prepare output ───────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    fieldnames = [
        "leaf_id", "patch_image", "patch_mask", "x", "y", "pad_right", "pad_bottom",
        "patch_type",
        "dice_global_teacher", "dice_local_teacher",
        "iou_global_teacher", "iou_local_teacher",
        "dice_local_global",
        "theta_global_mu", "theta_global_lambda", "theta_global_diffusion_rate",
        "theta_global_alpha", "theta_global_beta", "theta_global_gamma",
        "theta_global_energy_threshold",
        "theta_local_mu", "theta_local_lambda", "theta_local_diffusion_rate",
        "theta_local_alpha", "theta_local_beta", "theta_local_gamma",
        "theta_local_energy_threshold",
        "local_score", "runtime_seconds",
    ]

    all_rows = []
    leaf_patch_results = {}  # for montages

    total_patches = args.patches_per_leaf * len(selected_leaves)
    done = 0

    for leaf_stem in selected_leaves:
        patches = by_leaf[leaf_stem]
        params_global = leaf_params[leaf_stem]

        # Split into interior and boundary
        interior = [p for p in patches if int(p["pad_right"]) == 0 and int(p["pad_bottom"]) == 0]
        boundary = [p for p in patches if int(p["pad_right"]) > 0 or int(p["pad_bottom"]) > 0]

        n_interior = min(args.patches_per_leaf // 2, len(interior))
        n_boundary = min(args.patches_per_leaf - n_interior, len(boundary))
        # If we couldn't get enough boundary, fill from interior
        if n_boundary < args.patches_per_leaf - n_interior:
            n_interior = min(args.patches_per_leaf - n_boundary, len(interior))

        sel_interior = rng.sample(interior, n_interior) if n_interior > 0 else []
        sel_boundary = rng.sample(boundary, n_boundary) if n_boundary > 0 else []
        selected_patches = [(p, "interior") for p in sel_interior] + [(p, "boundary") for p in sel_boundary]

        leaf_results_for_montage = []

        for patch_row, patch_type in selected_patches:
            t0 = time.time()
            patch_img_path = patch_row["patch_image"]
            patch_mask_path = patch_row["patch_mask"]
            px = int(patch_row["x"])
            py = int(patch_row["y"])
            pad_r = int(patch_row["pad_right"])
            pad_b = int(patch_row["pad_bottom"])

            # Load patch image
            bgr = cv2.imread(patch_img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  SKIP: cannot read {patch_img_path}")
                continue

            # Load teacher mask
            m_teacher_raw = cv2.imread(patch_mask_path, cv2.IMREAD_UNCHANGED)
            if m_teacher_raw is None:
                print(f"  SKIP: cannot read {patch_mask_path}")
                continue
            if m_teacher_raw.ndim == 3:
                m_teacher_raw = m_teacher_raw[..., 0]
            m_teacher = (m_teacher_raw > 0).astype(np.uint8)

            # Prepare working image (fused channel)
            gray8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            leaf_mask = binary_threshold_mask(gray8, 127)
            work = fused_channel_u8(bgr, leaf_mask)

            # ── M_global_on_patch: run with leaf's global θ* ──────────
            m_global_inner, _ = finalize_inner_mask(work, bgr, params_global, max_blob_frac=1.0)

            # ── M_local_opt_patch: run patch-level optimizer ──────────
            local_params, m_local_inner, local_score = optimize_patch(
                work, bgr, args.budget, args.refine, rng
            )
            elapsed = time.time() - t0

            if m_local_inner is None:
                m_local_inner = np.zeros_like(m_teacher)
                local_params = sample_candidate(rng)  # placeholder
                local_score = -1e9

            # ── Compute metrics ───────────────────────────────────────
            d_global_teacher = dice(m_global_inner, m_teacher)
            d_local_teacher  = dice(m_local_inner, m_teacher)
            d_local_global   = dice(m_local_inner, m_global_inner)
            i_global_teacher = iou(m_global_inner, m_teacher)
            i_local_teacher  = iou(m_local_inner, m_teacher)

            row_out = {
                "leaf_id": leaf_stem,
                "patch_image": os.path.basename(patch_img_path),
                "patch_mask": os.path.basename(patch_mask_path),
                "x": px, "y": py,
                "pad_right": pad_r, "pad_bottom": pad_b,
                "patch_type": patch_type,
                "dice_global_teacher": f"{d_global_teacher:.6f}",
                "dice_local_teacher":  f"{d_local_teacher:.6f}",
                "iou_global_teacher":  f"{i_global_teacher:.6f}",
                "iou_local_teacher":   f"{i_local_teacher:.6f}",
                "dice_local_global":   f"{d_local_global:.6f}",
                "theta_global_mu": params_global["mu"],
                "theta_global_lambda": params_global["lambda_param"],
                "theta_global_diffusion_rate": params_global["diffusion_rate"],
                "theta_global_alpha": params_global["alpha"],
                "theta_global_beta": params_global["beta"],
                "theta_global_gamma": params_global["gamma"],
                "theta_global_energy_threshold": params_global["energy_threshold"],
                "theta_local_mu": local_params["mu"],
                "theta_local_lambda": local_params["lambda_param"],
                "theta_local_diffusion_rate": local_params["diffusion_rate"],
                "theta_local_alpha": local_params["alpha"],
                "theta_local_beta": local_params["beta"],
                "theta_local_gamma": local_params["gamma"],
                "theta_local_energy_threshold": local_params["energy_threshold"],
                "local_score": f"{local_score:.4f}",
                "runtime_seconds": f"{elapsed:.2f}",
            }
            all_rows.append(row_out)

            leaf_results_for_montage.append({
                "patch_image_path": patch_img_path,
                "m_teacher": m_teacher,
                "m_global": m_global_inner,
                "m_local": m_local_inner,
                "dice_global_teacher": d_global_teacher,
                "dice_local_teacher": d_local_teacher,
            })

            done += 1
            print(f"  [{done}/{total_patches}] {os.path.basename(patch_img_path)} "
                  f"type={patch_type} "
                  f"Dice(glob,teacher)={d_global_teacher:.3f} "
                  f"Dice(local,teacher)={d_local_teacher:.3f} "
                  f"delta={d_local_teacher - d_global_teacher:+.3f} "
                  f"({elapsed:.1f}s)")

        leaf_patch_results[leaf_stem] = leaf_results_for_montage

        # Generate montage for this leaf
        montage_path = Path(args.out_dir) / f"montage_patch_opt_{leaf_stem}.png"
        make_leaf_montage(leaf_results_for_montage, leaf_stem, montage_path)
        if montage_path.exists():
            print(f"  Wrote montage: {montage_path}")

    # ── Write CSV ────────────────────────────────────────────────────────
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {args.out_csv} ({len(all_rows)} rows)")

    # ── Summary statistics ───────────────────────────────────────────────
    if not all_rows:
        print("No results to summarize.")
        return

    def stats(values):
        if not values:
            return {"mean": 0.0, "median": 0.0, "n": 0}
        values = sorted(values)
        n = len(values)
        mean = sum(values) / n
        median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
        return {"mean": mean, "median": median, "n": n}

    all_delta = []
    interior_delta = []
    boundary_delta = []

    for r in all_rows:
        d = float(r["dice_local_teacher"]) - float(r["dice_global_teacher"])
        all_delta.append(d)
        if r["patch_type"] == "interior":
            interior_delta.append(d)
        else:
            boundary_delta.append(d)

    all_stats = stats(all_delta)
    int_stats = stats(interior_delta)
    bnd_stats = stats(boundary_delta)

    all_dice_global = [float(r["dice_global_teacher"]) for r in all_rows]
    all_dice_local = [float(r["dice_local_teacher"]) for r in all_rows]

    print("\n" + "=" * 70)
    print("SUMMARY: Dice(local, teacher) - Dice(global, teacher)")
    print("=" * 70)
    print(f"  ALL patches (n={all_stats['n']}):      "
          f"mean delta = {all_stats['mean']:+.4f}, "
          f"median delta = {all_stats['median']:+.4f}")
    print(f"    Dice(global,teacher): mean={stats(all_dice_global)['mean']:.4f}, "
          f"median={stats(all_dice_global)['median']:.4f}")
    print(f"    Dice(local,teacher):  mean={stats(all_dice_local)['mean']:.4f}, "
          f"median={stats(all_dice_local)['median']:.4f}")
    print(f"  INTERIOR patches (n={int_stats['n']}): "
          f"mean delta = {int_stats['mean']:+.4f}, "
          f"median delta = {int_stats['median']:+.4f}")
    print(f"  BOUNDARY patches (n={bnd_stats['n']}): "
          f"mean delta = {bnd_stats['mean']:+.4f}, "
          f"median delta = {bnd_stats['median']:+.4f}")
    print()

    # ── Interpretation ───────────────────────────────────────────────────
    mean_all = all_stats["mean"]
    mean_int = int_stats["mean"]
    mean_bnd = bnd_stats["mean"]

    print("INTERPRETATION:")
    if abs(mean_all) < 0.03:
        print(f"  - Mean improvement |{mean_all:+.4f}| < 0.03:")
        print(f"    Per-leaf theta* is SUFFICIENT for patches.")
        print(f"    Local re-optimization provides negligible benefit.")
    else:
        print(f"  - Mean improvement {mean_all:+.4f} >= 0.03:")
        print(f"    Local re-optimization provides meaningful benefit.")

    if bnd_stats["n"] > 0 and int_stats["n"] > 0:
        if mean_bnd > mean_int + 0.02:
            print(f"  - Boundary patches show LARGER improvement ({mean_bnd:+.4f} vs {mean_int:+.4f}):")
            print(f"    Mismatch is boundary/context-driven.")
            print(f"    Padding/ROI gating is more appropriate than per-patch optimization.")
        elif mean_int > mean_bnd + 0.02:
            print(f"  - Interior patches show LARGER improvement ({mean_int:+.4f} vs {mean_bnd:+.4f}):")
            print(f"    Mismatch is content-driven, not boundary artifacts.")
        else:
            print(f"  - Interior ({mean_int:+.4f}) and boundary ({mean_bnd:+.4f}) show similar deltas:")
            print(f"    No clear boundary-specific effect.")


if __name__ == "__main__":
    main()
