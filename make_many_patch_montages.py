#!/usr/bin/env python3
"""Generate analysis CSV, summary, and 12 montage sets for patch-level
morphology transfer evaluation.

No stochastic optimization — only deterministic forward passes from stored
theta_global / theta_local in patch_local_opt_v2.csv.

Usage (from repo root, with venv active):
    python3 make_many_patch_montages.py
"""

import csv, math, os, random, sys, time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import laplace
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.draw import polygon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MORPH_CSV = ROOT / "experiments/exp_v1/outputs/patch_transfer_morphology_eval.csv"
V2_CSV    = ROOT / "experiments/exp_v1/outputs/patch_local_opt_v2.csv"
PATCH_IMG_DIR  = ROOT / "leaves_patches"
PATCH_MASK_DIR = ROOT / "mask_patches"

BASE_OUT = ROOT / "experiments/exp_v1/outputs/patch_transfer_morphology_montages"
ANALYSIS_DIR = BASE_OUT / "analysis"
MONTAGE_DIR  = BASE_OUT / "montages"

SMALL_MIN_AREA = 16
BG_THRESH = 5
ROWS_PER_PAGE = 10          # split montages taller than this

# ── Physics helpers (deterministic forward pass) ──────────────────────────────

def _to_float01(img):
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)

def apply_mask(u8, mask_u8):
    return cv2.bitwise_and(u8, u8, mask=mask_u8) if mask_u8 is not None else u8

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
    return gx, gy, np.sqrt(gx * gx + gy * gy)

def build_seed(gx, gy, sigma=1.5):
    sgx = cv2.GaussianBlur(gx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sgy = cv2.GaussianBlur(gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return sgx, sgy, np.sqrt(sgx * sgx + sgy * sgy)

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
        diff += diffusion_rate * (mu * lap + (lambda_param + mu) * div +
                                  edge_mask * (ftm - diff))
    diff = np.maximum(diff, 0.0)
    mx = float(diff.max())
    if mx < 1e-8:
        return np.zeros_like(diff, dtype=np.uint8)
    return np.uint8(255.0 * diff / mx)

def snake_seg(img_u8, energy_u8, its, alpha, beta, gamma,
              min_blob_frac=1e-7, max_blob_frac=1.0, thr=50):
    h, w = img_u8.shape
    img_area = h * w
    l_size = max(SMALL_MIN_AREA, int(min_blob_frac * img_area))
    u_size = int(max_blob_frac * img_area)
    labeled_arr = label(energy_u8 > thr)
    props = regionprops(labeled_arr)
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

def finalize_inner_mask(work_u8, params):
    gx, gy, mag = compute_grads(work_u8)
    sgx, sgy, smag = build_seed(gx, gy, 1.5)
    energy = elastic_energy(smag, mag, gx, gy, sgx, sgy, iters=30,
                            diffusion_rate=params["diffusion_rate"],
                            mu=params["mu"], lambda_param=params["lambda_param"],
                            edge_thr=int(params["energy_threshold"]))
    base_thresh = int(params["energy_threshold"])
    seg_outer = snake_seg(work_u8, energy, its=100,
                          alpha=params["alpha"], beta=params["beta"],
                          gamma=params["gamma"], thr=base_thresh,
                          max_blob_frac=1.0)
    outer_u8 = seg_outer.astype(np.uint8) * 255
    energy_masked = apply_mask(energy, outer_u8)
    thresh_inner = min(254, base_thresh + 25)
    return (energy_masked > thresh_inner).astype(np.uint8)

def parse_theta(row, prefix):
    return {
        "mu":              float(row[f"{prefix}_mu"]),
        "lambda_param":    float(row[f"{prefix}_lambda"]),
        "diffusion_rate":  float(row[f"{prefix}_diffusion_rate"]),
        "alpha":           float(row[f"{prefix}_alpha"]),
        "beta":            float(row[f"{prefix}_beta"]),
        "gamma":           float(row[f"{prefix}_gamma"]),
        "energy_threshold": int(float(row[f"{prefix}_energy_threshold"])),
    }

# ── Overlay helpers ───────────────────────────────────────────────────────────

def overlay_fill(rgb, mask_bool, color, alpha=0.35):
    out = rgb.copy()
    if mask_bool.any():
        out[mask_bool] = (
            (1 - alpha) * out[mask_bool].astype(np.float32) +
            alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
    return out

def draw_contour(rgb, mask_bool, color, thickness=2):
    out = rgb.copy()
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out

def add_border(rgb, color=(40, 40, 40), thickness=2):
    out = rgb.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)
    return out

# ── Row rendering (returns 4 RGB images + metrics dict) ──────────────────────

def render_row(row):
    """Returns (img_raw, img_teacher, img_global, img_local) or None on failure."""
    pimg_path = PATCH_IMG_DIR / row["patch_image"]
    pmask_path = PATCH_MASK_DIR / row["patch_mask"]
    bgr = cv2.imread(str(pimg_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    teacher_raw = cv2.imread(str(pmask_path), cv2.IMREAD_UNCHANGED)
    if teacher_raw is None:
        return None
    if teacher_raw.ndim == 3:
        teacher_raw = teacher_raw[..., 0]
    teacher_bool = teacher_raw > 0

    gray8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    leaf_mask = cv2.threshold(gray8, 127, 255, cv2.THRESH_BINARY)[1]
    work = fused_channel_u8(bgr, leaf_mask)

    theta_g = parse_theta(row, "theta_global")
    theta_l = parse_theta(row, "theta_local")
    global_mask = finalize_inner_mask(work, theta_g).astype(bool)
    local_mask  = finalize_inner_mask(work, theta_l).astype(bool)

    img_raw     = add_border(rgb.copy())
    img_teacher = add_border(draw_contour(
                      overlay_fill(rgb, teacher_bool, [0, 220, 0], 0.40),
                      teacher_bool, (0, 180, 0), 2))
    img_global  = add_border(draw_contour(
                      overlay_fill(rgb, global_mask, [60, 120, 255], 0.30),
                      global_mask, (30, 80, 220), 2))
    img_local   = add_border(draw_contour(
                      overlay_fill(rgb, local_mask, [255, 80, 60], 0.30),
                      local_mask, (220, 40, 30), 2))

    return img_raw, img_teacher, img_global, img_local

# ── Montage figure builder (handles pagination) ──────────────────────────────

COL_TITLES = ["Raw Patch", "Teacher (green)", "Global \u03b8* (blue)",
              "Local \u03b8*_local (red)"]

def _fmt(v, decimals=3):
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)

def _fmti(v):
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v)

def build_page(rows_subset, page_path, title, figwidth=18, dpi=200):
    """Build one page of a montage (up to ROWS_PER_PAGE rows)."""
    rendered = []
    for r in rows_subset:
        t0 = time.time()
        result = render_row(r)
        if result is None:
            continue
        rendered.append((result, r))

    if not rendered:
        return

    n = len(rendered)
    row_h = 1.7
    fig_h = n * row_h + 1.0
    fig, axes = plt.subplots(n, 4, figsize=(figwidth, fig_h),
                             gridspec_kw={"hspace": 0.58, "wspace": 0.06})
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.995)

    for i, ((img_raw, img_t, img_g, img_l), r) in enumerate(rendered):
        for j, img in enumerate([img_raw, img_t, img_g, img_l]):
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(COL_TITLES[j], fontsize=7.5,
                                     fontweight="bold", pad=4)

        # Row label
        dD = float(r.get("dD", 0))
        sign = "+" if dD >= 0 else ""
        leaf_short = r["leaf_id"][:14]
        ptype = r.get("patch_type", "?")
        cfrac = _fmt(r.get("content_frac", "?"), 2)

        lbl = f"{leaf_short} | {ptype} | cf={cfrac}"
        axes[i, 0].set_ylabel(lbl, fontsize=5.5, rotation=0,
                              labelpad=95, ha="right", va="center")

        # Metrics under row
        y0 = axes[i, 0].get_position().y0
        line1 = (f"Dice(G,T)={_fmt(r.get('dice_global_teacher','?'))}  "
                 f"Dice(L,T)={_fmt(r.get('dice_local_teacher','?'))}  "
                 f"\u0394D={sign}{dD:.3f}")
        line2 = (f"Count T/G/L="
                 f"{_fmti(r.get('teacher_count','?'))}/"
                 f"{_fmti(r.get('global_count','?'))}/"
                 f"{_fmti(r.get('local_count','?'))}  "
                 f"AreaFrac T/G/L="
                 f"{_fmt(r.get('teacher_area_frac','?'))}/"
                 f"{_fmt(r.get('global_area_frac','?'))}/"
                 f"{_fmt(r.get('local_area_frac','?'))}")
        fig.text(0.5, y0 - 0.003, line1, ha="center", va="top",
                 fontsize=5.5, fontfamily="monospace", color="#333")
        fig.text(0.5, y0 - 0.020, line2, ha="center", va="top",
                 fontsize=5.0, fontfamily="monospace", color="#555")

    fig.text(0.5, 0.003,
             "\u0394D = Dice(Local,Teacher) \u2212 Dice(Global,Teacher). "
             "Positive \u0394D \u2192 local re-optimization improved.",
             ha="center", va="bottom", fontsize=5.5, fontstyle="italic",
             color="#666")

    fig.savefig(str(page_path), dpi=dpi, bbox_inches="tight",
                pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"    Wrote {page_path.name}  ({n} rows)")


def build_montage_paginated(rows, stem, title, figwidth=18, dpi=200):
    """Build montage(s), splitting into pages if > ROWS_PER_PAGE rows."""
    if not rows:
        print(f"  SKIP {stem}: no rows")
        return
    pages = [rows[i:i + ROWS_PER_PAGE]
             for i in range(0, len(rows), ROWS_PER_PAGE)]
    if len(pages) == 1:
        build_page(pages[0], MONTAGE_DIR / f"{stem}.png", title,
                   figwidth, dpi)
    else:
        for pi, page in enumerate(pages, 1):
            page_title = f"{title}  (page {pi}/{len(pages)})"
            build_page(page, MONTAGE_DIR / f"{stem}_page{pi}.png",
                       page_title, figwidth, dpi)


# ══════════════════════════════════════════════════════════════════════════════
#  PART A — Analysis CSV + summary
# ══════════════════════════════════════════════════════════════════════════════

def load_and_merge():
    """Merge morphology eval CSV with v2 CSV (for theta columns)."""
    # Read v2 (has theta)
    v2 = {}
    with open(V2_CSV, newline="") as f:
        for r in csv.DictReader(f):
            v2[r["patch_image"]] = r

    # Read morphology (has counts/areas)
    rows = []
    with open(MORPH_CSV, newline="") as f:
        for r in csv.DictReader(f):
            merged = dict(r)
            key = r["patch_image"]
            if key in v2:
                # bring in theta columns
                for col in v2[key]:
                    if col.startswith("theta_"):
                        merged[col] = v2[key][col]
            rows.append(merged)
    return rows


def add_derived_columns(rows):
    """Add all derived columns in-place; return the list."""
    for r in rows:
        dgt = float(r["dice_global_teacher"])
        dlt = float(r["dice_local_teacher"])
        igt = float(r["iou_global_teacher"])
        ilt = float(r["iou_local_teacher"])
        gce = float(r["global_count_err"])
        lce = float(r["local_count_err"])
        gafe = float(r["global_area_err_frac"])
        lafe = float(r["local_area_err_frac"])
        taf = float(r["teacher_area_frac"])

        r["dD"]  = dlt - dgt
        r["dIoU"] = ilt - igt
        r["abs_count_err_global"]    = abs(gce)
        r["abs_count_err_local"]     = abs(lce)
        r["abs_area_frac_err_global"] = abs(gafe)
        r["abs_area_frac_err_local"]  = abs(lafe)
        r["delta_abs_count_err"]     = abs(lce) - abs(gce)
        r["delta_abs_area_frac_err"] = abs(lafe) - abs(gafe)
        r["flag_global_overfragment"] = int(gce > 0)
        r["flag_local_overfragment"]  = int(lce > 0)

        # Bins
        if r["dD"] > 0.02:
            r["dd_bin"] = "local_better"
        elif r["dD"] < -0.02:
            r["dd_bin"] = "local_worse"
        else:
            r["dd_bin"] = "tie"

        if taf < 0.05:
            r["severity_bin"] = "low"
        elif taf <= 0.20:
            r["severity_bin"] = "mid"
        else:
            r["severity_bin"] = "high"

    return rows


def write_analysis_csv(rows):
    out_path = ANALYSIS_DIR / "patch_transfer_morphology_analysis.csv"
    # Determine field order: original morph fields + derived
    base_fields = [
        "leaf_id", "patch_image", "patch_mask", "x", "y",
        "pad_right", "pad_bottom", "patch_type", "content_frac",
        "dice_global_teacher", "dice_local_teacher",
        "iou_global_teacher", "iou_local_teacher",
        "teacher_area_px", "teacher_area_frac", "teacher_count",
        "global_area_px", "global_area_frac", "global_count",
        "local_area_px", "local_area_frac", "local_count",
        "global_area_err_px", "global_area_err_frac", "global_count_err",
        "local_area_err_px", "local_area_err_frac", "local_count_err",
    ]
    derived_fields = [
        "dD", "dIoU",
        "abs_count_err_global", "abs_count_err_local",
        "abs_area_frac_err_global", "abs_area_frac_err_local",
        "delta_abs_count_err", "delta_abs_area_frac_err",
        "flag_global_overfragment", "flag_local_overfragment",
        "dd_bin", "severity_bin",
    ]
    fields = base_fields + derived_fields

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_path}  ({len(rows)} rows)")


def write_summary(rows):
    out_path = ANALYSIS_DIR / "summary.txt"

    def _stats(vals):
        a = np.array(vals, dtype=float)
        return float(np.mean(a)), float(np.median(a)), len(a)

    all_dD = [r["dD"] for r in rows]
    int_dD = [r["dD"] for r in rows if r["patch_type"] == "interior"]
    bnd_dD = [r["dD"] for r in rows if r["patch_type"] == "boundary"]

    n_better = sum(1 for d in all_dD if d > 0.02)
    n_worse  = sum(1 for d in all_dD if d < -0.02)
    n_tie    = len(all_dD) - n_better - n_worse

    mean_ace_g = np.mean([r["abs_count_err_global"] for r in rows])
    mean_ace_l = np.mean([r["abs_count_err_local"] for r in rows])
    mean_aafe_g = np.mean([r["abs_area_frac_err_global"] for r in rows])
    mean_aafe_l = np.mean([r["abs_area_frac_err_local"] for r in rows])

    sorted_dD = sorted(rows, key=lambda r: r["dD"])
    top10 = sorted_dD[-10:][::-1]
    bot10 = sorted_dD[:10]

    lines = [
        "Patch Transfer Morphology — Analysis Summary",
        "=" * 60,
        "",
        f"Total patches: {len(rows)}",
        f"  Interior: {len(int_dD)}, Boundary: {len(bnd_dD)}",
        "",
        "--- dD = Dice(Local,Teacher) - Dice(Global,Teacher) ---",
        f"  All:      mean={_stats(all_dD)[0]:+.4f}  median={_stats(all_dD)[1]:+.4f}  n={_stats(all_dD)[2]}",
        f"  Interior: mean={_stats(int_dD)[0]:+.4f}  median={_stats(int_dD)[1]:+.4f}  n={_stats(int_dD)[2]}",
        f"  Boundary: mean={_stats(bnd_dD)[0]:+.4f}  median={_stats(bnd_dD)[1]:+.4f}  n={_stats(bnd_dD)[2]}",
        "",
        f"Fraction where local improves Dice by >0.02: {n_better}/{len(all_dD)} = {n_better/len(all_dD):.1%}",
        f"Fraction where local worsens Dice by >0.02:  {n_worse}/{len(all_dD)} = {n_worse/len(all_dD):.1%}",
        f"Fraction tie (|dD|<=0.02):                   {n_tie}/{len(all_dD)} = {n_tie/len(all_dD):.1%}",
        "",
        "--- Absolute Count Error ---",
        f"  Global: mean={mean_ace_g:.2f}",
        f"  Local:  mean={mean_ace_l:.2f}",
        f"  Improvement: {mean_ace_g - mean_ace_l:+.2f} (positive = local better)",
        "",
        "--- Absolute Area-Frac Error ---",
        f"  Global: mean={mean_aafe_g:.4f}",
        f"  Local:  mean={mean_aafe_l:.4f}",
        f"  Improvement: {mean_aafe_g - mean_aafe_l:+.4f} (positive = local better)",
        "",
        "--- Top 10 best dD (local better) ---",
    ]
    for r in top10:
        lines.append(f"  dD={r['dD']:+.4f}  {r['patch_image']}  {r['patch_type']}")
    lines.append("")
    lines.append("--- Bottom 10 worst dD (local worse) ---")
    for r in bot10:
        lines.append(f"  dD={r['dD']:+.4f}  {r['patch_image']}  {r['patch_type']}")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PART B — Montage generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_montages(rows):
    interior = [r for r in rows if r["patch_type"] == "interior"]
    boundary = [r for r in rows if r["patch_type"] == "boundary"]

    # Sort helpers
    by_dD_asc  = sorted(rows, key=lambda r: r["dD"])
    by_dD_desc = sorted(rows, key=lambda r: r["dD"], reverse=True)
    int_dD_asc  = sorted(interior, key=lambda r: r["dD"])
    int_dD_desc = sorted(interior, key=lambda r: r["dD"], reverse=True)
    bnd_dD_asc  = sorted(boundary, key=lambda r: r["dD"])
    bnd_dD_desc = sorted(boundary, key=lambda r: r["dD"], reverse=True)

    montage_specs = [
        # (stem, title, selected_rows)
        ("montage_top20_dD_overall",
         "Top-20 \u0394D overall (local better)",
         by_dD_desc[:20]),

        ("montage_bottom20_dD_overall",
         "Bottom-20 \u0394D overall (local worse)",
         by_dD_asc[:20]),

        ("montage_top20_dD_interior",
         "Top-20 \u0394D interior (local better)",
         int_dD_desc[:20]),

        ("montage_bottom20_dD_interior",
         "Bottom-20 \u0394D interior (local worse)",
         int_dD_asc[:20]),

        ("montage_top20_dD_boundary",
         "Top-20 \u0394D boundary (local better)",
         bnd_dD_desc[:20]),

        ("montage_bottom20_dD_boundary",
         "Bottom-20 \u0394D boundary (local worse)",
         bnd_dD_asc[:20]),
    ]

    # Random-20 interior/boundary
    rng = random.Random(42)
    rand_int = rng.sample(interior, min(20, len(interior)))
    rand_bnd = rng.sample(boundary, min(20, len(boundary)))
    montage_specs.append((
        "montage_random20_interior_seed42",
        "Random-20 interior patches (seed=42)",
        rand_int,
    ))
    montage_specs.append((
        "montage_random20_boundary_seed42",
        "Random-20 boundary patches (seed=42)",
        rand_bnd,
    ))

    # Over-fragmentation: largest positive global_count_err
    by_gce_desc = sorted(rows, key=lambda r: float(r["global_count_err"]),
                         reverse=True)
    montage_specs.append((
        "montage_global_overfragment_top20",
        "Top-20 global over-fragmentation (largest global_count_err > 0)",
        by_gce_desc[:20],
    ))

    # Local fixes count: largest improvement in abs count error
    by_count_improve = sorted(
        rows,
        key=lambda r: r["abs_count_err_global"] - r["abs_count_err_local"],
        reverse=True,
    )
    montage_specs.append((
        "montage_local_improves_count_top20",
        "Top-20 where local most improves |count_err|",
        by_count_improve[:20],
    ))

    # Under-coverage: most negative global_area_err_frac
    by_area_under = sorted(rows, key=lambda r: float(r["global_area_err_frac"]))
    montage_specs.append((
        "montage_global_undercoverage_top20",
        "Top-20 global under-coverage (most negative area_err_frac)",
        by_area_under[:20],
    ))

    # High severity teacher
    high_sev = [r for r in rows if float(r["teacher_area_frac"]) > 0.35]
    high_sev = sorted(high_sev, key=lambda r: float(r["teacher_area_frac"]),
                      reverse=True)[:20]
    montage_specs.append((
        "montage_high_severity_teacher",
        f"High-severity teacher patches (area_frac > 0.35, n={len(high_sev)})",
        high_sev,
    ))

    total = len(montage_specs)
    for idx, (stem, title, sel) in enumerate(montage_specs, 1):
        print(f"\n[{idx}/{total}] {stem}  ({len(sel)} rows)")
        t0 = time.time()
        build_montage_paginated(sel, stem, title)
        print(f"    ({time.time() - t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(MONTAGE_DIR, exist_ok=True)

    print("Loading and merging CSVs...")
    rows = load_and_merge()
    print(f"  {len(rows)} patches loaded")

    print("\n=== PART A: Analysis CSV + Summary ===")
    rows = add_derived_columns(rows)
    write_analysis_csv(rows)
    write_summary(rows)

    print("\n=== PART B: Generating montages ===")
    generate_all_montages(rows)

    print("\nAll done.")


if __name__ == "__main__":
    main()
