#!/usr/bin/env python3
"""Filled-mask evaluation: deterministic forward passes with contour-filled
predicted masks, magenta teacher overlays, and comprehensive montages.

Reads theta_global / theta_local from patch_local_opt_v2.csv, runs finalize,
fills predicted masks via cv2.drawContours (thickness=-1), then computes
metrics and generates montages with filled overlays.

No stochastic optimization is performed.

Usage:
    python3 patch_transfer_morphology_eval_filled.py \
      --in_csv experiments/exp_v1/outputs/patch_local_opt_v2.csv \
      --patch_img_dir leaves_patches \
      --patch_mask_dir mask_patches \
      --out_dir experiments/exp_v1/outputs/patch_transfer_morphology_filled
"""

import argparse, csv, os, random, sys, time
from pathlib import Path
from collections import defaultdict

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

# ── Constants ─────────────────────────────────────────────────────────────────
SMALL_MIN_AREA = 16
BG_THRESH = 5
ROWS_PER_PAGE = 10

# Overlay colours (RGB)
COLOR_TEACHER = (220, 0, 220)    # magenta
COLOR_GLOBAL  = (50, 110, 255)   # blue
COLOR_LOCAL   = (240, 60, 50)    # red

# ── Physics helpers ───────────────────────────────────────────────────────────

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
    """Deterministic forward pass -> inner necrotic mask (unfilled)."""
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
    mask_inner = (energy_masked > thresh_inner).astype(np.uint8)
    return mask_inner


def fill_mask(mask_u8):
    """Contour-fill a binary mask to ensure solid filled regions.
    Finds all external contours and draws them filled (thickness=-1)."""
    filled = np.zeros_like(mask_u8)
    # Ensure input is proper uint8
    inp = (mask_u8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(inp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return (filled > 0).astype(np.uint8)


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

# ── Valid-pixel mask + metrics ────────────────────────────────────────────────

def build_valid_mask(gray8, pad_right, pad_bottom):
    h, w = gray8.shape
    valid = np.ones((h, w), dtype=bool)
    if pad_right > 0:
        valid[:, w - pad_right:] = False
    if pad_bottom > 0:
        valid[h - pad_bottom:, :] = False
    valid &= (gray8 > BG_THRESH)
    return valid

def dice_masked(a, b, valid, eps=1e-6):
    a = a.astype(bool) & valid
    b = b.astype(bool) & valid
    inter = float((a & b).sum())
    return (2 * inter + eps) / (float(a.sum()) + float(b.sum()) + eps)

def iou_masked(a, b, valid, eps=1e-6):
    a = a.astype(bool) & valid
    b = b.astype(bool) & valid
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return (inter + eps) / (union + eps)

def morphology_stats(mask_bool, valid):
    masked = mask_bool & valid
    area_px = int(masked.sum())
    valid_px = int(valid.sum())
    area_frac = area_px / valid_px if valid_px > 0 else 0.0
    labeled_arr = label(masked)
    props = regionprops(labeled_arr)
    lesion_count = sum(1 for p in props if p.area >= SMALL_MIN_AREA)
    return area_px, area_frac, lesion_count

# ── Overlay rendering ────────────────────────────────────────────────────────

def overlay_fill(rgb, mask_bool, color, alpha=0.40):
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

def add_border(rgb, color=(50, 50, 50), thickness=2):
    out = rgb.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)
    return out

# ── Process one patch → metrics dict + overlay images ─────────────────────────

def process_patch(row, patch_img_dir, patch_mask_dir):
    """Returns (result_dict, overlay_images) or None on failure."""
    pimg_path = patch_img_dir / row["patch_image"]
    pmask_path = patch_mask_dir / row["patch_mask"]

    bgr = cv2.imread(str(pimg_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    teacher_raw = cv2.imread(str(pmask_path), cv2.IMREAD_UNCHANGED)
    if teacher_raw is None:
        return None
    if teacher_raw.ndim == 3:
        teacher_raw = teacher_raw[..., 0]
    teacher_bool = (teacher_raw > 0)

    # Valid mask
    gray8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pad_right = int(row.get("pad_right", 0))
    pad_bottom = int(row.get("pad_bottom", 0))
    valid = build_valid_mask(gray8, pad_right, pad_bottom)

    # Work channel
    leaf_mask = cv2.threshold(gray8, 127, 255, cv2.THRESH_BINARY)[1]
    work = fused_channel_u8(bgr, leaf_mask)

    # Forward passes
    theta_g = parse_theta(row, "theta_global")
    theta_l = parse_theta(row, "theta_local")
    global_raw = finalize_inner_mask(work, theta_g)
    local_raw  = finalize_inner_mask(work, theta_l)

    # Fill masks
    global_filled = fill_mask(global_raw)
    local_filled  = fill_mask(local_raw)
    teacher_filled = fill_mask(teacher_bool.astype(np.uint8))

    g_bool = global_filled.astype(bool)
    l_bool = local_filled.astype(bool)
    t_bool = teacher_filled.astype(bool)

    # Metrics
    dice_gt = dice_masked(g_bool, t_bool, valid)
    dice_lt = dice_masked(l_bool, t_bool, valid)
    iou_gt  = iou_masked(g_bool, t_bool, valid)
    iou_lt  = iou_masked(l_bool, t_bool, valid)

    t_apx, t_af, t_cnt = morphology_stats(t_bool, valid)
    g_apx, g_af, g_cnt = morphology_stats(g_bool, valid)
    l_apx, l_af, l_cnt = morphology_stats(l_bool, valid)

    result = {
        "leaf_id": row["leaf_id"],
        "patch_image": row["patch_image"],
        "patch_mask": row["patch_mask"],
        "x": row["x"], "y": row["y"],
        "pad_right": pad_right, "pad_bottom": pad_bottom,
        "patch_type": row.get("patch_type", "unknown"),
        "content_frac": row.get("content_frac", ""),
        "dice_global_teacher": f"{dice_gt:.6f}",
        "dice_local_teacher":  f"{dice_lt:.6f}",
        "iou_global_teacher":  f"{iou_gt:.6f}",
        "iou_local_teacher":   f"{iou_lt:.6f}",
        "dD": f"{dice_lt - dice_gt:.6f}",
        "teacher_area_px": t_apx, "teacher_area_frac": f"{t_af:.6f}", "teacher_count": t_cnt,
        "global_area_px": g_apx,  "global_area_frac": f"{g_af:.6f}",  "global_count": g_cnt,
        "local_area_px": l_apx,   "local_area_frac": f"{l_af:.6f}",   "local_count": l_cnt,
        "global_area_err_px": g_apx - t_apx,
        "global_area_err_frac": f"{g_af - t_af:.6f}",
        "global_count_err": g_cnt - t_cnt,
        "local_area_err_px": l_apx - t_apx,
        "local_area_err_frac": f"{l_af - t_af:.6f}",
        "local_count_err": l_cnt - t_cnt,
    }

    # Overlay images for montages
    img_raw = add_border(rgb.copy())
    img_teacher = add_border(draw_contour(
        overlay_fill(rgb, t_bool, COLOR_TEACHER, 0.40),
        t_bool, (180, 0, 180), 2))
    img_global = add_border(draw_contour(
        overlay_fill(rgb, g_bool, COLOR_GLOBAL, 0.35),
        g_bool, (30, 80, 220), 2))
    img_local = add_border(draw_contour(
        overlay_fill(rgb, l_bool, COLOR_LOCAL, 0.35),
        l_bool, (200, 40, 30), 2))

    overlays = {
        "raw": img_raw,
        "teacher": img_teacher,
        "global": img_global,
        "local": img_local,
        "t_bool": t_bool,
        "g_bool": g_bool,
        "l_bool": l_bool,
    }

    return result, overlays


# ── Montage builder ───────────────────────────────────────────────────────────

COL_TITLES = ["Raw Patch", "Teacher (magenta)", "Global filled (blue)",
              "Local filled (red)"]

def _fmt(v, d=3):
    try: return f"{float(v):.{d}f}"
    except: return str(v)

def _fmti(v):
    try: return str(int(float(v)))
    except: return str(v)

def build_montage_page(items, page_path, title, figwidth=18, dpi=200):
    """items = list of (result_dict, overlays_dict)."""
    if not items:
        return
    n = len(items)
    # 4 rows per patch: Raw, Teacher, Global, Local
    # Use column-oriented layout: each COLUMN = one patch
    # But for many patches this gets too wide. Use row-oriented instead:
    # Each ROW block = one patch, 4 columns = Raw/T/G/L
    row_h = 1.65
    fig_h = n * row_h + 1.2
    fig, axes = plt.subplots(n, 4, figsize=(figwidth, fig_h),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.06})
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.997)

    for i, (res, ovl) in enumerate(items):
        imgs = [ovl["raw"], ovl["teacher"], ovl["global"], ovl["local"]]
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(COL_TITLES[j], fontsize=7, fontweight="bold", pad=3)

        dD = float(res["dD"])
        sign = "+" if dD >= 0 else ""
        ptype = res["patch_type"]
        cfrac = _fmt(res.get("content_frac", "?"), 2)
        leaf_short = res["leaf_id"][:14]

        axes[i, 0].set_ylabel(f"{leaf_short}\n{ptype} cf={cfrac}",
                              fontsize=5, rotation=0, labelpad=72,
                              ha="right", va="center")

        y0 = axes[i, 0].get_position().y0
        line1 = (f"Dice G/L={_fmt(res['dice_global_teacher'])}/{_fmt(res['dice_local_teacher'])}  "
                 f"\u0394D={sign}{dD:.3f}")
        line2 = (f"Count T/G/L={_fmti(res['teacher_count'])}/{_fmti(res['global_count'])}/{_fmti(res['local_count'])}  "
                 f"AreaFrac T/G/L={_fmt(res['teacher_area_frac'])}/{_fmt(res['global_area_frac'])}/{_fmt(res['local_area_frac'])}")
        fig.text(0.5, y0 - 0.002, line1, ha="center", va="top",
                 fontsize=5.5, fontfamily="monospace", color="#333")
        fig.text(0.5, y0 - 0.018, line2, ha="center", va="top",
                 fontsize=4.8, fontfamily="monospace", color="#555")

    fig.text(0.5, 0.002,
             "Filled masks: contour-fill applied. \u0394D = Dice(Local,Teacher) \u2212 Dice(Global,Teacher).",
             ha="center", va="bottom", fontsize=5.5, fontstyle="italic", color="#666")

    fig.savefig(str(page_path), dpi=dpi, bbox_inches="tight",
                pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print(f"    Wrote {page_path.name}  ({n} rows)")


def build_montage(items, stem, title, montage_dir, figwidth=18, dpi=200):
    """Build paginated montage."""
    if not items:
        print(f"  SKIP {stem}: no items")
        return
    pages = [items[i:i+ROWS_PER_PAGE] for i in range(0, len(items), ROWS_PER_PAGE)]
    if len(pages) == 1:
        build_montage_page(pages[0], montage_dir / f"{stem}.png", title, figwidth, dpi)
    else:
        for pi, page in enumerate(pages, 1):
            build_montage_page(page, montage_dir / f"{stem}_page{pi}.png",
                               f"{title} (page {pi}/{len(pages)})", figwidth, dpi)


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(results, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    g_af_err = [float(r["global_area_err_frac"]) for r in results]
    l_af_err = [float(r["local_area_err_frac"]) for r in results]
    g_cnt_err = [int(r["global_count_err"]) for r in results]
    l_cnt_err = [int(r["local_count_err"]) for r in results]
    t_af = [float(r["teacher_area_frac"]) for r in results]
    g_af = [float(r["global_area_frac"]) for r in results]
    l_af = [float(r["local_area_frac"]) for r in results]

    # 1) Histogram: area_frac error
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-0.5, 0.2, 50)
    ax.hist(g_af_err, bins=bins, alpha=0.6, label="Global", color="steelblue")
    ax.hist(l_af_err, bins=bins, alpha=0.6, label="Local", color="indianred")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Area-fraction error (pred - teacher)")
    ax.set_ylabel("Count")
    ax.set_title("Filled Masks: Area-Fraction Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(plot_dir / "hist_area_frac_error_filled.png"), dpi=200)
    plt.close(fig)

    # 2) Histogram: count error
    fig, ax = plt.subplots(figsize=(6, 4))
    all_cnt = g_cnt_err + l_cnt_err
    lo = min(all_cnt) - 2
    hi = max(all_cnt) + 2
    bins = np.arange(lo, hi + 1, max(1, (hi - lo) // 40))
    ax.hist(g_cnt_err, bins=bins, alpha=0.6, label="Global", color="steelblue")
    ax.hist(l_cnt_err, bins=bins, alpha=0.6, label="Local", color="indianred")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Count error (pred - teacher)")
    ax.set_ylabel("Count")
    ax.set_title("Filled Masks: Lesion-Count Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(plot_dir / "hist_count_error_filled.png"), dpi=200)
    plt.close(fig)

    # 3) Scatter: teacher vs pred area_frac
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(t_af, g_af, alpha=0.5, s=18, label="Global", color="steelblue", edgecolors="none")
    ax.scatter(t_af, l_af, alpha=0.5, s=18, label="Local", color="indianred", edgecolors="none")
    lim = max(max(t_af), max(g_af), max(l_af)) * 1.05 + 0.01
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Identity")
    ax.set_xlabel("Teacher area_frac")
    ax.set_ylabel("Predicted area_frac")
    ax.set_title("Filled Masks: Teacher vs Predicted Area Fraction")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(str(plot_dir / "scatter_teacher_vs_pred_area_frac_filled.png"), dpi=200)
    plt.close(fig)

    print(f"  Wrote 3 plots to {plot_dir}")


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(results, out_path):
    def _s(vals):
        a = np.array(vals, dtype=float)
        return float(np.mean(a)), float(np.median(a)), len(a)

    all_rows = results
    interior = [r for r in all_rows if r["patch_type"] == "interior"]
    boundary = [r for r in all_rows if r["patch_type"] == "boundary"]

    def _block(subset, label):
        if not subset:
            return [f"  {label}: (no data)"]
        dgt = [float(r["dice_global_teacher"]) for r in subset]
        dlt = [float(r["dice_local_teacher"]) for r in subset]
        dD = [float(r["dD"]) for r in subset]
        igt = [float(r["iou_global_teacher"]) for r in subset]
        ilt = [float(r["iou_local_teacher"]) for r in subset]
        taf = [float(r["teacher_area_frac"]) for r in subset]
        gaf = [float(r["global_area_frac"]) for r in subset]
        laf = [float(r["local_area_frac"]) for r in subset]
        ace_g = [abs(int(r["global_count_err"])) for r in subset]
        ace_l = [abs(int(r["local_count_err"])) for r in subset]
        aafe_g = [abs(float(r["global_area_err_frac"])) for r in subset]
        aafe_l = [abs(float(r["local_area_err_frac"])) for r in subset]
        n = len(subset)
        lines = [
            f"  {label} (n={n}):",
            f"    Dice(global,teacher): mean={np.mean(dgt):.4f}  median={np.median(dgt):.4f}",
            f"    Dice(local,teacher):  mean={np.mean(dlt):.4f}  median={np.median(dlt):.4f}",
            f"    dD (local-global):    mean={np.mean(dD):+.4f}  median={np.median(dD):+.4f}",
            f"    IoU(global,teacher):  mean={np.mean(igt):.4f}  median={np.median(igt):.4f}",
            f"    IoU(local,teacher):   mean={np.mean(ilt):.4f}  median={np.median(ilt):.4f}",
            f"    Teacher area_frac:    mean={np.mean(taf):.4f}",
            f"    Global area_frac:     mean={np.mean(gaf):.4f}",
            f"    Local area_frac:      mean={np.mean(laf):.4f}",
            f"    |area_frac_err| (G):  mean={np.mean(aafe_g):.4f}",
            f"    |area_frac_err| (L):  mean={np.mean(aafe_l):.4f}",
            f"    |count_err| (G):      mean={np.mean(ace_g):.2f}",
            f"    |count_err| (L):      mean={np.mean(ace_l):.2f}",
        ]
        return lines

    lines = [
        "Filled-Mask Patch Transfer Morphology Evaluation Summary",
        "=" * 62,
        f"Total patches: {len(all_rows)}",
        f"  Interior: {len(interior)}, Boundary: {len(boundary)}",
        f"Masks: contour-filled via cv2.drawContours (thickness=-1)",
        "",
    ]
    lines += _block(all_rows, "ALL")
    lines.append("")
    lines += _block(interior, "INTERIOR")
    lines.append("")
    lines += _block(boundary, "BOUNDARY")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")

    # Also print to terminal
    for l in lines:
        print(l)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--patch_img_dir", required=True)
    ap.add_argument("--patch_mask_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    patch_img_dir = root / args.patch_img_dir
    patch_mask_dir = root / args.patch_mask_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    montage_dir = out_dir / "montages"
    plot_dir = out_dir / "plots"
    os.makedirs(montage_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Read input CSV
    in_rows = []
    with open(args.in_csv, newline="") as f:
        for row in csv.DictReader(f):
            in_rows.append(row)
    print(f"Read {len(in_rows)} patches from {args.in_csv}")

    # Process all patches
    results = []
    items_by_leaf = defaultdict(list)  # leaf_id -> [(result, overlays)]
    all_items = []  # [(result, overlays)]

    for idx, row in enumerate(in_rows):
        t0 = time.time()
        out = process_patch(row, patch_img_dir, patch_mask_dir)
        elapsed = time.time() - t0
        if out is None:
            print(f"  [{idx+1}/{len(in_rows)}] SKIP {row['patch_image']} (load failed)")
            continue
        res, ovl = out
        results.append(res)
        all_items.append((res, ovl))
        items_by_leaf[res["leaf_id"]].append((res, ovl))
        dD = float(res["dD"])
        sign = "+" if dD >= 0 else ""
        print(f"  [{idx+1}/{len(in_rows)}] {row['patch_image']}  "
              f"Dice(G,T)={_fmt(res['dice_global_teacher'])} "
              f"Dice(L,T)={_fmt(res['dice_local_teacher'])} "
              f"\u0394D={sign}{dD:.3f}  ({elapsed:.1f}s)")

    print(f"\nProcessed {len(results)} patches successfully.")

    # ── Write CSV ──
    csv_path = out_dir / "patch_transfer_morphology_filled_eval.csv"
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {csv_path} ({len(results)} rows)")

    # ── Write summary ──
    summary_path = out_dir / "patch_transfer_morphology_filled_summary.txt"
    write_summary(results, summary_path)

    # ── Write plots ──
    make_plots(results, plot_dir)

    # ── Build montages ──
    print("\n=== Generating montages ===")

    # Per-leaf montages
    print(f"\nPer-leaf montages ({len(items_by_leaf)} leaves):")
    for leaf_id in sorted(items_by_leaf.keys()):
        leaf_items = items_by_leaf[leaf_id]
        leaf_short = leaf_id[:20]
        build_montage(leaf_items,
                      f"montage_filled_patch_transfer_{leaf_id}",
                      f"Leaf: {leaf_short} (n={len(leaf_items)}, filled masks)",
                      montage_dir)

    # Sort items by various criteria for selection montages
    by_dD = sorted(all_items, key=lambda x: float(x[0]["dD"]))
    by_dice_g = sorted(all_items, key=lambda x: float(x[0]["dice_global_teacher"]))
    by_dice_l = sorted(all_items, key=lambda x: float(x[0]["dice_local_teacher"]))
    by_cnt_err_g = sorted(all_items, key=lambda x: abs(int(x[0]["global_count_err"])), reverse=True)
    by_cnt_err_l = sorted(all_items, key=lambda x: abs(int(x[0]["local_count_err"])), reverse=True)
    by_af_err_g = sorted(all_items, key=lambda x: abs(float(x[0]["global_area_err_frac"])), reverse=True)
    by_af_err_l = sorted(all_items, key=lambda x: abs(float(x[0]["local_area_err_frac"])), reverse=True)

    interior_items = [x for x in all_items if x[0]["patch_type"] == "interior"]
    boundary_items = [x for x in all_items if x[0]["patch_type"] == "boundary"]

    selection_specs = [
        ("montage_filled_worst_dice_global", "Worst Dice(Global,Teacher) — filled", by_dice_g[:20]),
        ("montage_filled_worst_dice_local", "Worst Dice(Local,Teacher) — filled", by_dice_l[:20]),
        ("montage_filled_best_dD", "Best \u0394D (local better) — filled", by_dD[-20:][::-1]),
        ("montage_filled_worst_dD", "Worst \u0394D (local worse) — filled", by_dD[:20]),
        ("montage_filled_top_count_err_global", "Largest |count_err| Global — filled", by_cnt_err_g[:20]),
        ("montage_filled_top_count_err_local", "Largest |count_err| Local — filled", by_cnt_err_l[:20]),
        ("montage_filled_top_area_frac_err_global", "Largest |area_frac_err| Global — filled", by_af_err_g[:20]),
        ("montage_filled_top_area_frac_err_local", "Largest |area_frac_err| Local — filled", by_af_err_l[:20]),
        ("montage_filled_interior_subset", "Interior patches (all) — filled", interior_items),
        ("montage_filled_boundary_subset", "Boundary patches (all) — filled", boundary_items),
    ]

    print(f"\nSelection montages ({len(selection_specs)}):")
    for stem, title, sel in selection_specs:
        print(f"\n  {stem} ({len(sel)} items)")
        build_montage(sel, stem, title, montage_dir)

    print("\n=== All done ===")

    # Final listing
    print(f"\nOutput directory: {out_dir}")
    for dirpath, dirnames, filenames in os.walk(out_dir):
        level = dirpath.replace(str(out_dir), "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(dirpath)}/")
        for fn in sorted(filenames):
            fpath = os.path.join(dirpath, fn)
            sz = os.path.getsize(fpath)
            if sz > 1024 * 1024:
                szs = f"{sz / 1024 / 1024:.1f}M"
            elif sz > 1024:
                szs = f"{sz / 1024:.0f}K"
            else:
                szs = f"{sz}B"
            print(f"{indent}  {fn}  ({szs})")


if __name__ == "__main__":
    main()
