#!/usr/bin/env python3
"""Generate publication-ready montages comparing teacher / global / local masks
on selected patches.

Reads theta from patch_local_opt_v2.csv, regenerates masks via deterministic
forward passes (no re-optimization), and renders overlay montages.

Outputs:
  montage_patch_transfer_clear.png       – 2 rows (best/worst dD)
  montage_patch_transfer_top6.png        – 6 rows (3 worst + 3 best dD)
  *_singlecol.png / *_doublecol.png      – width variants for IEEEtran
"""

import csv, os, sys
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
from matplotlib.patches import FancyBboxPatch

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
V2_CSV = ROOT / "experiments/exp_v1/outputs/patch_local_opt_v2.csv"
MORPH_CSV = ROOT / "experiments/exp_v1/outputs/patch_transfer_morphology_eval.csv"
PATCH_IMG_DIR = ROOT / "leaves_patches"
PATCH_MASK_DIR = ROOT / "mask_patches"
OUT_DIR = ROOT / "experiments/exp_v1/outputs/patch_transfer_morphology_plots"

SMALL_MIN_AREA = 16
BG_THRESH = 5

# ── Physics helpers (from patch_transfer_morphology_eval.py) ──────────────────

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
    energy = elastic_energy(
        smag, mag, gx, gy, sgx, sgy,
        iters=30,
        diffusion_rate=params["diffusion_rate"],
        mu=params["mu"],
        lambda_param=params["lambda_param"],
        edge_thr=int(params["energy_threshold"]),
    )
    base_thresh = int(params["energy_threshold"])
    seg_outer = snake_seg(
        work_u8, energy, its=100,
        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
        thr=base_thresh, max_blob_frac=1.0,
    )
    outer_u8 = seg_outer.astype(np.uint8) * 255
    energy_masked = apply_mask(energy, outer_u8)
    thresh_inner = min(254, base_thresh + 25)
    mask_inner = (energy_masked > thresh_inner).astype(np.uint8)
    return mask_inner

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


# ── Overlay rendering ────────────────────────────────────────────────────────

def overlay_fill(rgb, mask_bool, color, alpha=0.35):
    """Semi-transparent fill overlay."""
    out = rgb.copy()
    out[mask_bool] = (
        (1 - alpha) * out[mask_bool].astype(np.float32) +
        alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return out

def draw_contour(rgb, mask_bool, color, thickness=2):
    """Draw contour on RGB image."""
    out = rgb.copy()
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out

def add_border(rgb, color=(40, 40, 40), thickness=2):
    """Add a border around an image."""
    out = rgb.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)
    return out


# ── Montage building ─────────────────────────────────────────────────────────

def make_row_images(row_data):
    """Given a row dict with paths and theta, produce 4 overlay images + metrics dict."""
    pimg_path = PATCH_IMG_DIR / row_data["patch_image"]
    pmask_path = PATCH_MASK_DIR / row_data["patch_mask"]

    bgr = cv2.imread(str(pimg_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"  WARNING: cannot read {pimg_path}")
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Teacher mask
    teacher_raw = cv2.imread(str(pmask_path), cv2.IMREAD_UNCHANGED)
    if teacher_raw is None:
        print(f"  WARNING: cannot read {pmask_path}")
        return None
    if teacher_raw.ndim == 3:
        teacher_raw = teacher_raw[..., 0]
    teacher_bool = teacher_raw > 0

    # Work channel
    gray8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    leaf_mask = cv2.threshold(gray8, 127, 255, cv2.THRESH_BINARY)[1]
    work = fused_channel_u8(bgr, leaf_mask)

    # Deterministic forward passes
    theta_g = parse_theta(row_data, "theta_global")
    theta_l = parse_theta(row_data, "theta_local")

    global_mask = finalize_inner_mask(work, theta_g).astype(bool)
    local_mask  = finalize_inner_mask(work, theta_l).astype(bool)

    # Build overlay images
    # 1) Raw
    img_raw = add_border(rgb.copy())

    # 2) Teacher: green fill
    img_teacher = overlay_fill(rgb, teacher_bool, [0, 220, 0], alpha=0.4)
    img_teacher = draw_contour(img_teacher, teacher_bool, (0, 180, 0), 2)
    img_teacher = add_border(img_teacher)

    # 3) Global: blue fill + contour
    img_global = overlay_fill(rgb, global_mask, [60, 120, 255], alpha=0.3)
    img_global = draw_contour(img_global, global_mask, (30, 80, 220), 2)
    img_global = add_border(img_global)

    # 4) Local: red fill + contour
    img_local = overlay_fill(rgb, local_mask, [255, 80, 60], alpha=0.3)
    img_local = draw_contour(img_local, local_mask, (220, 40, 30), 2)
    img_local = add_border(img_local)

    # Metrics
    dice_gt = float(row_data["dice_global_teacher"])
    dice_lt = float(row_data["dice_local_teacher"])
    dD = dice_lt - dice_gt

    metrics = {
        "dice_gt": dice_gt,
        "dice_lt": dice_lt,
        "dD": dD,
        "t_count": row_data.get("teacher_count", "?"),
        "g_count": row_data.get("global_count", "?"),
        "l_count": row_data.get("local_count", "?"),
        "t_afrac": row_data.get("teacher_area_frac", "?"),
        "g_afrac": row_data.get("global_area_frac", "?"),
        "l_afrac": row_data.get("local_area_frac", "?"),
        "patch_type": row_data.get("patch_type", "?"),
        "leaf_id": row_data.get("leaf_id", "?"),
        "patch_image": row_data.get("patch_image", "?"),
    }

    return img_raw, img_teacher, img_global, img_local, metrics


def build_montage(selected_rows, out_path, figwidth=7.16, dpi=250, title_text=None):
    """Build a multi-row, 4-column montage figure."""
    n_rows = len(selected_rows)
    col_titles = ["Raw Patch", "Teacher (green)", "Global θ* (blue)", "Local θ*_local (red)"]

    # Generate all row data
    all_images = []
    for row_data in selected_rows:
        result = make_row_images(row_data)
        if result is None:
            continue
        all_images.append(result)

    if not all_images:
        print(f"  No valid rows for {out_path}")
        return

    n_rows = len(all_images)
    # Each row: 4 image cells + metrics annotation below
    row_height = 1.8
    fig_height = n_rows * row_height + 0.7  # extra for top title + col headers
    if title_text:
        fig_height += 0.4

    fig, axes = plt.subplots(n_rows, 4, figsize=(figwidth, fig_height),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Title
    if title_text:
        fig.suptitle(title_text, fontsize=9, fontweight="bold", y=0.99)

    for i, (img_raw, img_teacher, img_global, img_local, metrics) in enumerate(all_images):
        imgs = [img_raw, img_teacher, img_global, img_local]
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=7, fontweight="bold", pad=4)

        # Row label + metrics
        leaf_short = metrics["leaf_id"][:12]
        ptype = metrics["patch_type"]
        dD = metrics["dD"]
        sign = "+" if dD >= 0 else ""

        # Format area fracs
        def _fmt_af(v):
            try:
                return f"{float(v):.3f}"
            except (ValueError, TypeError):
                return str(v)

        def _fmt_cnt(v):
            try:
                return f"{int(float(v))}"
            except (ValueError, TypeError):
                return str(v)

        label_line1 = f"{leaf_short} | {ptype}"
        label_line2 = (
            f"Dice(G,T)={metrics['dice_gt']:.3f}  "
            f"Dice(L,T)={metrics['dice_lt']:.3f}  "
            f"ΔD={sign}{dD:.3f}"
        )
        label_line3 = (
            f"count T/G/L={_fmt_cnt(metrics['t_count'])}/{_fmt_cnt(metrics['g_count'])}/{_fmt_cnt(metrics['l_count'])}  "
            f"area_frac T/G/L={_fmt_af(metrics['t_afrac'])}/{_fmt_af(metrics['g_afrac'])}/{_fmt_af(metrics['l_afrac'])}"
        )

        # Add text annotation below the row
        axes[i, 0].set_ylabel(label_line1, fontsize=6, rotation=0,
                              labelpad=80, ha="right", va="center")

        # Metrics text below the row, spanning all columns
        fig.text(0.5, axes[i, 0].get_position().y0 - 0.005, label_line2,
                 ha="center", va="top", fontsize=5.5, fontfamily="monospace",
                 color="#333333")
        fig.text(0.5, axes[i, 0].get_position().y0 - 0.025, label_line3,
                 ha="center", va="top", fontsize=5.0, fontfamily="monospace",
                 color="#555555")

    # Definition note at bottom
    fig.text(0.5, 0.005,
             "ΔD = Dice(Local, Teacher) − Dice(Global, Teacher).  Positive ΔD → local re-optimization improved agreement.",
             ha="center", va="bottom", fontsize=5.5, fontstyle="italic", color="#666666")

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.1,
                facecolor="white")
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load v2 CSV (has theta columns)
    v2_rows = {}
    with open(V2_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = row["patch_image"]
            v2_rows[key] = row

    # Load morphology CSV (has count/area metrics)
    morph_rows = {}
    with open(MORPH_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = row["patch_image"]
            morph_rows[key] = row

    # Merge: use v2 for theta, morph for metrics
    merged = []
    for key in v2_rows:
        r = dict(v2_rows[key])
        if key in morph_rows:
            mr = morph_rows[key]
            for col in ["teacher_area_px", "teacher_area_frac", "teacher_count",
                        "global_area_px", "global_area_frac", "global_count",
                        "local_area_px", "local_area_frac", "local_count"]:
                r[col] = mr.get(col, "?")
        merged.append(r)

    # Compute dD for all
    for r in merged:
        r["_dD"] = float(r["dice_local_teacher"]) - float(r["dice_global_teacher"])

    sorted_by_dD = sorted(merged, key=lambda r: r["_dD"])

    print(f"Loaded {len(merged)} patches. dD range: [{sorted_by_dD[0]['_dD']:.4f}, {sorted_by_dD[-1]['_dD']:.4f}]")

    # ── Montage 1: 2-row (worst + best dD) ──
    worst1 = sorted_by_dD[0]
    best1 = sorted_by_dD[-1]
    selected_2 = [worst1, best1]
    print("\n--- Montage: 2-row (worst/best dD) ---")
    print(f"  Worst dD: {worst1['patch_image']} dD={worst1['_dD']:.4f}")
    print(f"  Best  dD: {best1['patch_image']}  dD={best1['_dD']:.4f}")

    # Double-column version
    build_montage(
        selected_2,
        OUT_DIR / "montage_patch_transfer_clear.png",
        figwidth=7.16, dpi=250,
        title_text="Patch Transfer Comparison: Worst vs Best ΔDice(Local−Global)"
    )
    # Single-column version
    build_montage(
        selected_2,
        OUT_DIR / "montage_patch_transfer_clear_singlecol.png",
        figwidth=3.5, dpi=300,
        title_text="Patch Transfer: Worst vs Best ΔD"
    )

    # ── Montage 2: 6-row (3 worst + 3 best dD) ──
    worst3 = sorted_by_dD[:3]
    best3 = sorted_by_dD[-3:][::-1]  # reverse so best is first
    selected_6 = worst3 + best3
    print("\n--- Montage: 6-row (3 worst + 3 best dD) ---")
    for r in worst3:
        print(f"  Worst: {r['patch_image']} dD={r['_dD']:.4f}")
    for r in best3:
        print(f"  Best:  {r['patch_image']} dD={r['_dD']:.4f}")

    build_montage(
        selected_6,
        OUT_DIR / "montage_patch_transfer_top6.png",
        figwidth=7.16, dpi=250,
        title_text="Patch Transfer: 3 Worst (top) + 3 Best (bottom) ΔDice(Local−Global)"
    )
    # Single-column version
    build_montage(
        selected_6,
        OUT_DIR / "montage_patch_transfer_top6_singlecol.png",
        figwidth=3.5, dpi=300,
        title_text="3 Worst + 3 Best ΔD"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
