#!/usr/bin/env python3
"""Post-hoc morphology transfer test using existing patch experiment CSV.

For each patch in patch_local_opt_v2.csv, re-runs finalize (inner necrotic mask)
using theta_global and theta_local columns, then compares morphology metrics
(area, area_frac, lesion_count) against the saved teacher patch mask.

No stochastic search is performed — only deterministic forward passes.

Usage:
  python3 patch_transfer_morphology_eval.py \
    --in_csv experiments/exp_v1/outputs/patch_local_opt_v2.csv \
    --patch_img_dir leaves_patches \
    --patch_mask_dir mask_patches \
    --out_csv experiments/exp_v1/outputs/patch_transfer_morphology_eval.csv \
    --out_summary experiments/exp_v1/outputs/patch_transfer_morphology_summary.txt \
    --plot_dir experiments/exp_v1/outputs/patch_transfer_morphology_plots
"""

import argparse, csv, math, os, sys
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

# ── Constants ────────────────────────────────────────────────────────────────
SMALL_MIN_AREA = 16
BG_THRESH = 5  # pixels with gray <= this are deep background

# ── Physics helpers (identical to patch_local_optimize_compare_v2.py) ────────

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


def finalize_inner_mask(work_u8, params):
    """Deterministic forward pass: work channel -> inner necrotic mask."""
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
        thr=base_thresh,
        max_blob_frac=1.0,
    )

    outer_u8 = seg_outer.astype(np.uint8) * 255
    energy_masked = apply_mask(energy, outer_u8)
    thresh_inner = min(254, base_thresh + 25)
    mask_inner = (energy_masked > thresh_inner).astype(np.uint8)

    return mask_inner


# ── Valid-pixel mask + masked metrics ────────────────────────────────────────

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
    """Compute area_px, area_frac, lesion_count within valid region."""
    masked = mask_bool & valid
    area_px = int(masked.sum())
    valid_px = int(valid.sum())
    area_frac = area_px / valid_px if valid_px > 0 else 0.0
    # Count connected components >= SMALL_MIN_AREA within valid
    labeled_arr = label(masked)
    props = regionprops(labeled_arr)
    lesion_count = sum(1 for p in props if p.area >= SMALL_MIN_AREA)
    return area_px, area_frac, lesion_count


# ── Parse theta from CSV row ────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Post-hoc morphology transfer evaluation")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--patch_img_dir", required=True)
    ap.add_argument("--patch_mask_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary", required=True)
    ap.add_argument("--plot_dir", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    patch_img_dir = root / args.patch_img_dir
    patch_mask_dir = root / args.patch_mask_dir

    # Read input CSV
    in_rows = []
    with open(args.in_csv, newline="") as f:
        for row in csv.DictReader(f):
            in_rows.append(row)
    print(f"Read {len(in_rows)} patches from {args.in_csv}")

    # Output fieldnames
    out_fields = [
        "leaf_id", "patch_image", "patch_mask", "x", "y",
        "pad_right", "pad_bottom", "patch_type", "content_frac",
        # Dice/IoU (recomputed with valid masking)
        "dice_global_teacher", "dice_local_teacher",
        "iou_global_teacher", "iou_local_teacher",
        # Teacher morphology
        "teacher_area_px", "teacher_area_frac", "teacher_count",
        # Global morphology
        "global_area_px", "global_area_frac", "global_count",
        # Local morphology
        "local_area_px", "local_area_frac", "local_count",
        # Errors vs teacher
        "global_area_err_px", "global_area_err_frac", "global_count_err",
        "local_area_err_px", "local_area_err_frac", "local_count_err",
    ]

    out_rows = []
    n = len(in_rows)

    for i, row in enumerate(in_rows):
        patch_img_path = patch_img_dir / row["patch_image"]
        patch_mask_path = patch_mask_dir / row["patch_mask"]
        pad_r = int(row["pad_right"])
        pad_b = int(row["pad_bottom"])

        # Load patch image
        bgr = cv2.imread(str(patch_img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  SKIP [{i+1}/{n}]: cannot read {patch_img_path}")
            continue

        # Load teacher mask
        m_teacher_raw = cv2.imread(str(patch_mask_path), cv2.IMREAD_UNCHANGED)
        if m_teacher_raw is None:
            print(f"  SKIP [{i+1}/{n}]: cannot read {patch_mask_path}")
            continue
        if m_teacher_raw.ndim == 3:
            m_teacher_raw = m_teacher_raw[..., 0]
        m_teacher = (m_teacher_raw > 0).astype(bool)

        # Build valid mask
        gray8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        valid = build_valid_mask(gray8, pad_r, pad_b)

        # Prepare work channel
        leaf_mask = binary_threshold_mask(gray8, 127)
        work = fused_channel_u8(bgr, leaf_mask)

        # Parse theta_global and theta_local
        theta_global = parse_theta(row, "theta_global")
        theta_local = parse_theta(row, "theta_local")

        # Finalize masks (deterministic, no search)
        m_global = finalize_inner_mask(work, theta_global).astype(bool)
        m_local = finalize_inner_mask(work, theta_local).astype(bool)

        # Dice/IoU
        d_gt = dice_masked(m_global, m_teacher, valid)
        d_lt = dice_masked(m_local, m_teacher, valid)
        i_gt = iou_masked(m_global, m_teacher, valid)
        i_lt = iou_masked(m_local, m_teacher, valid)

        # Morphology
        t_apx, t_af, t_cnt = morphology_stats(m_teacher, valid)
        g_apx, g_af, g_cnt = morphology_stats(m_global, valid)
        l_apx, l_af, l_cnt = morphology_stats(m_local, valid)

        # Errors (signed: positive = overestimate)
        g_aerr_px = g_apx - t_apx
        g_aerr_frac = g_af - t_af
        g_cerr = g_cnt - t_cnt
        l_aerr_px = l_apx - t_apx
        l_aerr_frac = l_af - t_af
        l_cerr = l_cnt - t_cnt

        out_row = {
            "leaf_id": row["leaf_id"],
            "patch_image": row["patch_image"],
            "patch_mask": row["patch_mask"],
            "x": row["x"], "y": row["y"],
            "pad_right": pad_r, "pad_bottom": pad_b,
            "patch_type": row["patch_type"],
            "content_frac": row["content_frac"],
            "dice_global_teacher": f"{d_gt:.6f}",
            "dice_local_teacher": f"{d_lt:.6f}",
            "iou_global_teacher": f"{i_gt:.6f}",
            "iou_local_teacher": f"{i_lt:.6f}",
            "teacher_area_px": t_apx,
            "teacher_area_frac": f"{t_af:.6f}",
            "teacher_count": t_cnt,
            "global_area_px": g_apx,
            "global_area_frac": f"{g_af:.6f}",
            "global_count": g_cnt,
            "local_area_px": l_apx,
            "local_area_frac": f"{l_af:.6f}",
            "local_count": l_cnt,
            "global_area_err_px": g_aerr_px,
            "global_area_err_frac": f"{g_aerr_frac:.6f}",
            "global_count_err": g_cerr,
            "local_area_err_px": l_aerr_px,
            "local_area_err_frac": f"{l_aerr_frac:.6f}",
            "local_count_err": l_cerr,
        }
        out_rows.append(out_row)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] {row['patch_image']} "
                  f"Dice(g,t)={d_gt:.3f} Dice(l,t)={d_lt:.3f} "
                  f"area_frac t={t_af:.4f} g={g_af:.4f} l={l_af:.4f} "
                  f"count t={t_cnt} g={g_cnt} l={l_cnt}")

    # ── Write output CSV ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {args.out_csv} ({len(out_rows)} rows)")

    # ── Summary statistics ───────────────────────────────────────────────
    def mean_of(rows, key):
        vals = [float(r[key]) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    def mean_abs(rows, key):
        vals = [abs(float(r[key])) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    def median_of(vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    groups = {
        "All": out_rows,
        "Interior": [r for r in out_rows if r["patch_type"] == "interior"],
        "Boundary": [r for r in out_rows if r["patch_type"] == "boundary"],
    }

    metrics = [
        ("N (patches)",                   lambda rs: len(rs)),
        ("Mean Dice(global, teacher)",    lambda rs: mean_of(rs, "dice_global_teacher")),
        ("Mean Dice(local, teacher)",     lambda rs: mean_of(rs, "dice_local_teacher")),
        ("Mean dDice (local-global)",     lambda rs: mean_of(rs, "dice_local_teacher") - mean_of(rs, "dice_global_teacher")),
        ("Mean IoU(global, teacher)",     lambda rs: mean_of(rs, "iou_global_teacher")),
        ("Mean IoU(local, teacher)",      lambda rs: mean_of(rs, "iou_local_teacher")),
        ("Mean teacher area_frac",        lambda rs: mean_of(rs, "teacher_area_frac")),
        ("Mean global area_frac",         lambda rs: mean_of(rs, "global_area_frac")),
        ("Mean local area_frac",          lambda rs: mean_of(rs, "local_area_frac")),
        ("Mean |area_frac err| (global)", lambda rs: mean_abs(rs, "global_area_err_frac")),
        ("Mean |area_frac err| (local)",  lambda rs: mean_abs(rs, "local_area_err_frac")),
        ("Mean teacher count",            lambda rs: mean_of(rs, "teacher_count")),
        ("Mean global count",             lambda rs: mean_of(rs, "global_count")),
        ("Mean local count",              lambda rs: mean_of(rs, "local_count")),
        ("Mean |count_err| (global)",     lambda rs: mean_abs(rs, "global_count_err")),
        ("Mean |count_err| (local)",      lambda rs: mean_abs(rs, "local_count_err")),
    ]

    # Build table
    col_w_metric = max(len(m[0]) for m in metrics) + 2
    col_w_val = 12
    header_line = f"{'Metric':<{col_w_metric}}" + "".join(f"{g:>{col_w_val}}" for g in groups)
    sep = "-" * len(header_line)

    lines = [sep, header_line, sep]
    for mname, mfunc in metrics:
        vals = []
        for gname, gdata in groups.items():
            v = mfunc(gdata)
            if mname.startswith("N "):
                vals.append(f"{int(v)}")
            else:
                vals.append(f"{v:+.4f}" if "dDice" in mname or "err" in mname else f"{v:.4f}")
        line = f"{mname:<{col_w_metric}}" + "".join(f"{v:>{col_w_val}}" for v in vals)
        lines.append(line)
    lines.append(sep)

    summary_text = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    with open(args.out_summary, "w") as f:
        f.write("Patch Transfer Morphology Evaluation Summary\n")
        f.write(f"Input: {args.in_csv}\n")
        f.write(f"N patches evaluated: {len(out_rows)}\n")
        f.write(f"Valid-pixel masking: pad_right/pad_bottom excluded, gray <= {BG_THRESH} excluded\n\n")
        f.write(summary_text)
    print(f"Wrote {args.out_summary}")
    print()
    print(summary_text)

    # ── Plots ────────────────────────────────────────────────────────────
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        g_af_err = [float(r["global_area_err_frac"]) for r in out_rows]
        l_af_err = [float(r["local_area_err_frac"]) for r in out_rows]
        g_cerr = [int(r["global_count_err"]) for r in out_rows]
        l_cerr = [int(r["local_count_err"]) for r in out_rows]
        t_af = [float(r["teacher_area_frac"]) for r in out_rows]
        g_af = [float(r["global_area_frac"]) for r in out_rows]
        l_af = [float(r["local_area_frac"]) for r in out_rows]

        # 1. Histogram of area_frac error
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        bins = np.linspace(-0.3, 0.3, 40)
        axes[0].hist(g_af_err, bins=bins, alpha=0.7, edgecolor="k", linewidth=0.5)
        axes[0].axvline(0, color="r", ls="--", lw=1)
        axes[0].set_xlabel("area_frac error (global - teacher)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Global-on-patch")
        axes[1].hist(l_af_err, bins=bins, alpha=0.7, color="tab:orange", edgecolor="k", linewidth=0.5)
        axes[1].axvline(0, color="r", ls="--", lw=1)
        axes[1].set_xlabel("area_frac error (local - teacher)")
        axes[1].set_title("Local-opt-on-patch")
        fig.suptitle("Area fraction error distribution", fontsize=13)
        fig.tight_layout()
        p = plot_dir / "hist_area_frac_error.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Wrote {p}")

        # 2. Histogram of count error
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        cmin = min(min(g_cerr), min(l_cerr)) - 1
        cmax = max(max(g_cerr), max(l_cerr)) + 1
        cbins = np.arange(cmin, cmax + 1) - 0.5
        axes[0].hist(g_cerr, bins=cbins, alpha=0.7, edgecolor="k", linewidth=0.5)
        axes[0].axvline(0, color="r", ls="--", lw=1)
        axes[0].set_xlabel("count error (global - teacher)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Global-on-patch")
        axes[1].hist(l_cerr, bins=cbins, alpha=0.7, color="tab:orange", edgecolor="k", linewidth=0.5)
        axes[1].axvline(0, color="r", ls="--", lw=1)
        axes[1].set_xlabel("count error (local - teacher)")
        axes[1].set_title("Local-opt-on-patch")
        fig.suptitle("Lesion count error distribution", fontsize=13)
        fig.tight_layout()
        p = plot_dir / "hist_count_error.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Wrote {p}")

        # 3. Scatter: teacher area_frac vs global/local area_frac
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        lim = max(max(t_af), max(g_af), max(l_af)) * 1.05 + 0.01
        for ax, pred_af, title, color in [
            (axes[0], g_af, "Global-on-patch", "tab:blue"),
            (axes[1], l_af, "Local-opt-on-patch", "tab:orange"),
        ]:
            ax.scatter(t_af, pred_af, s=14, alpha=0.6, edgecolors="k",
                       linewidths=0.3, color=color)
            ax.plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
            ax.set_xlabel("Teacher area_frac")
            ax.set_ylabel("Predicted area_frac")
            ax.set_title(title)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.legend(fontsize=9)
        fig.suptitle("Teacher vs. predicted area fraction", fontsize=13)
        fig.tight_layout()
        p = plot_dir / "scatter_teacher_vs_pred_area_frac.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
