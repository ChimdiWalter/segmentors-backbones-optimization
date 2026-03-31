#!/usr/bin/env python3
"""Generate param_morphology_metrics.csv and two scatter plots.

Reads opt_summary_local.csv and the corresponding mask files to compute:
  - area_frac = area_px / (image_width * image_height)
  - lesion_count = n_contours (from CSV)
  - contrast = mean(foreground) - mean(background) in grayscale leaf image

Outputs:
  Paper_3/tables/param_morphology_metrics.csv
  Paper_3/figs/exp_v1/scatter_energy_threshold_vs_lesion_count.png
  Paper_3/figs/exp_v1/scatter_beta_vs_area_frac.png
"""

import csv, os, sys, statistics
from pathlib import Path

# Optional: PIL for image-based contrast; matplotlib for plots
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent.parent  # segmentors_backbones/
CSV_PATH = BASE / "experiments" / "exp_v1" / "outputs" / "opt_summary_local.csv"
LEAVES_DIR = BASE / "leaves"
FIG_DIR = BASE / "Paper_3" / "figs" / "exp_v1"
TABLE_DIR = BASE / "Paper_3" / "tables"


def compute_contrast(leaf_path, mask_path):
    """Compute mean grayscale contrast between foreground and background."""
    try:
        leaf = np.array(Image.open(leaf_path).convert("L"), dtype=np.float64)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float64)
        # Binarize mask
        mask_bin = mask > 127
        fg_count = mask_bin.sum()
        bg_count = (~mask_bin).sum()
        if fg_count == 0 or bg_count == 0:
            return 0.0
        fg_mean = leaf[mask_bin].mean()
        bg_mean = leaf[~mask_bin].mean()
        return abs(fg_mean - bg_mean)
    except Exception as e:
        print(f"  WARNING: contrast computation failed for {leaf_path}: {e}", file=sys.stderr)
        return 0.0


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows_in = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status", "").strip() == "ok":
                rows_in.append(r)

    print(f"Read {len(rows_in)} ok rows from {CSV_PATH}")

    out_rows = []
    for i, r in enumerate(rows_in):
        fname = r["filename"]
        leaf_path = LEAVES_DIR / fname
        mask_path = Path(r["mask16_path"])

        # Get image dimensions for area_frac
        try:
            img = Image.open(leaf_path)
            total_px = img.width * img.height
        except Exception:
            total_px = 1  # fallback

        area_px = float(r["area_px"])
        area_frac = area_px / total_px if total_px > 0 else 0.0
        lesion_count = int(float(r["n_contours"]))
        contrast = compute_contrast(leaf_path, mask_path)

        out_rows.append({
            "filename": fname,
            "energy_threshold": float(r["energy_threshold"]),
            "beta": float(r["beta"]),
            "mu": float(r["mu"]),
            "diffusion_rate": float(r["diffusion_rate"]),
            "alpha": float(r["alpha"]),
            "gamma": float(r["gamma"]),
            "score": float(r["score"]),
            "area_px": int(area_px),
            "area_frac": area_frac,
            "lesion_count": lesion_count,
            "contrast": contrast,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(rows_in)}")

    # Write CSV
    csv_out = TABLE_DIR / "param_morphology_metrics.csv"
    fieldnames = ["filename", "energy_threshold", "beta", "mu", "diffusion_rate",
                  "alpha", "gamma", "score", "area_px", "area_frac",
                  "lesion_count", "contrast"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {csv_out} ({len(out_rows)} rows)")

    # --- Scatter plot 1: energy_threshold vs lesion_count ---
    et = [r["energy_threshold"] for r in out_rows]
    lc = [r["lesion_count"] for r in out_rows]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(et, lc, s=18, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Energy threshold $E_{\\mathrm{thr}}$", fontsize=11)
    ax.set_ylabel("Lesion count $n_c$", fontsize=11)
    ax.set_title("Energy threshold vs. lesion count", fontsize=12)
    fig.tight_layout()
    p1 = FIG_DIR / "scatter_energy_threshold_vs_lesion_count.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"Wrote {p1}")

    # --- Scatter plot 2: beta vs area_frac ---
    beta = [r["beta"] for r in out_rows]
    af = [r["area_frac"] for r in out_rows]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(beta, af, s=18, alpha=0.6, edgecolors="k", linewidths=0.3, color="tab:orange")
    ax.set_xlabel("Balloon force $\\beta$", fontsize=11)
    ax.set_ylabel("Area fraction $A/A_{\\mathrm{img}}$", fontsize=11)
    ax.set_title("Balloon force vs. area fraction", fontsize=12)
    fig.tight_layout()
    p2 = FIG_DIR / "scatter_beta_vs_area_frac.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
