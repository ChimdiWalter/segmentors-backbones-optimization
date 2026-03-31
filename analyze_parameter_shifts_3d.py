#!/usr/bin/env python3
"""3D parameter-space analysis of theta_global -> theta_local shifts.

Reads existing CSVs (no optimization), computes parameter deltas,
generates PCA 3D plots, raw 3D plots, delta plots, pairplot,
correlation heatmap, analysis CSV, and summary.

Usage:
    python3 analyze_parameter_shifts_3d.py
"""

import csv, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
V2_CSV = ROOT / "experiments/exp_v1/outputs/patch_local_opt_v2.csv"
MORPH_CSV = ROOT / "experiments/exp_v1/outputs/patch_transfer_morphology_eval.csv"
OUT_DIR = ROOT / "experiments/exp_v1/outputs/parameter_shift_3d_analysis"

PARAM_NAMES = ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "energy_threshold"]


def load_and_merge():
    """Load v2 CSV (theta) and morphology CSV (count/area errors), merge by patch_image."""
    v2 = {}
    with open(V2_CSV, newline="") as f:
        for r in csv.DictReader(f):
            v2[r["patch_image"]] = r

    morph = {}
    with open(MORPH_CSV, newline="") as f:
        for r in csv.DictReader(f):
            morph[r["patch_image"]] = r

    rows = []
    for key, r in v2.items():
        merged = dict(r)
        if key in morph:
            m = morph[key]
            for col in ["global_count_err", "local_count_err",
                        "global_area_err_frac", "local_area_err_frac",
                        "teacher_count", "global_count", "local_count",
                        "teacher_area_frac", "global_area_frac", "local_area_frac"]:
                merged[col] = m.get(col, "0")
        else:
            for col in ["global_count_err", "local_count_err",
                        "global_area_err_frac", "local_area_err_frac"]:
                merged[col] = "0"
        rows.append(merged)
    return rows


def add_derived(rows):
    """Add delta parameters and performance deltas."""
    for r in rows:
        for p in PARAM_NAMES:
            g = float(r[f"theta_global_{p}"])
            l = float(r[f"theta_local_{p}"])
            r[f"d_{p}"] = l - g

        dgt = float(r["dice_global_teacher"])
        dlt = float(r["dice_local_teacher"])
        igt = float(r["iou_global_teacher"])
        ilt = float(r["iou_local_teacher"])
        gce = float(r.get("global_count_err", 0))
        lce = float(r.get("local_count_err", 0))
        gafe = float(r.get("global_area_err_frac", 0))
        lafe = float(r.get("local_area_err_frac", 0))

        r["dD"] = dlt - dgt
        r["dIoU"] = ilt - igt
        r["delta_abs_count_err"] = abs(lce) - abs(gce)
        r["delta_abs_area_frac_err"] = abs(lafe) - abs(gafe)
    return rows


def write_analysis_csv(rows, out_path):
    fields = [
        "leaf_id", "patch_image", "patch_type", "content_frac",
    ]
    for p in PARAM_NAMES:
        fields.append(f"theta_global_{p}")
    for p in PARAM_NAMES:
        fields.append(f"theta_local_{p}")
    for p in PARAM_NAMES:
        fields.append(f"d_{p}")
    fields += ["dD", "dIoU", "delta_abs_count_err", "delta_abs_area_frac_err"]

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save(fig, name):
    png = OUT_DIR / f"{name}.png"
    fig.savefig(str(png), dpi=220, bbox_inches="tight", facecolor="white")
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(str(pdf), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


def make_pca_data(rows):
    """Standardize 7 global + 7 local params together, fit PCA on all 300 points."""
    global_params = np.array([[float(r[f"theta_global_{p}"]) for p in PARAM_NAMES] for r in rows])
    local_params  = np.array([[float(r[f"theta_local_{p}"]) for p in PARAM_NAMES] for r in rows])

    all_params = np.vstack([global_params, local_params])  # (300, 7)
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_params)

    pca = PCA(n_components=3)
    all_pca = pca.fit_transform(all_scaled)

    n = len(rows)
    global_pca = all_pca[:n]
    local_pca  = all_pca[n:]
    return global_pca, local_pca, pca, scaler


def plot_pca_3d(global_pca, local_pca, color_vals, cmap, clabel, title, name,
                vmin=None, vmax=None, vcenter=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Global points
    ax.scatter(global_pca[:, 0], global_pca[:, 1], global_pca[:, 2],
               c="steelblue", marker="o", s=25, alpha=0.5, label="Global θ*", edgecolors="none")

    # Arrows
    for i in range(len(global_pca)):
        ax.plot([global_pca[i, 0], local_pca[i, 0]],
                [global_pca[i, 1], local_pca[i, 1]],
                [global_pca[i, 2], local_pca[i, 2]],
                color="gray", alpha=0.15, lw=0.5)

    # Local points colored by metric
    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = None
    sc = ax.scatter(local_pca[:, 0], local_pca[:, 1], local_pca[:, 2],
                    c=color_vals, cmap=cmap, marker="^", s=35, alpha=0.8,
                    edgecolors="k", linewidths=0.3, norm=norm,
                    label="Local θ*_local")

    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label(clabel, fontsize=10)

    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_zlabel("PC3", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, name)


def plot_raw_3d(rows, xp, yp, zp, color_vals, cmap, clabel, title, name,
                vcenter=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    gx = np.array([float(r[f"theta_global_{xp}"]) for r in rows])
    gy = np.array([float(r[f"theta_global_{yp}"]) for r in rows])
    gz = np.array([float(r[f"theta_global_{zp}"]) for r in rows])
    lx = np.array([float(r[f"theta_local_{xp}"]) for r in rows])
    ly = np.array([float(r[f"theta_local_{yp}"]) for r in rows])
    lz = np.array([float(r[f"theta_local_{zp}"]) for r in rows])

    ax.scatter(gx, gy, gz, c="steelblue", marker="o", s=25, alpha=0.5,
               label="Global", edgecolors="none")

    for i in range(len(rows)):
        ax.plot([gx[i], lx[i]], [gy[i], ly[i]], [gz[i], lz[i]],
                color="gray", alpha=0.15, lw=0.5)

    cv = np.array(color_vals)
    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=cv.min(), vcenter=vcenter, vmax=cv.max())
    else:
        norm = None
    sc = ax.scatter(lx, ly, lz, c=cv, cmap=cmap, marker="^", s=35,
                    alpha=0.8, edgecolors="k", linewidths=0.3, norm=norm,
                    label="Local")

    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label(clabel, fontsize=10)

    ax.set_xlabel(xp, fontsize=10)
    ax.set_ylabel(yp, fontsize=10)
    ax.set_zlabel(zp, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, name)


def plot_delta_3d(rows, xp, yp, zp, color_vals, cmap, clabel, title, name,
                  vcenter=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    dx = np.array([r[f"d_{xp}"] for r in rows])
    dy = np.array([r[f"d_{yp}"] for r in rows])
    dz = np.array([r[f"d_{zp}"] for r in rows])
    cv = np.array(color_vals)

    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=cv.min(), vcenter=vcenter, vmax=cv.max())
    else:
        norm = None
    sc = ax.scatter(dx, dy, dz, c=cv, cmap=cmap, s=40, alpha=0.8,
                    edgecolors="k", linewidths=0.3, norm=norm)

    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label(clabel, fontsize=10)

    ax.set_xlabel(f"Δ{xp}", fontsize=10)
    ax.set_ylabel(f"Δ{yp}", fontsize=10)
    ax.set_zlabel(f"Δ{zp}", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def plot_delta_3d_patchtype(rows, xp, yp, zp, title, name):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    int_rows = [r for r in rows if r["patch_type"] == "interior"]
    bnd_rows = [r for r in rows if r["patch_type"] == "boundary"]

    for subset, color, label, marker in [(int_rows, "steelblue", "Interior", "o"),
                                          (bnd_rows, "indianred", "Boundary", "^")]:
        dx = np.array([r[f"d_{xp}"] for r in subset])
        dy = np.array([r[f"d_{yp}"] for r in subset])
        dz = np.array([r[f"d_{zp}"] for r in subset])
        ax.scatter(dx, dy, dz, c=color, marker=marker, s=40, alpha=0.7,
                   edgecolors="k", linewidths=0.3, label=label)

    ax.set_xlabel(f"Δ{xp}", fontsize=10)
    ax.set_ylabel(f"Δ{yp}", fontsize=10)
    ax.set_zlabel(f"Δ{zp}", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, name)


# ── Part C: 2D plots ─────────────────────────────────────────────────────────

def plot_pairplot(rows, name):
    """Scatter matrix of 7 delta parameters colored by patch_type."""
    delta_names = [f"d_{p}" for p in PARAM_NAMES]
    short_names = ["Δμ", "Δλ", "Δdiff", "Δα", "Δβ", "Δγ", "ΔE_thr"]
    n = len(delta_names)

    fig, axes = plt.subplots(n, n, figsize=(16, 14))

    colors = {"interior": "steelblue", "boundary": "indianred"}

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Histogram
                for ptype, color in colors.items():
                    vals = [r[delta_names[i]] for r in rows if r["patch_type"] == ptype]
                    ax.hist(vals, bins=20, alpha=0.5, color=color, label=ptype)
                if i == 0:
                    ax.legend(fontsize=5)
            else:
                for ptype, color in colors.items():
                    xv = [r[delta_names[j]] for r in rows if r["patch_type"] == ptype]
                    yv = [r[delta_names[i]] for r in rows if r["patch_type"] == ptype]
                    ax.scatter(xv, yv, c=color, s=8, alpha=0.4, edgecolors="none")

            if j == 0:
                ax.set_ylabel(short_names[i], fontsize=7)
            if i == n - 1:
                ax.set_xlabel(short_names[j], fontsize=7)
            ax.tick_params(labelsize=5)

    fig.suptitle("Delta Parameter Pairplot (by patch type)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, name)


def plot_correlation_heatmap(rows, name):
    """Heatmap: correlation between delta params and performance metrics."""
    delta_cols = [f"d_{p}" for p in PARAM_NAMES]
    perf_cols = ["dD", "delta_abs_count_err", "delta_abs_area_frac_err"]
    all_cols = delta_cols + perf_cols

    n_rows = len(rows)
    data = np.zeros((n_rows, len(all_cols)))
    for i, r in enumerate(rows):
        for j, c in enumerate(all_cols):
            data[i, j] = float(r[c])

    corr = np.corrcoef(data.T)

    # Extract the cross-correlation block: delta params (rows) x perf (cols)
    n_delta = len(delta_cols)
    n_perf = len(perf_cols)
    cross_corr = corr[:n_delta, n_delta:]  # (7, 3)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cross_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    delta_labels = ["Δμ", "Δλ", "Δdiff_rate", "Δα", "Δβ", "Δγ", "ΔE_thr"]
    perf_labels = ["ΔDice", "Δ|count_err|", "Δ|area_frac_err|"]

    ax.set_xticks(range(n_perf))
    ax.set_xticklabels(perf_labels, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(n_delta))
    ax.set_yticklabels(delta_labels, fontsize=10)

    for i in range(n_delta):
        for j in range(n_perf):
            val = cross_corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Pearson r", fontsize=10)
    ax.set_title("Parameter Δ vs Performance Δ Correlation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(rows, pca, out_path):
    lines = [
        "Parameter Shift 3D Analysis Summary",
        "=" * 55,
        f"N patches: {len(rows)}",
        f"  Interior: {sum(1 for r in rows if r['patch_type']=='interior')}",
        f"  Boundary: {sum(1 for r in rows if r['patch_type']=='boundary')}",
        "",
        "--- PCA Explained Variance ---",
        f"  PC1: {pca.explained_variance_ratio_[0]:.4f}  ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        f"  PC2: {pca.explained_variance_ratio_[1]:.4f}  ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        f"  PC3: {pca.explained_variance_ratio_[2]:.4f}  ({pca.explained_variance_ratio_[2]*100:.1f}%)",
        f"  Cumulative: {sum(pca.explained_variance_ratio_[:3])*100:.1f}%",
        "",
        "--- PCA Component Loadings (top contributors per PC) ---",
    ]
    for pc_idx in range(3):
        loadings = pca.components_[pc_idx]
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        top3 = [(PARAM_NAMES[i], loadings[i]) for i in sorted_idx[:3]]
        lines.append(f"  PC{pc_idx+1}: " + ", ".join(f"{n}={v:+.3f}" for n, v in top3))

    lines += ["", "--- Mean / Median Delta Parameters ---"]
    for p in PARAM_NAMES:
        vals = [r[f"d_{p}"] for r in rows]
        lines.append(f"  d_{p:20s}: mean={np.mean(vals):+.5f}  median={np.median(vals):+.5f}  std={np.std(vals):.5f}")

    lines += ["", "--- Performance Deltas ---"]
    int_rows = [r for r in rows if r["patch_type"] == "interior"]
    bnd_rows = [r for r in rows if r["patch_type"] == "boundary"]

    for label, subset in [("ALL", rows), ("INTERIOR", int_rows), ("BOUNDARY", bnd_rows)]:
        dDs = [r["dD"] for r in subset]
        dCE = [r["delta_abs_count_err"] for r in subset]
        dAE = [r["delta_abs_area_frac_err"] for r in subset]
        lines.append(f"  {label} (n={len(subset)}):")
        lines.append(f"    mean dD:               {np.mean(dDs):+.4f}")
        lines.append(f"    mean Δ|count_err|:     {np.mean(dCE):+.2f} (neg=improvement)")
        lines.append(f"    mean Δ|area_frac_err|: {np.mean(dAE):+.5f}")

    # Correlation analysis
    lines += ["", "--- Strongest Correlations with Performance ---"]
    delta_vals = {p: np.array([r[f"d_{p}"] for r in rows]) for p in PARAM_NAMES}
    perf_map = {
        "dD": np.array([r["dD"] for r in rows]),
        "Δ|count_err|": np.array([r["delta_abs_count_err"] for r in rows]),
        "Δ|area_frac_err|": np.array([r["delta_abs_area_frac_err"] for r in rows]),
    }

    for perf_name, perf_vals in perf_map.items():
        corrs = []
        for p in PARAM_NAMES:
            r_val = np.corrcoef(delta_vals[p], perf_vals)[0, 1]
            corrs.append((p, r_val))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        top = corrs[:3]
        lines.append(f"  {perf_name}: " + ", ".join(f"d_{n} (r={v:+.3f})" for n, v in top))

    # Interpretation
    lines += [
        "",
        "--- Interpretation ---",
    ]

    # Check if shifts are structured (compare std of deltas to mean)
    all_d_std = [np.std([r[f"d_{p}"] for r in rows]) for p in PARAM_NAMES]
    all_d_mean = [abs(np.mean([r[f"d_{p}"] for r in rows])) for p in PARAM_NAMES]

    # Compare interior vs boundary shift magnitudes
    int_shift_mag = np.mean([np.sqrt(sum(r[f"d_{p}"]**2 for p in PARAM_NAMES)) for r in int_rows])
    bnd_shift_mag = np.mean([np.sqrt(sum(r[f"d_{p}"]**2 for p in PARAM_NAMES)) for r in bnd_rows])

    lines.append(f"  Mean shift magnitude (L2 in param space):")
    lines.append(f"    Interior: {int_shift_mag:.4f}")
    lines.append(f"    Boundary: {bnd_shift_mag:.4f}")

    if bnd_shift_mag > int_shift_mag * 1.1:
        lines.append("  -> Boundary patches have LARGER parameter shifts (noisier context).")
    elif int_shift_mag > bnd_shift_mag * 1.1:
        lines.append("  -> Interior patches have LARGER parameter shifts.")
    else:
        lines.append("  -> Shift magnitudes are similar for interior and boundary.")

    # Which params change most
    param_change_rank = sorted(zip(PARAM_NAMES, all_d_std), key=lambda x: x[1], reverse=True)
    lines.append(f"  Most variable parameters (by std of delta):")
    for p, s in param_change_rank:
        lines.append(f"    d_{p}: std={s:.5f}")

    # Check structured vs random
    # Compute fraction of variance explained by mean shift
    for p in PARAM_NAMES:
        vals = np.array([r[f"d_{p}"] for r in rows])
        if np.std(vals) > 1e-10:
            snr = abs(np.mean(vals)) / np.std(vals)
        else:
            snr = 0.0
        if snr > 0.3:
            lines.append(f"  d_{p}: systematic shift (|mean|/std = {snr:.2f})")

    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    for l in lines:
        print(l)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading and merging CSVs...")
    rows = load_and_merge()
    rows = add_derived(rows)
    print(f"  {len(rows)} patches loaded")

    # Part A: analysis CSV
    print("\n=== Part A: Analysis CSV ===")
    write_analysis_csv(rows, OUT_DIR / "parameter_shift_analysis.csv")

    # Precompute arrays
    dD_vals = np.array([r["dD"] for r in rows])
    dCE_vals = np.array([r["delta_abs_count_err"] for r in rows])
    dAE_vals = np.array([r["delta_abs_area_frac_err"] for r in rows])

    # Part B: 3D plots
    print("\n=== Part B: 3D Plots ===")

    # PCA
    global_pca, local_pca, pca, scaler = make_pca_data(rows)

    # 1) PCA colored by dD
    plot_pca_3d(global_pca, local_pca, dD_vals, "RdYlBu_r",
                "ΔDice (local - global)",
                "PCA 3D: Global→Local Parameter Shift (colored by ΔDice)",
                "pca3d_global_to_local_dD",
                vmin=dD_vals.min(), vcenter=0, vmax=dD_vals.max())

    # 2) PCA colored by count improvement
    plot_pca_3d(global_pca, local_pca, dCE_vals, "RdYlGn_r",
                "Δ|count_err| (neg=improved)",
                "PCA 3D: Global→Local (colored by count improvement)",
                "pca3d_global_to_local_count_improvement",
                vmin=dCE_vals.min(), vcenter=0, vmax=dCE_vals.max())

    # 3) PCA colored by area_frac improvement
    plot_pca_3d(global_pca, local_pca, dAE_vals, "RdYlGn_r",
                "Δ|area_frac_err| (neg=improved)",
                "PCA 3D: Global→Local (colored by area improvement)",
                "pca3d_global_to_local_area_improvement",
                vmin=dAE_vals.min(), vcenter=0, vmax=dAE_vals.max())

    # 4) Raw 3D: threshold, beta, mu colored by dD
    plot_raw_3d(rows, "energy_threshold", "beta", "mu", dD_vals,
                "RdYlBu_r", "ΔDice",
                "Raw 3D: E_thr × β × μ (colored by ΔDice)",
                "raw3d_threshold_beta_mu_dD", vcenter=0)

    # 5) Raw 3D: threshold, gamma, diffusion_rate colored by dD
    plot_raw_3d(rows, "energy_threshold", "gamma", "diffusion_rate", dD_vals,
                "RdYlBu_r", "ΔDice",
                "Raw 3D: E_thr × γ × diff_rate (colored by ΔDice)",
                "raw3d_threshold_gamma_diffusion_dD", vcenter=0)

    # 6) Delta 3D: d_threshold, d_beta, d_mu colored by dD
    plot_delta_3d(rows, "energy_threshold", "beta", "mu", dD_vals,
                  "RdYlBu_r", "ΔDice",
                  "Delta 3D: ΔE_thr × Δβ × Δμ (colored by ΔDice)",
                  "delta3d_threshold_beta_mu_dD", vcenter=0)

    # 7) Delta 3D colored by patch_type
    plot_delta_3d_patchtype(rows, "energy_threshold", "beta", "mu",
                            "Delta 3D: ΔE_thr × Δβ × Δμ (by patch type)",
                            "delta3d_threshold_beta_mu_patchtype")

    # Part C: 2D plots
    print("\n=== Part C: 2D Plots ===")
    plot_pairplot(rows, "delta_parameter_pairplot_patchtype")
    plot_correlation_heatmap(rows, "parameter_delta_correlation_heatmap")

    # Part D: Summary
    print("\n=== Part D: Summary ===")
    write_summary(rows, pca, OUT_DIR / "summary.txt")

    # Final listing
    print(f"\n=== Output listing ===")
    for fn in sorted(os.listdir(OUT_DIR)):
        fpath = OUT_DIR / fn
        sz = os.path.getsize(fpath)
        if sz > 1024 * 1024:
            szs = f"{sz / 1024 / 1024:.1f}M"
        elif sz > 1024:
            szs = f"{sz / 1024:.0f}K"
        else:
            szs = f"{sz}B"
        print(f"  {fn}  ({szs})")

    print("\nDone.")


if __name__ == "__main__":
    main()
