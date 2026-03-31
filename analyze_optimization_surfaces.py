#!/usr/bin/env python3
"""
True objective-surface / minimization analysis.

No candidate-evaluation logs exist in this repo; the optimizer only saves
final best parameters. This script therefore builds LOCAL objective slices
by rescoring dense parameter grids around existing optima using the
deterministic score function from navier_optimize_robust_mostrecent.py.

All outputs are clearly labeled as "local objective slices" — they are NOT
global full optimization landscapes.

Output: experiments/exp_v1/outputs/optimization_surface_analysis/
"""

import os, sys, time, math, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

# ── Import scoring functions from the optimizer ──
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from navier_optimize_robust_mostrecent import (
    evaluate_params_refined, fused_channel_u8, binary_threshold_mask,
    compute_image_gradients, build_seed_from_image_gradients, BOUNDS,
    read_color_anydepth,
)

# ── Paths ──
OPT_CSV   = os.path.join(REPO, "experiments/exp_v1/outputs/opt_summary_local.csv")
V2_CSV    = os.path.join(REPO, "experiments/exp_v1/outputs/patch_local_opt_v2.csv")
LEAVES_DIR = os.path.join(REPO, "leaves")
PATCHES_DIR = os.path.join(REPO, "leaves_patches")
OUT_DIR   = os.path.join(REPO, "experiments/exp_v1/outputs/optimization_surface_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 220
GRID_N = 20       # 20x20 grid for 2D slices (balanced speed vs resolution)
PROFILE_N = 30    # points for 1D profiles
DOWNSCALE = 0.25  # downscale factor for leaves (patches use full res)
PARAM_NAMES = ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "energy_threshold"]

# 2D slices to produce
SLICE_PAIRS = [
    ("energy_threshold", "beta"),
    ("energy_threshold", "mu"),
    ("beta", "gamma"),
]


# ── Part 1: Log Discovery ──
def write_log_discovery():
    lines = [
        "Candidate Evaluation Log Discovery Report",
        "=" * 50,
        "",
        "Search performed across:",
        "  - experiments/exp_v1/outputs/ (CSV, JSON, log, npy, pkl)",
        "  - experiments/exp_v1/logs/ (log files)",
        "",
        "Result: NO per-candidate evaluation logs were found.",
        "",
        "The optimizer (navier_optimize_robust_mostrecent.py) uses a",
        "random-sample + local-refinement strategy but only saves the",
        "final best parameters and score per image. Intermediate candidates",
        "and their scores are not logged to disk.",
        "",
        "Files inspected:",
        "  - opt_summary_local.csv: 174 leaves, final best score + 7 params (no candidates)",
        "  - patch_local_opt_v2.csv: 150 patches, final theta_global + theta_local (no candidates)",
        "  - experiments/exp_v1/logs/ablations/*.log: per-image final scores only",
        "",
        "Conclusion: A true global optimization surface CANNOT be reconstructed.",
        "Instead, this script creates LOCAL OBJECTIVE SLICES by rescoring",
        "dense parameter grids around saved optima using the deterministic",
        "score function (evaluate_params_refined).",
        "",
        "These slices show local curvature and sensitivity around the optimum,",
        "not the full landscape the optimizer explored.",
    ]
    path = os.path.join(OUT_DIR, "log_discovery.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


# ── Image preparation helpers ──
def prepare_leaf_data(filename, downscale=DOWNSCALE):
    """Load leaf image and prepare all inputs for evaluate_params_refined."""
    path = os.path.join(LEAVES_DIR, filename)
    raw, bgr8_full, gray8_full = read_color_anydepth(path)
    if raw is None:
        raise FileNotFoundError(f"Cannot load {path}")
    leaf_mask = binary_threshold_mask(gray8_full, threshold=127)
    work_full = fused_channel_u8(bgr8_full, leaf_mask)
    if 0 < downscale < 1:
        work = cv2.resize(work_full, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        bgr_work = cv2.resize(bgr8_full, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        work = work_full
        bgr_work = bgr8_full
    img_gx, img_gy, img_mag = compute_image_gradients(work)
    seed_gx, seed_gy, seed_mag = build_seed_from_image_gradients(img_gx, img_gy, sigma=1.5)
    grad_mag_u8 = np.uint8(255.0 * img_mag / max(float(img_mag.max()), 1e-8))
    lab_work = cv2.cvtColor(bgr_work, cv2.COLOR_BGR2Lab)
    return work, img_mag, img_gx, img_gy, seed_mag, seed_gx, seed_gy, grad_mag_u8, lab_work


def prepare_patch_data(patch_name):
    """Load patch image (PNG, already 448x448) and prepare scoring inputs."""
    path = os.path.join(PATCHES_DIR, patch_name)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    leaf_mask = binary_threshold_mask(gray, threshold=5)
    work = fused_channel_u8(bgr, leaf_mask)
    img_gx, img_gy, img_mag = compute_image_gradients(work)
    seed_gx, seed_gy, seed_mag = build_seed_from_image_gradients(img_gx, img_gy, sigma=1.5)
    grad_mag_u8 = np.uint8(255.0 * img_mag / max(float(img_mag.max()), 1e-8))
    lab_work = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    return work, img_mag, img_gx, img_gy, seed_mag, seed_gx, seed_gy, grad_mag_u8, lab_work


def score_at(data_tuple, params_dict, its_snake=6, max_blob_frac=1.0):
    """Evaluate score for given params on prepared data."""
    work, img_mag, img_gx, img_gy, seed_mag, seed_gx, seed_gy, grad_mag_u8, lab_work = data_tuple
    p = dict(params_dict)
    p["iters"] = 8
    res = evaluate_params_refined(
        work, img_mag, img_gx, img_gy, seed_mag, seed_gx, seed_gy,
        grad_mag_u8, lab_work, its_snake, p,
        min_blob_frac=1e-7, max_blob_frac=max_blob_frac,
    )
    return res.score


# ── 2D Slice Evaluation ──
def evaluate_2d_slice(data_tuple, base_params, param_x, param_y, n=GRID_N, max_blob_frac=1.0):
    """Evaluate score on a 2D grid varying param_x and param_y."""
    lo_x, hi_x = BOUNDS[param_x]
    lo_y, hi_y = BOUNDS[param_y]
    xs = np.linspace(lo_x, hi_x, n)
    ys = np.linspace(lo_y, hi_y, n)
    if param_x == "energy_threshold":
        xs = np.round(xs).astype(int)
    if param_y == "energy_threshold":
        ys = np.round(ys).astype(int)

    scores = np.full((n, n), np.nan)
    total = n * n
    t0 = time.time()
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            p = dict(base_params)
            p[param_x] = float(xv) if param_x != "energy_threshold" else int(xv)
            p[param_y] = float(yv) if param_y != "energy_threshold" else int(yv)
            scores[j, i] = score_at(data_tuple, p, max_blob_frac=max_blob_frac)
        done = (i + 1) * n
        elapsed = time.time() - t0
        if i % 5 == 4:
            eta = elapsed / done * (total - done)
            print(f"    grid {done}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    return xs, ys, scores


# ── 1D Profile Evaluation ──
def evaluate_1d_profile(data_tuple, base_params, param_name, n=PROFILE_N, max_blob_frac=1.0):
    """Score along one parameter axis."""
    lo, hi = BOUNDS[param_name]
    vals = np.linspace(lo, hi, n)
    if param_name == "energy_threshold":
        vals = np.round(vals).astype(int)
    scores = np.full(n, np.nan)
    for i, v in enumerate(vals):
        p = dict(base_params)
        p[param_name] = float(v) if param_name != "energy_threshold" else int(v)
        scores[i] = score_at(data_tuple, p, max_blob_frac=max_blob_frac)
    return vals, scores


# ── Plotting ──
def plot_2d_surface(xs, ys, scores, param_x, param_y, case_label, opt_x, opt_y, opt_score):
    """3D surface plot + top-down heatmap."""
    X, Y = np.meshgrid(xs, ys)

    # Replace -1e9 (infeasible) with NaN for cleaner plots
    S = scores.copy()
    S[S < -1e6] = np.nan

    # 3D surface
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, S, cmap="viridis", alpha=0.8, linewidth=0.3,
                           edgecolor="gray", antialiased=True)
    ax.scatter([opt_x], [opt_y], [opt_score], c="red", s=100, marker="*",
               zorder=10, label=f"Optimum ({opt_score:.2f})")
    ax.set_xlabel(param_x, fontsize=10)
    ax.set_ylabel(param_y, fontsize=10)
    ax.set_zlabel("Score", fontsize=10)
    ax.set_title(f"Local Score Surface: {case_label}\n{param_x} vs {param_y}", fontsize=11)
    ax.legend(fontsize=9)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Score")
    name = f"local_score_surface_{case_label}_{param_x}_{param_y}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    valid = S[~np.isnan(S)]
    if len(valid) > 0:
        vmin, vmax = np.percentile(valid, [2, 98])
    else:
        vmin, vmax = 0, 1
    im = ax.imshow(S, origin="lower", aspect="auto",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   cmap="viridis", vmin=vmin, vmax=vmax)
    ax.plot(opt_x, opt_y, "r*", markersize=18, label=f"Optimum ({opt_score:.2f})")
    # contour lines
    try:
        S_filled = np.where(np.isnan(S), np.nanmin(S) if np.any(~np.isnan(S)) else 0, S)
        ax.contour(X, Y, S_filled, levels=12, colors="white", alpha=0.5, linewidths=0.6)
    except Exception:
        pass
    ax.set_xlabel(param_x, fontsize=11)
    ax.set_ylabel(param_y, fontsize=11)
    ax.set_title(f"Score Heatmap: {case_label}\n{param_x} vs {param_y}", fontsize=11)
    ax.legend(fontsize=9)
    plt.colorbar(im, ax=ax, label="Score")
    hname = f"local_score_heatmap_{case_label}_{param_x}_{param_y}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{hname}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {name} + {hname}")

    # Plotly interactive
    try:
        import plotly.graph_objects as go
        fig_pl = go.Figure(data=[
            go.Surface(x=xs, y=ys, z=S, colorscale="Viridis", opacity=0.85,
                       name="Score surface"),
            go.Scatter3d(x=[opt_x], y=[opt_y], z=[opt_score],
                         mode="markers", marker=dict(size=8, color="red"),
                         name="Optimum"),
        ])
        fig_pl.update_layout(
            scene=dict(xaxis_title=param_x, yaxis_title=param_y, zaxis_title="Score"),
            title=f"Score Surface: {case_label} ({param_x} vs {param_y})",
            width=1000, height=800,
        )
        fig_pl.write_html(os.path.join(OUT_DIR, f"{name}.html"))
        print(f"    Saved {name}.html")
    except Exception:
        pass


def plot_1d_profile(vals, scores, param_name, case_label, opt_val, opt_score):
    """1D score profile along one parameter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = scores > -1e6
    ax.plot(vals[valid], scores[valid], "b-", lw=2)
    ax.axvline(opt_val, color="red", ls="--", lw=1.5, label=f"Optimum = {opt_val}")
    ax.scatter([opt_val], [opt_score], c="red", s=100, marker="*", zorder=10)
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Score Profile: {case_label}\n{param_name} (others fixed)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    name = f"score_profile_{param_name}_{case_label}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {name}")


# ── Case runners ──
def run_leaf_case(filename, params_dict, case_label):
    """Run 2D slices and 1D profiles for a full-leaf case."""
    print(f"\n  === Leaf case: {case_label} ({filename}) ===")
    data = prepare_leaf_data(filename, downscale=DOWNSCALE)
    opt_score_at_opt = score_at(data, params_dict, max_blob_frac=0.85)
    print(f"    Score at saved optimum (downscaled): {opt_score_at_opt:.3f}")

    for px, py in SLICE_PAIRS:
        print(f"    Slice: {px} vs {py}")
        xs, ys, scores = evaluate_2d_slice(data, params_dict, px, py, max_blob_frac=0.85)
        plot_2d_surface(xs, ys, scores, px, py, case_label,
                       params_dict[px], params_dict[py], opt_score_at_opt)

    # 1D profiles for key parameters
    for pname in ["energy_threshold", "mu", "beta", "diffusion_rate"]:
        vals, scores = evaluate_1d_profile(data, params_dict, pname, max_blob_frac=0.85)
        plot_1d_profile(vals, scores, pname, case_label, params_dict[pname], opt_score_at_opt)


def run_patch_case(patch_image, params_dict, case_label, use_local=True):
    """Run 2D slices and 1D profiles for a patch case."""
    print(f"\n  === Patch case: {case_label} ({patch_image}) ===")
    data = prepare_patch_data(patch_image)
    opt_score_at_opt = score_at(data, params_dict, max_blob_frac=1.0)
    print(f"    Score at saved params: {opt_score_at_opt:.3f}")

    for px, py in SLICE_PAIRS:
        print(f"    Slice: {px} vs {py}")
        xs, ys, scores = evaluate_2d_slice(data, params_dict, px, py, max_blob_frac=1.0)
        plot_2d_surface(xs, ys, scores, px, py, case_label,
                       params_dict[px], params_dict[py], opt_score_at_opt)

    for pname in ["energy_threshold", "mu", "beta", "diffusion_rate"]:
        vals, scores = evaluate_1d_profile(data, params_dict, pname, max_blob_frac=1.0)
        plot_1d_profile(vals, scores, pname, case_label, params_dict[pname], opt_score_at_opt)


# ── Summary ──
def write_summary():
    lines = [
        "Optimization Surface Analysis — Summary",
        "=" * 50,
        "",
        "1. Log Discovery",
        "   No per-candidate evaluation logs found. The optimizer saves",
        "   only the final best parameters per image. A true global",
        "   optimization surface CANNOT be reconstructed.",
        "",
        "2. Method Used: Local Objective Slices",
        "   For each selected case (leaf or patch), we held 5 of 7",
        "   parameters fixed at their saved optimum and varied 2 on a",
        f"   dense grid ({GRID_N}x{GRID_N}={GRID_N**2} evals) across the full BOUNDS range.",
        f"   1D profiles used {PROFILE_N} points per parameter.",
        f"   Leaves downscaled {DOWNSCALE}x for speed; snake iters=6, PDE iters=8.",
        "   Score was computed deterministically using evaluate_params_refined().",
        "",
        "   IMPORTANT: These are LOCAL slices through the 7-D objective,",
        "   not the full landscape. The true surface has interactions",
        "   among all 7 parameters that these 2D slices cannot capture.",
        "",
        "3. Cases Analyzed",
        "   Leaves (downscaled 0.5x for speed):",
        "     - leaf_high: DSC_0163 (highest score, 45.7)",
        "     - leaf_low: DSC_0261 (lowest score, 11.4)",
        "   Patches (full 448x448 resolution):",
        "     - patch_bestdD: DSC_0261__y00448_x02688 (dD = +0.498)",
        "     - patch_worstdD: DSC_0370__y00896_x02688 (dD = -0.651)",
        "",
        "4. Key Observations (to be filled after running)",
        "   - Check whether score surfaces show clear peaks or flat plateaus",
        "   - Check whether energy_threshold has sharp vs broad optima",
        "   - Check whether high-score vs low-score cases differ in curvature",
        "   - Check whether patch best-dD vs worst-dD differ in surface structure",
        "",
        "5. Plot Types",
        "   - local_score_surface_*: 3D surface plots (score as z-axis)",
        "   - local_score_heatmap_*: top-down heatmaps with contour lines",
        "   - score_profile_*: 1D score curves along single parameter axes",
        "   All include a red star at the saved optimum.",
        "",
        "6. Reminder",
        "   These are LOCAL RESCORED SLICES, not globally recovered surfaces.",
        "   The optimizer may have explored different regions during its",
        "   random sampling phase that are not represented here.",
    ]
    path = os.path.join(OUT_DIR, "optimization_surface_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Wrote {path}")


# ── Main ──
def main():
    print("=== Part 1: Log Discovery ===")
    write_log_discovery()

    print("\n=== Part 3: Local Objective Slices ===")
    print("  (No candidate logs found -> using local rescoring)")

    # Load leaf data
    opt_df = pd.read_csv(OPT_CSV)
    opt_df = opt_df[opt_df["status"] == "ok"].copy()
    opt_df = opt_df.sort_values("score")

    leaf_cases = {
        "leaf_low": opt_df.iloc[0],
        "leaf_high": opt_df.iloc[-1],
    }

    for label, row in leaf_cases.items():
        params = {p: (int(row[p]) if p == "energy_threshold" else float(row[p])) for p in PARAM_NAMES}
        run_leaf_case(row["filename"], params, label)

    # Load patch data
    v2_df = pd.read_csv(V2_CSV)
    v2_df["dD"] = v2_df["dice_local_teacher"] - v2_df["dice_global_teacher"]
    v2_df = v2_df.sort_values("dD")

    patch_cases = {
        "patch_worstdD": v2_df.iloc[0],
        "patch_bestdD": v2_df.iloc[-1],
    }

    for label, row in patch_cases.items():
        # Use theta_local for the patch case
        params = {}
        for p in PARAM_NAMES:
            val = row[f"theta_local_{p}"]
            params[p] = int(val) if p == "energy_threshold" else float(val)
        run_patch_case(row["patch_image"], params, label)

    print("\n=== Part 5: Summary ===")
    write_summary()

    print("\n=== Output listing ===")
    files = sorted(os.listdir(OUT_DIR))
    for f in files:
        if f.startswith("_"):
            continue
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        if sz > 1024*1024:
            print(f"  {f}  ({sz/1024/1024:.1f}M)")
        else:
            print(f"  {f}  ({sz//1024}K)")
    print(f"\nTotal files: {len(files)}")
    print("Done.")


if __name__ == "__main__":
    main()
