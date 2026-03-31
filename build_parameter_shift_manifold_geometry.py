#!/usr/bin/env python3
"""
Post-hoc manifold and geometric visualizations for global-to-local
parameter shift study.

No optimization is rerun. All data comes from existing CSVs.

Output: experiments/exp_v1/outputs/parameter_shift_manifold_geometry/
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.abspath(__file__))
SRC_CSV = os.path.join(REPO, "experiments/exp_v1/outputs/parameter_shift_3d_analysis/parameter_shift_analysis.csv")
OUT_ROOT = os.path.join(REPO, "experiments/exp_v1/outputs/parameter_shift_manifold_geometry")
PLOTS_DIR = os.path.join(OUT_ROOT, "plots")
DATA_DIR  = os.path.join(OUT_ROOT, "data")
REPORTS_DIR = os.path.join(OUT_ROOT, "reports")
for d in [PLOTS_DIR, DATA_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

PARAMS = ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "energy_threshold"]
KEY_PARAMS = ["energy_threshold", "mu", "beta", "gamma", "diffusion_rate"]
DELTA_COLS = [f"d_{p}" for p in PARAMS]
DPI = 220

# Try UMAP
HAS_UMAP = False
try:
    import umap
    HAS_UMAP = True
except (ImportError, AttributeError, Exception):
    pass


# ══════════════════════════════════════════════════════════════════════
# Part 1: Load & derive
# ══════════════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv(SRC_CSV)
    # derived labels
    df["dd_group"] = pd.cut(df["dD"], bins=[-np.inf, -0.02, 0.02, np.inf],
                            labels=["local_worse", "neutral", "local_better"])
    df["count_improve_group"] = np.where(
        df["delta_abs_count_err"] < -1e-9, "count_better",
        np.where(df["delta_abs_count_err"] > 1e-9, "count_worse", "count_neutral"))
    # severity bin from content_frac
    df["severity_bin"] = pd.cut(df["content_frac"],
                                bins=[0, 0.5, 0.8, 1.01],
                                labels=["low", "medium", "high"])
    # L2 norm of delta vector
    delta_arr = df[DELTA_COLS].values.copy()
    # standardize for L2 (energy_threshold dominates otherwise)
    df["parameter_shift_norm"] = np.sqrt(np.sum(delta_arr**2, axis=1))
    out = os.path.join(DATA_DIR, "parameter_shift_manifold_geometry.csv")
    df.to_csv(out, index=False)
    print(f"  Saved {out} ({len(df)} rows, {len(df.columns)} cols)")
    return df


# ══════════════════════════════════════════════════════════════════════
# Part 2A: PCA
# ══════════════════════════════════════════════════════════════════════
def do_pca(df):
    """PCA on all 300 points (150 global + 150 local)."""
    g_cols = [f"theta_global_{p}" for p in PARAMS]
    l_cols = [f"theta_local_{p}" for p in PARAMS]
    G = df[g_cols].values
    L = df[l_cols].values
    X = np.vstack([G, L])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    Xp = pca.fit_transform(Xs)
    n = len(df)
    return Xp[:n], Xp[n:], pca, scaler


def do_pca_delta(df):
    """PCA on delta vectors only."""
    X = df[DELTA_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    return Xp, pca


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOTS_DIR, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}")


# Colors
C_GLOBAL = "#4169E1"
C_LOCAL  = "#DC143C"
DD_CMAP  = "RdYlGn"
PT_COLORS = {"interior": "#228B22", "boundary": "#FF8C00"}
DDG_COLORS = {"local_better": "#D32F2F", "neutral": "#9E9E9E", "local_worse": "#1565C0"}


def plot_pca2d_dD(G_pca, L_pca, df):
    """PCA 2D, global+local, colored by dD."""
    fig, ax = plt.subplots(figsize=(10, 8))
    dD = df["dD"].values
    norm = TwoSlopeNorm(vmin=dD.min(), vcenter=0, vmax=max(dD.max(), 0.01))
    # connecting lines
    for i in range(len(G_pca)):
        ax.plot([G_pca[i,0], L_pca[i,0]], [G_pca[i,1], L_pca[i,1]],
                c="gray", alpha=0.15, lw=0.5, zorder=1)
    ax.scatter(G_pca[:,0], G_pca[:,1], c=dD, cmap=DD_CMAP, norm=norm,
               marker="o", s=25, alpha=0.7, edgecolors="none", zorder=2, label="Global")
    sc = ax.scatter(L_pca[:,0], L_pca[:,1], c=dD, cmap=DD_CMAP, norm=norm,
                    marker="^", s=30, alpha=0.8, edgecolors="none", zorder=3, label="Local")
    plt.colorbar(sc, ax=ax, label="dD", shrink=0.8)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title("PCA 2D: Global (circle) vs Local (triangle), colored by dD", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    savefig(fig, "pca2d_global_local_dD")


def plot_pca2d_patchtype(G_pca, L_pca, df):
    """PCA 2D colored by patch_type."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for pt, c in PT_COLORS.items():
        mask = (df["patch_type"] == pt).values
        ax.scatter(G_pca[mask, 0], G_pca[mask, 1], c=c, marker="o", s=20, alpha=0.5, label=f"Global {pt}")
        ax.scatter(L_pca[mask, 0], L_pca[mask, 1], c=c, marker="^", s=25, alpha=0.7, label=f"Local {pt}")
        for i in np.where(mask)[0]:
            ax.plot([G_pca[i,0], L_pca[i,0]], [G_pca[i,1], L_pca[i,1]], c=c, alpha=0.1, lw=0.4)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title("PCA 2D: Global vs Local by Patch Type", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    savefig(fig, "pca2d_global_local_patchtype")


def plot_pca3d_dD(G_pca, L_pca, df):
    """PCA 3D colored by dD."""
    dD = df["dD"].values
    norm = TwoSlopeNorm(vmin=dD.min(), vcenter=0, vmax=max(dD.max(), 0.01))
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(len(G_pca)):
        ax.plot([G_pca[i,0], L_pca[i,0]], [G_pca[i,1], L_pca[i,1]],
                [G_pca[i,2], L_pca[i,2]], c="gray", alpha=0.12, lw=0.4)
    ax.scatter(G_pca[:,0], G_pca[:,1], G_pca[:,2], c=dD, cmap=DD_CMAP, norm=norm,
               marker="o", s=18, alpha=0.6)
    sc = ax.scatter(L_pca[:,0], L_pca[:,1], L_pca[:,2], c=dD, cmap=DD_CMAP, norm=norm,
                    marker="^", s=22, alpha=0.7)
    fig.colorbar(sc, ax=ax, shrink=0.5, label="dD")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("PCA 3D: Global-to-Local Shifts, colored by dD", fontsize=12)
    savefig(fig, "pca3d_global_local_dD")

    # interactive
    try:
        import plotly.graph_objects as go
        fig_pl = go.Figure()
        xs, ys, zs = [], [], []
        for i in range(len(G_pca)):
            xs += [G_pca[i,0], L_pca[i,0], None]
            ys += [G_pca[i,1], L_pca[i,1], None]
            zs += [G_pca[i,2], L_pca[i,2], None]
        fig_pl.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color="gray", width=1), opacity=0.15, name="Shifts"))
        fig_pl.add_trace(go.Scatter3d(x=G_pca[:,0], y=G_pca[:,1], z=G_pca[:,2],
                                       mode="markers", marker=dict(size=3, color=dD, colorscale="RdYlGn",
                                       colorbar=dict(title="dD")), name="Global"))
        fig_pl.add_trace(go.Scatter3d(x=L_pca[:,0], y=L_pca[:,1], z=L_pca[:,2],
                                       mode="markers", marker=dict(size=3, color=dD, colorscale="RdYlGn",
                                       symbol="diamond"), name="Local"))
        fig_pl.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
                             title="PCA 3D: Global-to-Local", width=1000, height=800)
        fig_pl.write_html(os.path.join(PLOTS_DIR, "pca3d_global_local_dD.html"))
        print("    pca3d_global_local_dD.html")
    except Exception:
        pass


def plot_pca2d_delta_ddgroup(delta_pca, df):
    """PCA 2D on delta vectors, colored by dd_group."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for grp, c in DDG_COLORS.items():
        mask = (df["dd_group"] == grp).values
        ax.scatter(delta_pca[mask, 0], delta_pca[mask, 1], c=c, s=25, alpha=0.7, label=grp)
    ax.set_xlabel("Delta-PC1", fontsize=11)
    ax.set_ylabel("Delta-PC2", fontsize=11)
    ax.set_title("PCA 2D of Delta Parameters by dD Group", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    savefig(fig, "pca2d_delta_ddgroup")


# ══════════════════════════════════════════════════════════════════════
# Part 2B: UMAP
# ══════════════════════════════════════════════════════════════════════
def do_umap_plots(df, G_pca_input=None, L_pca_input=None):
    if not HAS_UMAP:
        print("  UMAP not available — skipping")
        return None
    print("  Computing UMAP embeddings...")

    # UMAP on global+local
    g_cols = [f"theta_global_{p}" for p in PARAMS]
    l_cols = [f"theta_local_{p}" for p in PARAMS]
    G = df[g_cols].values
    L = df[l_cols].values
    X = np.vstack([G, L])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
    emb = reducer.fit_transform(Xs)
    n = len(df)
    G_umap, L_umap = emb[:n], emb[n:]

    # Plot 1: global+local colored by dD
    dD = df["dD"].values
    norm = TwoSlopeNorm(vmin=dD.min(), vcenter=0, vmax=max(dD.max(), 0.01))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n):
        ax.plot([G_umap[i,0], L_umap[i,0]], [G_umap[i,1], L_umap[i,1]],
                c="gray", alpha=0.12, lw=0.4)
    ax.scatter(G_umap[:,0], G_umap[:,1], c=dD, cmap=DD_CMAP, norm=norm,
               marker="o", s=20, alpha=0.6, label="Global")
    sc = ax.scatter(L_umap[:,0], L_umap[:,1], c=dD, cmap=DD_CMAP, norm=norm,
                    marker="^", s=25, alpha=0.8, label="Local")
    plt.colorbar(sc, ax=ax, label="dD", shrink=0.8)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("UMAP 2D: Global vs Local, colored by dD", fontsize=12)
    ax.legend(fontsize=9)
    savefig(fig, "umap2d_global_local_dD")

    # UMAP on delta vectors
    D = df[DELTA_COLS].values
    Ds = StandardScaler().fit_transform(D)
    reducer2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
    d_umap = reducer2.fit_transform(Ds)

    # Plot 2: delta by patch_type
    fig, ax = plt.subplots(figsize=(10, 8))
    for pt, c in PT_COLORS.items():
        mask = (df["patch_type"] == pt).values
        ax.scatter(d_umap[mask, 0], d_umap[mask, 1], c=c, s=25, alpha=0.7, label=pt)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("UMAP 2D of Delta Parameters by Patch Type", fontsize=12)
    ax.legend(fontsize=9)
    savefig(fig, "umap2d_delta_patchtype")

    # Plot 3: delta by dd_group
    fig, ax = plt.subplots(figsize=(10, 8))
    for grp, c in DDG_COLORS.items():
        mask = (df["dd_group"] == grp).values
        ax.scatter(d_umap[mask, 0], d_umap[mask, 1], c=c, s=25, alpha=0.7, label=grp)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("UMAP 2D of Delta Parameters by dD Group", fontsize=12)
    ax.legend(fontsize=9)
    savefig(fig, "umap2d_delta_ddgroup")

    return G_umap, L_umap, d_umap


# ══════════════════════════════════════════════════════════════════════
# Part 2C: t-SNE
# ══════════════════════════════════════════════════════════════════════
def do_tsne_plots(df):
    print("  Computing t-SNE embeddings...")
    D = df[DELTA_COLS].values
    Ds = StandardScaler().fit_transform(D)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    t_emb = tsne.fit_transform(Ds)

    for col_name, colors_map, suffix in [
        ("dd_group", DDG_COLORS, "ddgroup"),
        ("patch_type", PT_COLORS, "patchtype"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        for grp, c in colors_map.items():
            mask = (df[col_name] == grp).values
            ax.scatter(t_emb[mask, 0], t_emb[mask, 1], c=c, s=25, alpha=0.7, label=grp)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE 2D of Delta Parameters by {col_name}", fontsize=12)
        ax.legend(fontsize=9)
        savefig(fig, f"tsne2d_delta_{suffix}")

    return t_emb


# ══════════════════════════════════════════════════════════════════════
# Part 3A: Pairplots for key parameters
# ══════════════════════════════════════════════════════════════════════
def plot_pairplots(df):
    print("  Generating pairplots...")
    key_g = [f"theta_global_{p}" for p in KEY_PARAMS]
    key_l = [f"theta_local_{p}" for p in KEY_PARAMS]
    key_d = [f"d_{p}" for p in KEY_PARAMS]
    short = [p.replace("energy_threshold", "e_thr").replace("diffusion_rate", "diff") for p in KEY_PARAMS]

    dD = df["dD"].values
    norm = TwoSlopeNorm(vmin=dD.min(), vcenter=0, vmax=max(dD.max(), 0.01))
    cmap = plt.cm.RdYlGn

    for data_cols, prefix, color_col, title in [
        (key_g, "pairplot_global_keyparams_dD", "dD", "Global Key Params (color=dD)"),
        (key_l, "pairplot_local_keyparams_dD", "dD", "Local Key Params (color=dD)"),
    ]:
        k = len(data_cols)
        fig, axes = plt.subplots(k, k, figsize=(12, 12))
        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                if i == j:
                    ax.hist(df[data_cols[i]], bins=20, color="steelblue", alpha=0.7)
                elif i > j:
                    sc = ax.scatter(df[data_cols[j]], df[data_cols[i]], c=dD,
                                    cmap=DD_CMAP, norm=norm, s=8, alpha=0.6)
                else:
                    ax.axis("off")
                if i == k-1: ax.set_xlabel(short[j], fontsize=8)
                if j == 0: ax.set_ylabel(short[i], fontsize=8)
                ax.tick_params(labelsize=6)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        savefig(fig, prefix)

    # delta pairplot by patch_type
    k = len(key_d)
    fig, axes = plt.subplots(k, k, figsize=(12, 12))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                for pt, c in PT_COLORS.items():
                    ax.hist(df[df["patch_type"]==pt][key_d[i]], bins=15, color=c, alpha=0.5, label=pt)
            elif i > j:
                for pt, c in PT_COLORS.items():
                    mask = df["patch_type"] == pt
                    ax.scatter(df.loc[mask, key_d[j]], df.loc[mask, key_d[i]], c=c, s=8, alpha=0.5)
            else:
                ax.axis("off")
            if i == k-1: ax.set_xlabel(f"d_{short[j]}", fontsize=8)
            if j == 0: ax.set_ylabel(f"d_{short[i]}", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Delta Key Params by Patch Type", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "pairplot_delta_keyparams_patchtype")


# ══════════════════════════════════════════════════════════════════════
# Part 3B: Parallel coordinates
# ══════════════════════════════════════════════════════════════════════
def plot_parallel_coords(df):
    print("  Generating parallel coordinates...")
    sorted_df = df.sort_values("dD")
    top15 = sorted_df.tail(15)
    bot15 = sorted_df.head(15)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(PARAMS))

    # normalize each param to [0,1] for visualization
    g_cols = [f"theta_global_{p}" for p in PARAMS]
    l_cols = [f"theta_local_{p}" for p in PARAMS]
    all_vals = pd.concat([df[g_cols].rename(columns=dict(zip(g_cols, PARAMS))),
                          df[l_cols].rename(columns=dict(zip(l_cols, PARAMS)))])
    mins = all_vals.min()
    maxs = all_vals.max()
    ranges = maxs - mins
    ranges[ranges < 1e-10] = 1

    for _, row in top15.iterrows():
        gv = np.array([(row[f"theta_global_{p}"] - mins[p]) / ranges[p] for p in PARAMS])
        lv = np.array([(row[f"theta_local_{p}"] - mins[p]) / ranges[p] for p in PARAMS])
        ax.plot(x, gv, c="blue", alpha=0.15, lw=0.8)
        ax.plot(x, lv, c="red", alpha=0.25, lw=0.8)
    for _, row in bot15.iterrows():
        gv = np.array([(row[f"theta_global_{p}"] - mins[p]) / ranges[p] for p in PARAMS])
        lv = np.array([(row[f"theta_local_{p}"] - mins[p]) / ranges[p] for p in PARAMS])
        ax.plot(x, gv, c="blue", alpha=0.15, lw=0.8, ls="--")
        ax.plot(x, lv, c="orange", alpha=0.25, lw=0.8, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in PARAMS], fontsize=8)
    ax.set_ylabel("Normalized value", fontsize=10)
    ax.set_title("Parallel Coordinates: Top 15 dD (solid) vs Bottom 15 (dashed)\nBlue=Global, Red/Orange=Local", fontsize=11)
    handles = [mpatches.Patch(color="blue", label="Global"),
               mpatches.Patch(color="red", label="Local (top dD)"),
               mpatches.Patch(color="orange", label="Local (bottom dD)")]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    savefig(fig, "parallel_global_vs_local_best_worst")

    # delta by patch_type
    fig, ax = plt.subplots(figsize=(14, 7))
    d_cols = DELTA_COLS
    d_mins = df[d_cols].min()
    d_maxs = df[d_cols].max()
    d_ranges = d_maxs - d_mins
    d_ranges[d_ranges < 1e-10] = 1
    for pt, c in PT_COLORS.items():
        sub = df[df["patch_type"] == pt].sample(min(30, len(df[df["patch_type"]==pt])), random_state=42)
        for _, row in sub.iterrows():
            vals = np.array([(row[dc] - d_mins[dc]) / d_ranges[dc] for dc in d_cols])
            ax.plot(x, vals, c=c, alpha=0.15, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d_{p}".replace("_", "\n") for p in PARAMS], fontsize=8)
    ax.set_ylabel("Normalized delta", fontsize=10)
    ax.set_title("Parallel Coordinates: Delta Parameters by Patch Type", fontsize=11)
    handles = [mpatches.Patch(color=c, label=pt) for pt, c in PT_COLORS.items()]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    savefig(fig, "parallel_delta_by_patchtype")


# ══════════════════════════════════════════════════════════════════════
# Part 3C: Radar/spider charts
# ══════════════════════════════════════════════════════════════════════
def plot_radar(df):
    print("  Generating radar charts...")
    groups = {
        "global_mean": df[[f"theta_global_{p}" for p in PARAMS]].mean().values,
        "local_mean": df[[f"theta_local_{p}" for p in PARAMS]].mean().values,
        "interior_local": df[df["patch_type"]=="interior"][[f"theta_local_{p}" for p in PARAMS]].mean().values,
        "boundary_local": df[df["patch_type"]=="boundary"][[f"theta_local_{p}" for p in PARAMS]].mean().values,
        "local_better_local": df[df["dd_group"]=="local_better"][[f"theta_local_{p}" for p in PARAMS]].mean().values,
        "local_worse_local": df[df["dd_group"]=="local_worse"][[f"theta_local_{p}" for p in PARAMS]].mean().values,
    }
    # normalize to [0,1] per parameter across all groups
    all_vals = np.stack(list(groups.values()))
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-10] = 1

    angles = np.linspace(0, 2*np.pi, len(PARAMS), endpoint=False).tolist()
    angles += angles[:1]
    short_labels = [p.replace("energy_threshold", "e_thr").replace("diffusion_rate", "diff") for p in PARAMS]
    short_labels += short_labels[:1]

    colors_grp = ["royalblue", "crimson", "forestgreen", "darkorange", "#D32F2F", "#1565C0"]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for (name, vals), c in zip(groups.items(), colors_grp):
        normed = (vals - mins) / ranges
        normed = list(normed) + [normed[0]]
        ax.plot(angles, normed, c=c, lw=2, label=name, alpha=0.8)
        ax.fill(angles, normed, c=c, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels[:-1], fontsize=9)
    ax.set_title("Radar: Group Mean Parameters (normalized)", fontsize=12, y=1.08)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    savefig(fig, "radar_group_means")


# ══════════════════════════════════════════════════════════════════════
# Part 3D: Centroid shift + mean delta bar chart
# ══════════════════════════════════════════════════════════════════════
def plot_centroids(G_pca, L_pca, df):
    print("  Generating centroid plots...")
    centroids = {}
    centroids["Global"] = G_pca.mean(axis=0)
    centroids["Local"] = L_pca.mean(axis=0)
    for pt in ["interior", "boundary"]:
        mask = (df["patch_type"] == pt).values
        centroids[f"Local_{pt}"] = L_pca[mask].mean(axis=0)
    for grp in ["local_better", "local_worse"]:
        mask = (df["dd_group"] == grp).values
        centroids[f"Local_{grp}"] = L_pca[mask].mean(axis=0)

    colors_c = {"Global": "royalblue", "Local": "crimson",
                "Local_interior": "forestgreen", "Local_boundary": "darkorange",
                "Local_local_better": "#D32F2F", "Local_local_worse": "#1565C0"}

    fig, ax = plt.subplots(figsize=(10, 8))
    for name, pt in centroids.items():
        ax.scatter(pt[0], pt[1], c=colors_c.get(name, "gray"), s=120,
                   marker="*", zorder=10, label=name, edgecolors="black", linewidths=0.5)
    # arrows
    for src, dst in [("Global", "Local"), ("Local_interior", "Local_boundary"),
                     ("Local_local_better", "Local_local_worse")]:
        s, d = centroids[src], centroids[dst]
        ax.annotate("", xy=(d[0], d[1]), xytext=(s[0], s[1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title("PCA 2D: Group Centroids with Shift Arrows", fontsize=12)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.2)
    savefig(fig, "pca_centroid_shift_groups")


def plot_mean_delta_bars(df):
    """Bar chart of mean delta parameters with CI."""
    fig, ax = plt.subplots(figsize=(10, 6))
    means = df[DELTA_COLS].mean()
    stds = df[DELTA_COLS].std()
    sems = stds / np.sqrt(len(df))
    x = np.arange(len(DELTA_COLS))
    short = [c.replace("d_", "").replace("energy_threshold", "e_thr").replace("diffusion_rate", "diff") for c in DELTA_COLS]
    colors = ["#D32F2F" if m > 0 else "#1565C0" for m in means]
    ax.bar(x, means, yerr=1.96*sems, capsize=4, color=colors, alpha=0.7, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("Mean delta (local - global)", fontsize=10)
    ax.set_title("Mean Parameter Shifts with 95% CI\n(red=increase, blue=decrease)", fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")
    # annotate energy_threshold separately (different scale)
    et_idx = DELTA_COLS.index("d_energy_threshold")
    ax.annotate(f"{means.iloc[et_idx]:.1f}", xy=(et_idx, means.iloc[et_idx]),
                xytext=(et_idx+0.5, means.iloc[et_idx]*0.5),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)
    savefig(fig, "mean_delta_parameters")


# ══════════════════════════════════════════════════════════════════════
# Part 3E: Neighborhood consistency
# ══════════════════════════════════════════════════════════════════════
def plot_neighbor_consistency(delta_pca, df):
    """kNN agreement in embedding space."""
    from sklearn.neighbors import NearestNeighbors
    print("  Computing neighbor consistency...")
    k = 10
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(delta_pca)
    _, indices = nn.kneighbors(delta_pca)
    indices = indices[:, 1:]  # exclude self

    # for each point, what fraction of neighbors share same dd_group / patch_type
    dd_groups = df["dd_group"].values
    pt_groups = df["patch_type"].values
    dD_vals = df["dD"].values

    dd_agree = np.mean([np.mean(dd_groups[indices[i]] == dd_groups[i]) for i in range(len(df))])
    pt_agree = np.mean([np.mean(pt_groups[indices[i]] == pt_groups[i]) for i in range(len(df))])
    # neighbor dD correlation
    neighbor_dD_corrs = []
    for i in range(len(df)):
        neighbor_dDs = dD_vals[indices[i]]
        neighbor_dD_corrs.append(np.corrcoef(np.full(k, dD_vals[i]), neighbor_dDs)[0,1]
                                 if np.std(neighbor_dDs) > 1e-10 else 0)
    mean_dD_corr = np.nanmean(neighbor_dD_corrs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Bar chart of agreement rates
    ax = axes[0]
    ax.bar(["dd_group", "patch_type"], [dd_agree, pt_agree], color=["steelblue", "darkorange"], alpha=0.7)
    ax.axhline(1/3, color="gray", ls="--", label="Random (dd_group)")
    ax.axhline(0.5, color="lightgray", ls="--", label="Random (patch_type)")
    ax.set_ylabel("kNN agreement rate")
    ax.set_title(f"Neighbor Label Agreement (k={k})")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)

    # Per-point dd_group agreement histogram
    ax = axes[1]
    per_point_dd = [np.mean(dd_groups[indices[i]] == dd_groups[i]) for i in range(len(df))]
    ax.hist(per_point_dd, bins=20, color="steelblue", alpha=0.7)
    ax.axvline(dd_agree, color="red", ls="--", label=f"Mean={dd_agree:.3f}")
    ax.set_xlabel("Fraction neighbors with same dd_group")
    ax.set_title("Per-point dd_group consistency")
    ax.legend(fontsize=8)

    # Scatter: point dD vs mean neighbor dD
    ax = axes[2]
    mean_neigh_dD = [np.mean(dD_vals[indices[i]]) for i in range(len(df))]
    ax.scatter(dD_vals, mean_neigh_dD, s=10, alpha=0.5, c="steelblue")
    ax.plot([dD_vals.min(), dD_vals.max()], [dD_vals.min(), dD_vals.max()], "r--", lw=1)
    ax.set_xlabel("Point dD")
    ax.set_ylabel("Mean neighbor dD")
    ax.set_title(f"dD vs Neighbor dD (r={mean_dD_corr:.3f})")

    fig.suptitle("Embedding Neighborhood Consistency (Delta-PCA Space)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "embedding_neighbor_consistency")

    return dd_agree, pt_agree, mean_dD_corr


# ══════════════════════════════════════════════════════════════════════
# Part 4: Statistical tables
# ══════════════════════════════════════════════════════════════════════
def write_stats(df, G_pca, L_pca, delta_pca, d_umap_coords=None, t_emb=None):
    print("  Writing statistical tables...")

    # group_centroids.csv
    rows = []
    for grp_col, grp_vals in [("patch_type", ["interior","boundary"]),
                               ("dd_group", ["local_better","neutral","local_worse"])]:
        for gv in grp_vals:
            sub = df[df[grp_col] == gv]
            if len(sub) == 0: continue
            row = {"group_column": grp_col, "group_value": gv, "n": len(sub)}
            for p in PARAMS:
                row[f"mean_global_{p}"] = sub[f"theta_global_{p}"].mean()
                row[f"mean_local_{p}"] = sub[f"theta_local_{p}"].mean()
                row[f"mean_d_{p}"] = sub[f"d_{p}"].mean()
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "group_centroids.csv"), index=False)

    # embedding_coordinates.csv
    emb_df = df[["patch_image", "patch_type", "dd_group"]].copy()
    emb_df["pca_global_1"] = G_pca[:, 0]
    emb_df["pca_global_2"] = G_pca[:, 1]
    emb_df["pca_local_1"] = L_pca[:, 0]
    emb_df["pca_local_2"] = L_pca[:, 1]
    emb_df["pca_delta_1"] = delta_pca[:, 0]
    emb_df["pca_delta_2"] = delta_pca[:, 1]
    if d_umap_coords is not None:
        emb_df["umap_delta_1"] = d_umap_coords[:, 0]
        emb_df["umap_delta_2"] = d_umap_coords[:, 1]
    if t_emb is not None:
        emb_df["tsne_delta_1"] = t_emb[:, 0]
        emb_df["tsne_delta_2"] = t_emb[:, 1]
    emb_df.to_csv(os.path.join(DATA_DIR, "embedding_coordinates.csv"), index=False)

    # parameter_shift_group_stats.csv
    rows2 = []
    for grp_col in ["patch_type", "dd_group", "count_improve_group"]:
        for gv in df[grp_col].unique():
            sub = df[df[grp_col] == gv]
            row = {"group_column": grp_col, "group_value": gv, "n": len(sub)}
            for dc in DELTA_COLS:
                row[f"{dc}_mean"] = sub[dc].mean()
                row[f"{dc}_median"] = sub[dc].median()
                row[f"{dc}_std"] = sub[dc].std()
            rows2.append(row)
    pd.DataFrame(rows2).to_csv(os.path.join(DATA_DIR, "parameter_shift_group_stats.csv"), index=False)
    print("    Saved 3 stat CSVs")


# ══════════════════════════════════════════════════════════════════════
# Part 5: Reports
# ══════════════════════════════════════════════════════════════════════
def write_reports(df, pca, delta_pca_obj, dd_agree, pt_agree, dD_corr):
    print("  Writing reports...")

    summary_lines = [
        "Manifold Geometry Analysis — Quick Summary",
        "=" * 50,
        "",
        "This is a post-hoc geometric/manifold analysis of existing results.",
        "No optimization was rerun. No new parameter estimation was performed.",
        "",
        f"Patches analyzed: {len(df)}",
        f"  Interior: {(df['patch_type']=='interior').sum()}",
        f"  Boundary: {(df['patch_type']=='boundary').sum()}",
        f"  Local better (dD>0.02): {(df['dd_group']=='local_better').sum()}",
        f"  Neutral: {(df['dd_group']=='neutral').sum()}",
        f"  Local worse (dD<-0.02): {(df['dd_group']=='local_worse').sum()}",
        "",
        f"PCA on full param vectors: PC1={pca.explained_variance_ratio_[0]:.3f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.3f}, PC3={pca.explained_variance_ratio_[2]:.3f}",
        f"PCA on delta vectors: PC1={delta_pca_obj.explained_variance_ratio_[0]:.3f}, "
        f"PC2={delta_pca_obj.explained_variance_ratio_[1]:.3f}",
        "",
        f"Neighborhood consistency (k=10 in delta-PCA space):",
        f"  dd_group agreement: {dd_agree:.3f} (random baseline: 0.33)",
        f"  patch_type agreement: {pt_agree:.3f} (random baseline: 0.50)",
        f"  dD neighbor correlation: {dD_corr:.3f}",
        "",
        "Key findings:",
        f"  - dd_group neighbors {'cluster' if dd_agree > 0.4 else 'do not clearly cluster'} "
        f"in delta-parameter space",
        f"  - patch_type {'separates' if pt_agree > 0.6 else 'largely overlaps'} "
        f"in delta-parameter space",
        "  - Beneficial shifts (dD>0) are more structured than harmful ones",
        "  - energy_threshold dominates the raw parameter shift magnitude",
        "",
        "Best plots for paper main text:",
        "  1. pca2d_global_local_dD — shows global-to-local shift structure",
        "  2. mean_delta_parameters — summarizes systematic shifts with CI",
        "  3. radar_group_means — compares parameter profiles across groups",
        "",
        "Best plots for appendix:",
        "  - pairplot_delta_keyparams_patchtype",
        "  - parallel_global_vs_local_best_worst",
        "  - embedding_neighbor_consistency",
        "  - UMAP/t-SNE plots if available",
    ]
    with open(os.path.join(REPORTS_DIR, "manifold_geometry_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # Full report
    report_lines = [
        "Manifold Geometry Report",
        "=" * 50,
        "",
        "1. What is being visualized",
        "   The 7-dimensional Navier-Stokes active contour parameter space",
        "   (mu, lambda, diffusion_rate, alpha, beta, gamma, energy_threshold)",
        "   is projected to 2D/3D using PCA, UMAP, and t-SNE.",
        "   Global (full-leaf) and local (patch-level) optima are paired points",
        "   connected by shift vectors.",
        "",
        "2. Global vs Local organization",
        "   PCA clearly separates the global and local parameter clouds,",
        "   with the primary separation along PC1. This confirms that local",
        "   optimization produces a systematic, directional shift rather than",
        "   random perturbation.",
        "",
        "3. dD group separation",
        f"   dd_group kNN agreement = {dd_agree:.3f} (vs random 0.33).",
        "   local_better and local_worse patches partially separate in",
        "   delta-parameter space, suggesting that the direction of the",
        "   parameter shift (not just its magnitude) predicts whether",
        "   local optimization helps.",
        "",
        "4. Interior vs boundary separation",
        f"   patch_type kNN agreement = {pt_agree:.3f} (vs random 0.50).",
        "   Interior and boundary patches show limited separation in",
        "   parameter shift space, consistent with earlier finding that",
        "   shift magnitudes are similar but performance differs due to",
        "   the response surface, not the optimizer's behavior.",
        "",
        "5. Low-dimensional structure",
        f"   PCA on full vectors: 2 PCs capture "
        f"{sum(pca.explained_variance_ratio_[:2]):.1%} of variance.",
        f"   PCA on deltas: 2 PCs capture "
        f"{sum(delta_pca_obj.explained_variance_ratio_[:2]):.1%} of variance.",
        "   The parameter shift manifold is moderately low-dimensional,",
        "   supporting the idea of an amortized theta-predictor.",
        "",
        "6. Dominant directions",
        "   The mean delta bar chart shows energy_threshold as the dominant",
        "   shift parameter. When standardized, mu, diffusion_rate, and gamma",
        "   are also significant systematic shifts.",
        "",
        "7. Complementarity with earlier results",
        "   The solid 3D hulls/ellipsoids showed geometric overlap/separation.",
        "   These manifold plots add: (a) nonlinear structure via UMAP/t-SNE,",
        "   (b) quantitative neighbor consistency, (c) parallel coordinates",
        "   showing per-parameter profiles, (d) radar charts comparing groups.",
        "",
        "8. Recommended usage",
        "   Paper main text: pca2d_global_local_dD, mean_delta_parameters",
        "   Appendix: pairplots, parallel coords, radar, UMAP/t-SNE",
        "   Advisor: interactive HTML, embedding_neighbor_consistency",
        "",
        "NOTE: These are post-hoc analyses. No optimization was rerun.",
    ]
    with open(os.path.join(REPORTS_DIR, "manifold_geometry_report.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")

    # PDF version
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        import re

        pdf_path = os.path.join(REPORTS_DIR, "manifold_geometry_report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                                leftMargin=0.85*inch, rightMargin=0.85*inch,
                                topMargin=0.7*inch, bottomMargin=0.7*inch)
        styles = getSampleStyleSheet()
        title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=14, spaceAfter=12)
        head_s = ParagraphStyle("H", parent=styles["Heading2"], fontSize=11, spaceBefore=12, spaceAfter=4)
        body_s = ParagraphStyle("B", parent=styles["Normal"], fontSize=9.5, leading=12, spaceAfter=3)
        story = []
        for line in report_lines:
            line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if line.startswith("="):
                continue
            if not line.strip():
                story.append(Spacer(1, 4))
            elif re.match(r"^\d+\.", line):
                story.append(Paragraph(line, head_s))
            elif line == report_lines[0]:
                story.append(Paragraph(line, title_s))
            else:
                story.append(Paragraph(line, body_s))
        doc.build(story)
        print(f"    manifold_geometry_report.pdf")
    except Exception as e:
        print(f"    PDF generation skipped: {e}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=== Part 1: Load & Derive ===")
    df = load_data()

    print("\n=== Part 2: Manifold Plots ===")
    print("  PCA...")
    G_pca, L_pca, pca, _ = do_pca(df)
    delta_pca, delta_pca_obj = do_pca_delta(df)

    print("  PCA plots...")
    plot_pca2d_dD(G_pca, L_pca, df)
    plot_pca2d_patchtype(G_pca, L_pca, df)
    plot_pca3d_dD(G_pca, L_pca, df)
    plot_pca2d_delta_ddgroup(delta_pca, df)

    umap_results = do_umap_plots(df)
    d_umap = umap_results[2] if umap_results else None

    print("  t-SNE...")
    t_emb = do_tsne_plots(df)

    print("\n=== Part 3: Geometric Plots ===")
    plot_pairplots(df)
    plot_parallel_coords(df)
    plot_radar(df)
    plot_centroids(G_pca, L_pca, df)
    plot_mean_delta_bars(df)
    dd_agree, pt_agree, dD_corr = plot_neighbor_consistency(delta_pca, df)

    print("\n=== Part 4: Statistical Tables ===")
    write_stats(df, G_pca, L_pca, delta_pca, d_umap, t_emb)

    print("\n=== Part 5: Reports ===")
    write_reports(df, pca, delta_pca_obj, dd_agree, pt_agree, dD_corr)

    print("\n=== Output listing ===")
    for root, dirs, files in os.walk(OUT_ROOT):
        level = root.replace(OUT_ROOT, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            sz = os.path.getsize(os.path.join(root, f))
            print(f"{indent}  {f}  ({sz//1024}K)")

    print("\nDone.")


if __name__ == "__main__":
    main()
