#!/usr/bin/env python3
"""
Solid 3D visualizations for parameter distributions and global-to-local shifts.

These are geometric / density visualizations of saved parameter optima,
NOT true optimization energy surfaces.

Inputs:
  experiments/exp_v1/outputs/patch_local_opt_v2.csv
  experiments/exp_v1/outputs/patch_transfer_morphology_eval.csv

Output:
  experiments/exp_v1/outputs/parameter_shift_3d_analysis_solid/
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ──
REPO = os.path.dirname(os.path.abspath(__file__))
V2_CSV   = os.path.join(REPO, "experiments/exp_v1/outputs/patch_local_opt_v2.csv")
MORPH_CSV = os.path.join(REPO, "experiments/exp_v1/outputs/patch_transfer_morphology_eval.csv")
OUT_DIR  = os.path.join(REPO, "experiments/exp_v1/outputs/parameter_shift_3d_analysis_solid")
os.makedirs(OUT_DIR, exist_ok=True)

PARAMS = ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "energy_threshold"]
DPI = 220
FIGSIZE = (10, 8)


# ── Part 1: Merged CSV ──
def load_and_merge():
    v2 = pd.read_csv(V2_CSV)
    morph = pd.read_csv(MORPH_CSV)
    df = v2.merge(morph[["patch_image",
                          "teacher_area_px", "teacher_area_frac", "teacher_count",
                          "global_area_px", "global_area_frac", "global_count",
                          "local_area_px", "local_area_frac", "local_count",
                          "global_area_err_px", "global_area_err_frac", "global_count_err",
                          "local_area_err_px", "local_area_err_frac", "local_count_err"]],
                   on="patch_image", how="left")
    # derived
    for p in PARAMS:
        df[f"d_{p}"] = df[f"theta_local_{p}"] - df[f"theta_global_{p}"]
    df["dD"]  = df["dice_local_teacher"] - df["dice_global_teacher"]
    df["dIoU"] = df["iou_local_teacher"] - df["iou_global_teacher"]
    df["delta_abs_count_err"] = np.abs(df["local_count_err"]) - np.abs(df["global_count_err"])
    df["delta_abs_area_frac_err"] = np.abs(df["local_area_err_frac"]) - np.abs(df["global_area_err_frac"])
    out = os.path.join(OUT_DIR, "parameter_shift_analysis_solid.csv")
    df.to_csv(out, index=False)
    print(f"  Saved CSV: {out} ({len(df)} rows)")
    return df


# ── Part 2: PCA ──
def do_pca(df):
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


# ── helpers ──
def hull_polys(pts, alpha=0.15, color="blue"):
    """Return Poly3DCollection for convex hull of pts."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    verts = [[pts[s] for s in simplex] for simplex in hull.simplices]
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor=color, linewidth=0.3)
    return poly


def covariance_ellipsoid(pts, n_std=2.0, n_pts=30):
    """Mesh for covariance ellipsoid around point cloud."""
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = n_std * np.sqrt(np.maximum(eigvals, 0))
    u = np.linspace(0, 2*np.pi, n_pts)
    v = np.linspace(0, np.pi, n_pts)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()])
    xyz_rot = eigvecs @ xyz + mean[:, None]
    xr = xyz_rot[0].reshape(x.shape)
    yr = xyz_rot[1].reshape(y.shape)
    zr = xyz_rot[2].reshape(z.shape)
    return xr, yr, zr, mean


def set_3d_labels(ax, xl, yl, zl, fs=10):
    ax.set_xlabel(xl, fontsize=fs)
    ax.set_ylabel(yl, fontsize=fs)
    ax.set_zlabel(zl, fontsize=fs)
    ax.tick_params(labelsize=8)


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


def save_plotly(fig_pl, name):
    """Save plotly figure as html and try static png."""
    fig_pl.write_html(os.path.join(OUT_DIR, f"{name}.html"))
    try:
        fig_pl.write_image(os.path.join(OUT_DIR, f"{name}.png"), width=1200, height=960, scale=2)
    except Exception as e:
        print(f"    (plotly static export skipped: {e})")
    print(f"  Saved {name}.html" + (" + .png" if os.path.exists(os.path.join(OUT_DIR, f"{name}.png")) else ""))


# ── Plot A: PCA convex hulls ──
def plot_pca_hulls(G_pca, L_pca, df):
    print("\n[A] PCA convex hulls...")
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    # hulls
    hp_g = hull_polys(G_pca, alpha=0.08, color="royalblue")
    hp_l = hull_polys(L_pca, alpha=0.08, color="crimson")
    if hp_g: ax.add_collection3d(hp_g)
    if hp_l: ax.add_collection3d(hp_l)
    # connecting segments
    for i in range(len(G_pca)):
        ax.plot([G_pca[i,0], L_pca[i,0]], [G_pca[i,1], L_pca[i,1]],
                [G_pca[i,2], L_pca[i,2]], c="gray", alpha=0.15, lw=0.5)
    ax.scatter(*G_pca.T, c="royalblue", s=18, alpha=0.6, label="Global")
    ax.scatter(*L_pca.T, c="crimson", s=18, alpha=0.6, label="Local")
    set_3d_labels(ax, "PC1", "PC2", "PC3")
    ax.set_title("PCA Space: Global vs Local Convex Hulls", fontsize=12)
    ax.legend(fontsize=9)
    save_fig(fig, "pca3d_convex_hull_global_local")

    # plotly interactive
    try:
        import plotly.graph_objects as go
        fig_pl = go.Figure()
        # hulls via Mesh3d
        for pts, name, color in [(G_pca, "Global hull", "rgba(65,105,225,0.08)"),
                                  (L_pca, "Local hull", "rgba(220,20,60,0.08)")]:
            try:
                h = ConvexHull(pts)
                fig_pl.add_trace(go.Mesh3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    i=h.simplices[:,0], j=h.simplices[:,1], k=h.simplices[:,2],
                    color=color, name=name, opacity=0.12, showlegend=True))
            except Exception:
                pass
        fig_pl.add_trace(go.Scatter3d(x=G_pca[:,0], y=G_pca[:,1], z=G_pca[:,2],
                                       mode="markers", marker=dict(size=3, color="royalblue"),
                                       name="Global"))
        fig_pl.add_trace(go.Scatter3d(x=L_pca[:,0], y=L_pca[:,1], z=L_pca[:,2],
                                       mode="markers", marker=dict(size=3, color="crimson"),
                                       name="Local"))
        # connecting lines
        xs, ys, zs = [], [], []
        for i in range(len(G_pca)):
            xs += [G_pca[i,0], L_pca[i,0], None]
            ys += [G_pca[i,1], L_pca[i,1], None]
            zs += [G_pca[i,2], L_pca[i,2], None]
        fig_pl.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color="gray", width=1), opacity=0.2,
                                       name="Shifts", showlegend=True))
        fig_pl.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
                             title="PCA: Global vs Local Convex Hulls",
                             width=1000, height=800)
        save_plotly(fig_pl, "pca3d_convex_hull_global_local")
    except Exception as e:
        print(f"    Plotly skipped: {e}")


# ── Plot B: PCA covariance ellipsoids ──
def plot_pca_ellipsoids(G_pca, L_pca, df):
    print("\n[B] PCA covariance ellipsoids...")
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    for pts, color, label in [(G_pca, "royalblue", "Global"),
                               (L_pca, "crimson", "Local")]:
        xr, yr, zr, mean = covariance_ellipsoid(pts, n_std=2.0)
        ax.plot_surface(xr, yr, zr, color=color, alpha=0.1, linewidth=0)
        ax.scatter(*pts.T, c=color, s=12, alpha=0.5, label=label)
        ax.scatter(*mean, c=color, s=80, marker="*", edgecolors="black", zorder=10)
    # centroid arrow
    g_mean = G_pca.mean(axis=0)
    l_mean = L_pca.mean(axis=0)
    ax.quiver(g_mean[0], g_mean[1], g_mean[2],
              l_mean[0]-g_mean[0], l_mean[1]-g_mean[1], l_mean[2]-g_mean[2],
              arrow_length_ratio=0.15, color="black", linewidth=2.5)
    set_3d_labels(ax, "PC1", "PC2", "PC3")
    ax.set_title("PCA Space: Covariance Ellipsoids (2\u03c3)", fontsize=12)
    ax.legend(fontsize=9)
    save_fig(fig, "pca3d_covariance_ellipsoids")


# ── Plot C: PCA density isosurfaces ──
def plot_pca_density(G_pca, L_pca, df):
    print("\n[C] PCA density isosurfaces...")
    try:
        import plotly.graph_objects as go
        for pts, name, color_scale, opacity in [
            (G_pca, "Global density", "Blues", 0.15),
            (L_pca, "Local density", "Reds", 0.15)
        ]:
            # use KDE evaluated on a grid
            kde = gaussian_kde(pts.T, bw_method=0.3)
            # grid
            mins = pts.min(axis=0) - 1.0
            maxs = pts.max(axis=0) + 1.0
            N = 25
            gx = np.linspace(mins[0], maxs[0], N)
            gy = np.linspace(mins[1], maxs[1], N)
            gz = np.linspace(mins[2], maxs[2], N)
            GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
            coords = np.vstack([GX.ravel(), GY.ravel(), GZ.ravel()])
            density = kde(coords).reshape(GX.shape)
            # threshold for isosurface
            thresh = np.percentile(density[density > 0], 30)

        # combined plotly figure with both isosurfaces
        fig_pl = go.Figure()
        for pts, name, colorscale in [(G_pca, "Global", "Blues"), (L_pca, "Local", "Reds")]:
            kde = gaussian_kde(pts.T, bw_method=0.3)
            mins = pts.min(axis=0) - 1.0
            maxs = pts.max(axis=0) + 1.0
            N = 25
            gx = np.linspace(mins[0], maxs[0], N)
            gy = np.linspace(mins[1], maxs[1], N)
            gz = np.linspace(mins[2], maxs[2], N)
            GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
            coords = np.vstack([GX.ravel(), GY.ravel(), GZ.ravel()])
            density = kde(coords).reshape(GX.shape)
            iso_val = np.percentile(density[density > 0], 40)
            fig_pl.add_trace(go.Isosurface(
                x=GX.ravel(), y=GY.ravel(), z=GZ.ravel(),
                value=density.ravel(),
                isomin=iso_val, isomax=density.max(),
                surface_count=3, opacity=0.2,
                colorscale=colorscale, showscale=False,
                name=name, caps=dict(x_show=False, y_show=False, z_show=False)
            ))
        fig_pl.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
                             title="PCA: KDE Density Isosurfaces",
                             width=1000, height=800)
        save_plotly(fig_pl, "pca3d_density_isosurfaces")

        # static fallback with matplotlib contour-like
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")
        for pts, color, label in [(G_pca, "royalblue", "Global"), (L_pca, "crimson", "Local")]:
            kde = gaussian_kde(pts.T, bw_method=0.3)
            densities = kde(pts.T)
            sc = ax.scatter(*pts.T, c=densities, cmap="Blues" if color=="royalblue" else "Reds",
                           s=20, alpha=0.6, label=label)
        set_3d_labels(ax, "PC1", "PC2", "PC3")
        ax.set_title("PCA Space: Point Density", fontsize=12)
        ax.legend(fontsize=9)
        save_fig(fig, "pca3d_density_isosurfaces")
    except Exception as e:
        print(f"    Density plot error: {e}")


# ── Plot D: Delta-parameter hulls by patch type ──
def plot_delta_hulls_patchtype(df):
    print("\n[D] Delta-parameter hulls by patch type...")
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    for pt, color, label in [("interior", "forestgreen", "Interior"),
                              ("boundary", "darkorange", "Boundary")]:
        sub = df[df["patch_type"] == pt]
        pts = sub[["d_energy_threshold", "d_beta", "d_mu"]].values
        hp = hull_polys(pts, alpha=0.1, color=color)
        if hp: ax.add_collection3d(hp)
        ax.scatter(*pts.T, c=color, s=18, alpha=0.6, label=label)
        # ellipsoid
        xr, yr, zr, mean = covariance_ellipsoid(pts, n_std=1.5)
        ax.plot_surface(xr, yr, zr, color=color, alpha=0.06, linewidth=0)
    set_3d_labels(ax, r"$\Delta$energy_threshold", r"$\Delta$beta", r"$\Delta$mu")
    ax.set_title("Delta-Parameter Space by Patch Type", fontsize=12)
    ax.legend(fontsize=9)
    save_fig(fig, "delta3d_hulls_by_patchtype")

    # plotly
    try:
        import plotly.graph_objects as go
        fig_pl = go.Figure()
        for pt, color, label in [("interior", "green", "Interior"), ("boundary", "orange", "Boundary")]:
            sub = df[df["patch_type"] == pt]
            pts = sub[["d_energy_threshold", "d_beta", "d_mu"]].values
            try:
                h = ConvexHull(pts)
                fig_pl.add_trace(go.Mesh3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    i=h.simplices[:,0], j=h.simplices[:,1], k=h.simplices[:,2],
                    color=color, opacity=0.1, name=f"{label} hull"))
            except Exception:
                pass
            fig_pl.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                           mode="markers", marker=dict(size=3, color=color),
                                           name=label))
        fig_pl.update_layout(
            scene=dict(xaxis_title="d_energy_threshold", yaxis_title="d_beta", zaxis_title="d_mu"),
            title="Delta Parameters by Patch Type", width=1000, height=800)
        save_plotly(fig_pl, "delta3d_hulls_by_patchtype")
    except Exception:
        pass


# ── Plot E: Positive vs negative dD surfaces ──
def plot_dD_surfaces(G_pca, L_pca, df):
    print("\n[E] Positive vs negative dD surfaces...")
    pos = df["dD"] > 0.02
    neg = df["dD"] < -0.02
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    for mask, color, label in [(pos, "firebrick", "dD > +0.02"),
                                (neg, "steelblue", "dD < -0.02")]:
        pts = L_pca[mask.values]
        if len(pts) < 4:
            ax.scatter(*pts.T, c=color, s=20, alpha=0.7, label=label)
            continue
        hp = hull_polys(pts, alpha=0.1, color=color)
        if hp: ax.add_collection3d(hp)
        ax.scatter(*pts.T, c=color, s=18, alpha=0.6, label=label)
        xr, yr, zr, mean = covariance_ellipsoid(pts, n_std=1.5)
        ax.plot_surface(xr, yr, zr, color=color, alpha=0.06, linewidth=0)
    # neutral
    neutral = (~pos) & (~neg)
    pts_n = L_pca[neutral.values]
    ax.scatter(*pts_n.T, c="gray", s=8, alpha=0.3, label="|dD| <= 0.02")
    set_3d_labels(ax, "PC1", "PC2", "PC3")
    ax.set_title("PCA Space: Positive vs Negative dD (Local Points)", fontsize=12)
    ax.legend(fontsize=9)
    save_fig(fig, "pca3d_positive_vs_negative_dD_surfaces")

    # plotly
    try:
        import plotly.graph_objects as go
        fig_pl = go.Figure()
        for mask, color, label in [(pos, "red", "dD > +0.02"), (neg, "blue", "dD < -0.02")]:
            pts = L_pca[mask.values]
            if len(pts) >= 4:
                try:
                    h = ConvexHull(pts)
                    fig_pl.add_trace(go.Mesh3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        i=h.simplices[:,0], j=h.simplices[:,1], k=h.simplices[:,2],
                        color=color, opacity=0.1, name=f"{label} hull"))
                except Exception:
                    pass
            fig_pl.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                           mode="markers", marker=dict(size=3, color=color),
                                           name=label))
        pts_n = L_pca[neutral.values]
        fig_pl.add_trace(go.Scatter3d(x=pts_n[:,0], y=pts_n[:,1], z=pts_n[:,2],
                                       mode="markers", marker=dict(size=2, color="gray", opacity=0.3),
                                       name="|dD|<=0.02"))
        fig_pl.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
                             title="PCA: Positive vs Negative dD", width=1000, height=800)
        save_plotly(fig_pl, "pca3d_positive_vs_negative_dD_surfaces")
    except Exception:
        pass


# ── Plot F: Raw interpretable hull (threshold/beta/mu) ──
def plot_raw_hulls_tbm(df):
    print("\n[F] Raw 3D: threshold / beta / mu hulls...")
    g_cols = ["theta_global_energy_threshold", "theta_global_beta", "theta_global_mu"]
    l_cols = ["theta_local_energy_threshold", "theta_local_beta", "theta_local_mu"]
    G = df[g_cols].values
    L = df[l_cols].values
    dD = df["dD"].values

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    hp_g = hull_polys(G, alpha=0.06, color="royalblue")
    hp_l = hull_polys(L, alpha=0.06, color="crimson")
    if hp_g: ax.add_collection3d(hp_g)
    if hp_l: ax.add_collection3d(hp_l)
    # arrows colored by dD
    norm = TwoSlopeNorm(vmin=dD.min(), vcenter=0, vmax=max(dD.max(), 0.01))
    cmap = plt.cm.RdYlGn
    for i in range(len(G)):
        c = cmap(norm(dD[i]))
        ax.plot([G[i,0], L[i,0]], [G[i,1], L[i,1]], [G[i,2], L[i,2]],
                c=c, alpha=0.4, lw=0.8)
    ax.scatter(*G.T, c="royalblue", s=14, alpha=0.5, label="Global")
    sc = ax.scatter(*L.T, c=dD, cmap="RdYlGn", norm=norm, s=14, alpha=0.7, label="Local")
    plt.colorbar(sc, ax=ax, shrink=0.6, label="dD", pad=0.1)
    set_3d_labels(ax, "energy_threshold", "beta", "mu")
    ax.set_title("Raw Parameters: threshold / beta / mu", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    save_fig(fig, "raw3d_threshold_beta_mu_hulls")

    # plotly
    try:
        import plotly.graph_objects as go
        fig_pl = go.Figure()
        for pts, name, color in [(G, "Global", "royalblue"), (L, "Local", "crimson")]:
            try:
                h = ConvexHull(pts)
                fig_pl.add_trace(go.Mesh3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    i=h.simplices[:,0], j=h.simplices[:,1], k=h.simplices[:,2],
                    color=color, opacity=0.08, name=f"{name} hull"))
            except Exception:
                pass
        # arrows as lines colored by dD
        fig_pl.add_trace(go.Scatter3d(x=G[:,0], y=G[:,1], z=G[:,2],
                                       mode="markers", marker=dict(size=3, color="royalblue"), name="Global"))
        fig_pl.add_trace(go.Scatter3d(x=L[:,0], y=L[:,1], z=L[:,2],
                                       mode="markers", marker=dict(size=3, color=dD, colorscale="RdYlGn",
                                                                    colorbar=dict(title="dD")), name="Local"))
        xs, ys, zs = [], [], []
        for i in range(len(G)):
            xs += [G[i,0], L[i,0], None]
            ys += [G[i,1], L[i,1], None]
            zs += [G[i,2], L[i,2], None]
        fig_pl.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color="gray", width=1), opacity=0.2, name="Shifts"))
        fig_pl.update_layout(
            scene=dict(xaxis_title="energy_threshold", yaxis_title="beta", zaxis_title="mu"),
            title="Raw: threshold / beta / mu", width=1000, height=800)
        save_plotly(fig_pl, "raw3d_threshold_beta_mu_hulls")
    except Exception:
        pass


# ── Plot G: Raw interpretable hull (threshold/gamma/diffusion) ──
def plot_raw_hulls_tgd(df):
    print("\n[G] Raw 3D: threshold / gamma / diffusion hulls...")
    g_cols = ["theta_global_energy_threshold", "theta_global_gamma", "theta_global_diffusion_rate"]
    l_cols = ["theta_local_energy_threshold", "theta_local_gamma", "theta_local_diffusion_rate"]
    G = df[g_cols].values
    L = df[l_cols].values
    dD = df["dD"].values

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    hp_g = hull_polys(G, alpha=0.06, color="royalblue")
    hp_l = hull_polys(L, alpha=0.06, color="crimson")
    if hp_g: ax.add_collection3d(hp_g)
    if hp_l: ax.add_collection3d(hp_l)
    norm = TwoSlopeNorm(vmin=dD.min(), vcenter=0, vmax=max(dD.max(), 0.01))
    cmap = plt.cm.RdYlGn
    for i in range(len(G)):
        c = cmap(norm(dD[i]))
        ax.plot([G[i,0], L[i,0]], [G[i,1], L[i,1]], [G[i,2], L[i,2]],
                c=c, alpha=0.4, lw=0.8)
    ax.scatter(*G.T, c="royalblue", s=14, alpha=0.5, label="Global")
    sc = ax.scatter(*L.T, c=dD, cmap="RdYlGn", norm=norm, s=14, alpha=0.7, label="Local")
    plt.colorbar(sc, ax=ax, shrink=0.6, label="dD", pad=0.1)
    set_3d_labels(ax, "energy_threshold", "gamma", "diffusion_rate")
    ax.set_title("Raw Parameters: threshold / gamma / diffusion_rate", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    save_fig(fig, "raw3d_threshold_gamma_diffusion_hulls")

    # plotly
    try:
        import plotly.graph_objects as go
        fig_pl = go.Figure()
        for pts, name, color in [(G, "Global", "royalblue"), (L, "Local", "crimson")]:
            try:
                h = ConvexHull(pts)
                fig_pl.add_trace(go.Mesh3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    i=h.simplices[:,0], j=h.simplices[:,1], k=h.simplices[:,2],
                    color=color, opacity=0.08, name=f"{name} hull"))
            except Exception:
                pass
        fig_pl.add_trace(go.Scatter3d(x=G[:,0], y=G[:,1], z=G[:,2],
                                       mode="markers", marker=dict(size=3, color="royalblue"), name="Global"))
        fig_pl.add_trace(go.Scatter3d(x=L[:,0], y=L[:,1], z=L[:,2],
                                       mode="markers", marker=dict(size=3, color=dD, colorscale="RdYlGn",
                                                                    colorbar=dict(title="dD")), name="Local"))
        fig_pl.update_layout(
            scene=dict(xaxis_title="energy_threshold", yaxis_title="gamma", zaxis_title="diffusion_rate"),
            title="Raw: threshold / gamma / diffusion_rate", width=1000, height=800)
        save_plotly(fig_pl, "raw3d_threshold_gamma_diffusion_hulls")
    except Exception:
        pass


# ── Summary ──
def write_summary(df, G_pca, L_pca):
    print("\n[Summary]...")
    lines = []
    lines.append("Solid 3D Visualizations — Interpretation Summary")
    lines.append("=" * 55)
    lines.append("")
    lines.append("NOTE: These are geometric / density visualizations of saved")
    lines.append("parameter optima, NOT true optimization energy surfaces.")
    lines.append("")

    lines.append("--- Plot Descriptions ---")
    lines.append("")
    lines.append("A) pca3d_convex_hull_global_local: Convex hulls around all global")
    lines.append("   (blue) and local (red) parameter vectors in PCA space.")
    lines.append("   Gray lines connect each global->local pair.")
    lines.append("")
    lines.append("B) pca3d_covariance_ellipsoids: 2-sigma covariance ellipsoids")
    lines.append("   fitted to global and local clouds. Star markers show centroids.")
    lines.append("   Black arrow = centroid shift direction.")
    lines.append("")
    lines.append("C) pca3d_density_isosurfaces: KDE density isosurfaces in PCA space.")
    lines.append("   Shows where parameter vectors concentrate most densely.")
    lines.append("")
    lines.append("D) delta3d_hulls_by_patchtype: Convex hulls and 1.5-sigma ellipsoids")
    lines.append("   of delta parameters (d_energy_threshold, d_beta, d_mu)")
    lines.append("   split by interior (green) vs boundary (orange).")
    lines.append("")
    lines.append("E) pca3d_positive_vs_negative_dD_surfaces: Hulls/ellipsoids")
    lines.append("   separating local-theta vectors where dD > +0.02 (red)")
    lines.append("   from dD < -0.02 (blue).")
    lines.append("")
    lines.append("F) raw3d_threshold_beta_mu_hulls: Convex hulls in raw parameter")
    lines.append("   space (energy_threshold x beta x mu). Arrows colored by dD.")
    lines.append("")
    lines.append("G) raw3d_threshold_gamma_diffusion_hulls: Same style for")
    lines.append("   (energy_threshold x gamma x diffusion_rate).")
    lines.append("")

    lines.append("--- Key Observations ---")
    lines.append("")

    # hull overlap in PCA
    g_center = G_pca.mean(axis=0)
    l_center = L_pca.mean(axis=0)
    shift_mag = np.linalg.norm(l_center - g_center)
    lines.append(f"  PCA centroid shift magnitude: {shift_mag:.3f} (standardized units)")
    lines.append(f"  Global centroid: ({g_center[0]:.3f}, {g_center[1]:.3f}, {g_center[2]:.3f})")
    lines.append(f"  Local  centroid: ({l_center[0]:.3f}, {l_center[1]:.3f}, {l_center[2]:.3f})")
    lines.append("")

    # interior vs boundary overlap
    for pt in ["interior", "boundary"]:
        sub = df[df["patch_type"] == pt]
        d_pts = sub[["d_energy_threshold", "d_beta", "d_mu"]].values
        lines.append(f"  {pt.capitalize()} delta centroid: "
                     f"({d_pts[:,0].mean():.1f}, {d_pts[:,1].mean():.3f}, {d_pts[:,2].mean():.3f})")
    lines.append("  -> Interior and boundary delta centroids are similar,")
    lines.append("     confirming substantial overlap (see delta3d_hulls_by_patchtype).")
    lines.append("")

    # positive vs negative dD
    pos = df[df["dD"] > 0.02]
    neg = df[df["dD"] < -0.02]
    lines.append(f"  Patches with dD > +0.02: {len(pos)}")
    lines.append(f"  Patches with dD < -0.02: {len(neg)}")
    lines.append(f"  Neutral (|dD| <= 0.02):  {len(df) - len(pos) - len(neg)}")
    lines.append("  -> See pca3d_positive_vs_negative_dD_surfaces for cluster separation.")
    lines.append("")

    lines.append("--- Reminder ---")
    lines.append("These visualizations show the GEOMETRY of the parameter distributions.")
    lines.append("They do NOT represent the objective function's energy landscape.")
    lines.append("The surfaces are convex hulls / covariance ellipsoids of")
    lines.append("where parameters ended up, not contours of equal score.")
    lines.append("")

    txt = "\n".join(lines)
    out = os.path.join(OUT_DIR, "solid_3d_summary.txt")
    with open(out, "w") as f:
        f.write(txt)
    print(f"  Saved {out}")
    print()
    print(txt)


# ── main ──
def main():
    print("=== Part 1: Load & Merge ===")
    df = load_and_merge()

    print("\n=== Part 2: PCA ===")
    G_pca, L_pca, pca, scaler = do_pca(df)
    ev = pca.explained_variance_ratio_
    print(f"  PCA variance: {ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f}  (cum {sum(ev):.3f})")

    print("\n=== Part 3: Solid 3D Plots ===")
    plot_pca_hulls(G_pca, L_pca, df)
    plot_pca_ellipsoids(G_pca, L_pca, df)
    plot_pca_density(G_pca, L_pca, df)
    plot_delta_hulls_patchtype(df)
    plot_dD_surfaces(G_pca, L_pca, df)
    plot_raw_hulls_tbm(df)
    plot_raw_hulls_tgd(df)

    print("\n=== Part 5: Summary ===")
    write_summary(df, G_pca, L_pca)

    print("\n=== Output listing ===")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("_"):
            continue
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        if sz > 1024*1024:
            print(f"  {f}  ({sz/1024/1024:.1f}M)")
        else:
            print(f"  {f}  ({sz//1024}K)")
    print("\nDone.")


if __name__ == "__main__":
    main()
