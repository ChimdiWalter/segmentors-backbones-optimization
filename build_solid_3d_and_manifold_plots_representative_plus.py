#!/usr/bin/env python3
"""Build solid 3D geometry plots, manifold plots, and combined reports for representative trace study."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

PARAMS = ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "energy_threshold"]
DPI = 240
PAIR_A = ("energy_threshold", "beta", "mu")
PAIR_B = ("energy_threshold", "gamma", "diffusion_rate")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="experiments/exp_v1/outputs/objective_surface_traces_representative_fullbudget_plus",
        help="Study root",
    )
    p.add_argument(
        "--opt-csv",
        default="experiments/exp_v1/outputs/opt_summary_local.csv",
        help="Global optimizer summary CSV",
    )
    return p.parse_args()


def write_pdf_from_text(text: str, out_pdf: Path):
    lines = []
    for ln in text.splitlines():
        if not ln.strip():
            lines.append("")
            continue
        lines.extend(wrap(ln, width=112, break_long_words=False, break_on_hyphens=False))

    with PdfPages(out_pdf) as pdf:
        page_size = 58
        page = 0
        for i in range(0, len(lines), page_size):
            page += 1
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.axis("off")
            ax.text(0, 1, "\n".join(lines[i:i + page_size]), va="top", ha="left", family="monospace", fontsize=9)
            ax.text(1, 0, f"Page {page}", va="bottom", ha="right", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


def load_inputs(root: Path, opt_csv: Path):
    reports = root / "reports"
    selected = pd.read_csv(reports / "selected_cases.txt", sep="\t")

    global_df = pd.read_csv(opt_csv)
    global_df.columns = [c.strip() for c in global_df.columns]
    global_df["filename"] = global_df["filename"].astype(str).str.strip()

    rerun_df = pd.read_csv(root / "rerun_outputs" / "opt_summary.csv")
    rerun_df.columns = [c.strip() for c in rerun_df.columns]
    rerun_df["filename"] = rerun_df["filename"].astype(str).str.strip()

    traces = []
    trace_dir = root / "candidate_traces"
    for _, r in selected.iterrows():
        filename = str(r["filename"]).strip()
        group = str(r["group"])
        case_id = Path(filename).stem
        p = trace_dir / f"{case_id}.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        for c in ["iteration", "candidate_index", "is_refine", "energy_threshold", "area_px", "n_contours"]:
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce")
        for c in ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "score", "best_score_so_far", "elapsed_seconds", "area_frac"]:
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce")
        t["filename"] = filename
        t["case_id"] = case_id
        t["group"] = group
        traces.append(t)

    trace_df = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    return selected, global_df, rerun_df, trace_df


def build_final_parameter_summary(selected: pd.DataFrame, global_df: pd.DataFrame, rerun_df: pd.DataFrame, out_csv: Path):
    rows = []
    for _, s in selected.iterrows():
        fn = str(s["filename"]).strip()
        group = str(s["group"])
        g = global_df[global_df["filename"] == fn]
        r = rerun_df[rerun_df["filename"] == fn]
        if len(g) == 0 or len(r) == 0:
            continue
        g0 = g.iloc[0]
        r0 = r.iloc[0]
        row = {
            "filename": fn,
            "case_id": Path(fn).stem,
            "group": group,
            "global_score": float(g0["score"]),
            "rerun_score": float(r0["score"]),
            "score_diff_rerun_minus_global": float(r0["score"] - g0["score"]),
            "global_elapsed_seconds": float(g0.get("elapsed_seconds", np.nan)),
            "rerun_elapsed_seconds": float(r0.get("elapsed_seconds", np.nan)),
        }
        for p in PARAMS:
            row[f"global_{p}"] = float(g0[p])
            row[f"rerun_{p}"] = float(r0[p])
            row[f"d_{p}"] = float(r0[p] - g0[p])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["group", "filename"]) if rows else pd.DataFrame()
    out.to_csv(out_csv, index=False)
    return out


def build_candidate_trace_summary(selected: pd.DataFrame, trace_df: pd.DataFrame, out_csv: Path):
    rows = []
    for _, s in selected.iterrows():
        fn = str(s["filename"]).strip()
        case_id = Path(fn).stem
        g = str(s["group"])
        c = trace_df[trace_df["case_id"] == case_id].copy()
        if len(c) == 0:
            rows.append({"case_id": case_id, "filename": fn, "group": g, "missing_trace": True})
            continue
        rand = c[c["phase"] == "random"]
        refi = c[c["phase"] == "refine"]
        rows.append({
            "case_id": case_id,
            "filename": fn,
            "group": g,
            "missing_trace": False,
            "n_candidates": int(len(c)),
            "n_random": int(len(rand)),
            "n_refine": int(len(refi)),
            "best_score": float(c["score"].max()),
            "best_random_score": float(rand["score"].max()) if len(rand) else np.nan,
            "best_refine_score": float(refi["score"].max()) if len(refi) else np.nan,
            "runtime_seconds_trace": float(c["elapsed_seconds"].max()) if "elapsed_seconds" in c.columns else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def _hull_poly(points: np.ndarray, color: str, alpha: float):
    if len(points) < 4:
        return None
    try:
        h = ConvexHull(points)
    except Exception:
        return None
    verts = [[points[s] for s in simplex] for simplex in h.simplices]
    return Poly3DCollection(verts, facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.2)


def _cov_ellipsoid(points: np.ndarray, n_std=2.0, n=28):
    mean = points.mean(axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    radii = n_std * np.sqrt(eigvals)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()])
    xyz = eigvecs @ xyz + mean[:, None]
    return xyz[0].reshape(x.shape), xyz[1].reshape(y.shape), xyz[2].reshape(z.shape), mean


def save_fig(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def solid_plot_convex_hulls(emb: pd.DataFrame, outdir: Path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    groups = [
        ("global", "tab:blue", "Global theta"),
        ("final", "tab:red", "Rerun final theta"),
        ("candidate", "0.6", "Candidates"),
    ]
    for kind, color, label in groups:
        pts = emb[emb["point_kind"] == kind][["pca1", "pca2", "pca3"]].to_numpy(dtype=float)
        if len(pts) == 0:
            continue
        poly = _hull_poly(pts, color, 0.07 if kind != "candidate" else 0.03)
        if poly is not None:
            ax.add_collection3d(poly)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=18 if kind != "candidate" else 8, alpha=0.65 if kind != "candidate" else 0.25, label=label)

    gl = emb[emb["point_kind"] == "global"].set_index("case_id")
    fn = emb[emb["point_kind"] == "final"].set_index("case_id")
    for case_id in sorted(set(gl.index).intersection(set(fn.index))):
        g = gl.loc[case_id]
        f = fn.loc[case_id]
        ax.plot([g["pca1"], f["pca1"]], [g["pca2"], f["pca2"]], [g["pca3"], f["pca3"]], c="0.5", lw=0.8, alpha=0.5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D Convex Hulls: global vs rerun-final vs candidates")
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, outdir / "pca3d_convex_hulls_global_final_candidates")

    if HAS_PLOTLY:
        figp = go.Figure()
        for kind, color, label in groups:
            pts = emb[emb["point_kind"] == kind][["pca1", "pca2", "pca3"]].to_numpy(dtype=float)
            if len(pts) == 0:
                continue
            if len(pts) >= 4:
                try:
                    h = ConvexHull(pts)
                    figp.add_trace(go.Mesh3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        i=h.simplices[:, 0], j=h.simplices[:, 1], k=h.simplices[:, 2],
                        color=color, opacity=0.1 if kind != "candidate" else 0.04,
                        name=f"{label} hull",
                    ))
                except Exception:
                    pass
            figp.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", marker=dict(size=3, color=color), name=label))
        figp.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"), width=1000, height=800)
        figp.write_html(str((outdir / "pca3d_convex_hulls_global_final_candidates.html")))


def solid_plot_cov_ellipsoids(emb: pd.DataFrame, outdir: Path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    specs = [("global", "tab:blue"), ("final", "tab:red")]
    means = {}
    for kind, color in specs:
        pts = emb[emb["point_kind"] == kind][["pca1", "pca2", "pca3"]].to_numpy(dtype=float)
        if len(pts) < 4:
            continue
        x, y, z, m = _cov_ellipsoid(pts, n_std=2.0)
        means[kind] = m
        ax.plot_surface(x, y, z, color=color, alpha=0.13, linewidth=0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=20, alpha=0.6, label=kind)
        ax.scatter([m[0]], [m[1]], [m[2]], c=color, s=100, marker="*", edgecolors="k")
    if "global" in means and "final" in means:
        gm, fm = means["global"], means["final"]
        ax.quiver(gm[0], gm[1], gm[2], fm[0] - gm[0], fm[1] - gm[1], fm[2] - gm[2], color="k", linewidth=2.0, arrow_length_ratio=0.12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA covariance ellipsoids (2-sigma): global vs rerun-final")
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, outdir / "pca3d_covariance_ellipsoids_global_vs_final")


def solid_plot_density_isosurfaces(emb: pd.DataFrame, outdir: Path):
    pts_g = emb[emb["point_kind"] == "global"][["pca1", "pca2", "pca3"]].to_numpy(dtype=float)
    pts_f = emb[emb["point_kind"] == "final"][["pca1", "pca2", "pca3"]].to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for pts, cmap, label in [(pts_g, "Blues", "Global"), (pts_f, "Reds", "Final")]:
        if len(pts) < 4:
            continue
        kde = gaussian_kde(pts.T)
        d = kde(pts.T)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=d, cmap=cmap, s=35, alpha=0.8, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA density proxy (pointwise KDE): global and rerun-final")
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, outdir / "pca3d_density_isosurfaces_global_final")

    if HAS_PLOTLY and len(pts_g) >= 6 and len(pts_f) >= 6:
        mins = np.minimum(pts_g.min(axis=0), pts_f.min(axis=0)) - 1.0
        maxs = np.maximum(pts_g.max(axis=0), pts_f.max(axis=0)) + 1.0
        gx = np.linspace(mins[0], maxs[0], 24)
        gy = np.linspace(mins[1], maxs[1], 24)
        gz = np.linspace(mins[2], maxs[2], 24)
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        coords = np.vstack([GX.ravel(), GY.ravel(), GZ.ravel()])

        figp = go.Figure()
        for pts, colorscale, name in [(pts_g, "Blues", "Global"), (pts_f, "Reds", "Final")]:
            kde = gaussian_kde(pts.T)
            dens = kde(coords).reshape(GX.shape)
            pos = dens[dens > 0]
            if len(pos) == 0:
                continue
            lo = float(np.percentile(pos, 45))
            figp.add_trace(go.Isosurface(
                x=GX.ravel(), y=GY.ravel(), z=GZ.ravel(),
                value=dens.ravel(),
                isomin=lo, isomax=float(dens.max()),
                surface_count=2,
                colorscale=colorscale,
                opacity=0.2,
                showscale=False,
                name=name,
                caps=dict(x_show=False, y_show=False, z_show=False),
            ))
        figp.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"), width=1000, height=800)
        figp.write_html(str(outdir / "pca3d_density_isosurfaces_global_final.html"))


def solid_plot_positive_negative_surfaces(emb: pd.DataFrame, final_summary: pd.DataFrame, outdir: Path):
    score_map = final_summary.set_index("case_id")["score_diff_rerun_minus_global"].to_dict()
    cand = emb[emb["point_kind"] == "candidate"].copy()
    cand["leaf_score_diff"] = cand["case_id"].map(score_map)

    pos = cand[cand["leaf_score_diff"] > 0.0]
    neg = cand[cand["leaf_score_diff"] < 0.0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for sub, color, label in [(pos, "tab:green", "leaf score diff > 0"), (neg, "tab:purple", "leaf score diff < 0")]:
        pts = sub[["pca1", "pca2", "pca3"]].to_numpy(dtype=float)
        if len(pts) == 0:
            continue
        poly = _hull_poly(pts, color, 0.07)
        if poly is not None:
            ax.add_collection3d(poly)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=8, alpha=0.35, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Candidate-cloud solid geometry: positive vs negative leaf score change")
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, outdir / "pca3d_positive_vs_negative_leaf_score_surfaces")


def solid_plot_raw_hulls(final_summary: pd.DataFrame, outdir: Path, xyz: tuple[str, str, str], stem: str):
    x, y, z = xyz
    g = final_summary[[f"global_{x}", f"global_{y}", f"global_{z}"]].to_numpy(dtype=float)
    f = final_summary[[f"rerun_{x}", f"rerun_{y}", f"rerun_{z}"]].to_numpy(dtype=float)
    dscore = final_summary["score_diff_rerun_minus_global"].to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=float(dscore.min()), vcenter=0.0, vmax=float(dscore.max()))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    pg = _hull_poly(g, "tab:blue", 0.08)
    pf = _hull_poly(f, "tab:red", 0.08)
    if pg is not None:
        ax.add_collection3d(pg)
    if pf is not None:
        ax.add_collection3d(pf)

    for i in range(len(g)):
        c = plt.cm.RdYlGn(norm(dscore[i]))
        ax.plot([g[i, 0], f[i, 0]], [g[i, 1], f[i, 1]], [g[i, 2], f[i, 2]], c=c, lw=1.1, alpha=0.65)

    ax.scatter(g[:, 0], g[:, 1], g[:, 2], c="tab:blue", s=24, alpha=0.7, label="global theta")
    sc = ax.scatter(f[:, 0], f[:, 1], f[:, 2], c=dscore, cmap="RdYlGn", norm=norm, s=28, alpha=0.85, label="rerun final theta")
    plt.colorbar(sc, ax=ax, shrink=0.62, pad=0.09, label="rerun score - global score")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f"Raw 3D hulls: {x} / {y} / {z}")
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, outdir / stem)


def make_manifold_embeddings(final_summary: pd.DataFrame, trace_df: pd.DataFrame):
    rows = []
    for _, r in final_summary.iterrows():
        base = {
            "filename": r["filename"],
            "case_id": r["case_id"],
            "group": r["group"],
            "leaf_score_diff": r["score_diff_rerun_minus_global"],
        }
        gr = base | {"point_kind": "global", "phase": "global", "score": r["global_score"]}
        fr = base | {"point_kind": "final", "phase": "final", "score": r["rerun_score"]}
        for p in PARAMS:
            gr[p] = r[f"global_{p}"]
            fr[p] = r[f"rerun_{p}"]
        rows.append(gr)
        rows.append(fr)

    if len(trace_df):
        c = trace_df[["filename", "case_id", "group", "phase", "score", *PARAMS]].copy()
        c["point_kind"] = "candidate"
        c["leaf_score_diff"] = c["case_id"].map(final_summary.set_index("case_id")["score_diff_rerun_minus_global"].to_dict())
        rows.extend(c.to_dict(orient="records"))

    emb = pd.DataFrame(rows)
    emb = emb.dropna(subset=PARAMS).reset_index(drop=True)

    X = emb[PARAMS].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=3, random_state=1337)
    pca_xyz = pca.fit_transform(Xs)
    emb["pca1"] = pca_xyz[:, 0]
    emb["pca2"] = pca_xyz[:, 1]
    emb["pca3"] = pca_xyz[:, 2]

    perplexity = max(5, min(35, (len(emb) - 1) // 3))
    tsne = TSNE(n_components=2, random_state=1337, perplexity=perplexity, init="pca", learning_rate="auto")
    ts = tsne.fit_transform(Xs)
    emb["tsne1"] = ts[:, 0]
    emb["tsne2"] = ts[:, 1]

    umap_info = {"available": HAS_UMAP, "generated_2d": False, "generated_3d": False}
    if HAS_UMAP and len(emb) >= 12:
        try:
            u2 = umap.UMAP(n_components=2, random_state=1337).fit_transform(Xs)
            emb["umap2_1"] = u2[:, 0]
            emb["umap2_2"] = u2[:, 1]
            umap_info["generated_2d"] = True
            if len(emb) >= 18:
                u3 = umap.UMAP(n_components=3, random_state=1337).fit_transform(Xs)
                emb["umap3_1"] = u3[:, 0]
                emb["umap3_2"] = u3[:, 1]
                emb["umap3_3"] = u3[:, 2]
                umap_info["generated_3d"] = True
        except Exception:
            pass

    return emb, pca.explained_variance_ratio_, umap_info


def scatter2d(df: pd.DataFrame, x: str, y: str, hue: str, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    if hue == "score":
        sc = ax.scatter(df[x], df[y], c=df["score"], cmap="viridis", s=20, alpha=0.75)
        plt.colorbar(sc, ax=ax, label="score")
    else:
        cats = list(pd.Series(df[hue]).fillna("NA").astype(str).unique())
        cmap = plt.cm.tab10(np.linspace(0, 1, max(3, len(cats))))
        for i, c in enumerate(cats):
            sub = df[df[hue].astype(str) == c]
            ax.scatter(sub[x], sub[y], c=[cmap[i]], s=22, alpha=0.75, label=c)
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    save_fig(fig, out)


def scatter3d(df: pd.DataFrame, x: str, y: str, z: str, hue: str, out: Path, title: str):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if hue == "score":
        sc = ax.scatter(df[x], df[y], df[z], c=df["score"], cmap="viridis", s=14, alpha=0.8)
        plt.colorbar(sc, ax=ax, shrink=0.62, pad=0.08, label="score")
    else:
        cats = list(pd.Series(df[hue]).fillna("NA").astype(str).unique())
        cmap = plt.cm.tab10(np.linspace(0, 1, max(3, len(cats))))
        for i, c in enumerate(cats):
            sub = df[df[hue].astype(str) == c]
            ax.scatter(sub[x], sub[y], sub[z], c=[cmap[i]], s=16, alpha=0.75, label=c)
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(title)
    save_fig(fig, out)


def build_manifold_plots(emb: pd.DataFrame, outdir: Path, umap_info: dict):
    scatter2d(emb, "pca1", "pca2", "point_kind", outdir / "pca_2d_by_point_kind", "PCA 2D by point kind")
    scatter2d(emb, "pca1", "pca2", "group", outdir / "pca_2d_by_leaf_group", "PCA 2D by leaf group")
    scatter2d(emb, "pca1", "pca2", "score", outdir / "pca_2d_by_score", "PCA 2D colored by score")
    scatter3d(emb, "pca1", "pca2", "pca3", "point_kind", outdir / "pca_3d_by_point_kind", "PCA 3D by point kind")

    scatter2d(emb, "tsne1", "tsne2", "phase", outdir / "tsne_2d_by_phase", "t-SNE 2D by phase")
    scatter2d(emb, "tsne1", "tsne2", "group", outdir / "tsne_2d_by_leaf_group", "t-SNE 2D by leaf group")

    if umap_info.get("generated_2d", False):
        scatter2d(emb, "umap2_1", "umap2_2", "phase", outdir / "umap_2d_by_phase", "UMAP 2D by phase")
        scatter2d(emb, "umap2_1", "umap2_2", "point_kind", outdir / "umap_2d_by_point_kind", "UMAP 2D by point kind")
    if umap_info.get("generated_3d", False):
        scatter3d(emb, "umap3_1", "umap3_2", "umap3_3", "point_kind", outdir / "umap_3d_by_point_kind", "UMAP 3D by point kind")

    if HAS_PLOTLY:
        figp = go.Figure()
        for pk, color in [("candidate", "gray"), ("global", "blue"), ("final", "red")]:
            sub = emb[emb["point_kind"] == pk]
            if len(sub) == 0:
                continue
            figp.add_trace(go.Scatter3d(
                x=sub["pca1"], y=sub["pca2"], z=sub["pca3"],
                mode="markers",
                marker=dict(size=2 if pk == "candidate" else 4, color=color, opacity=0.45 if pk == "candidate" else 0.9),
                name=pk,
            ))
        figp.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"), width=1000, height=800)
        figp.write_html(str(outdir / "pca_3d_interactive.html"))


def build_solid_report(final_summary: pd.DataFrame, emb: pd.DataFrame, out_txt: Path, out_pdf: Path):
    g = emb[emb["point_kind"] == "global"][["pca1", "pca2", "pca3"]].to_numpy(dtype=float)
    f = emb[emb["point_kind"] == "final"][["pca1", "pca2", "pca3"]].to_numpy(dtype=float)
    c = emb[emb["point_kind"] == "candidate"][["pca1", "pca2", "pca3"]].to_numpy(dtype=float)
    shift = np.linalg.norm(f.mean(axis=0) - g.mean(axis=0)) if len(g) and len(f) else np.nan

    lines = [
        "Solid 3D Geometry Report",
        "========================",
        "",
        "These figures are geometric analyses derived from traced candidates and final optima.",
        "They are not direct objective-energy surfaces.",
        "",
        f"Selected leaves: {len(final_summary)}",
        f"Candidate points used in geometry cloud: {len(c)}",
        f"PCA centroid shift (global -> rerun final): {shift:.4f}" if not np.isnan(shift) else "PCA centroid shift unavailable",
        "",
        "Key interpretation:",
        "- Convex hull overlap indicates the degree of shared parameter support between global and rerun-final optima.",
        "- Covariance ellipsoids summarize spread and centroid shift direction.",
        "- Density isosurfaces approximate where parameter mass concentrates in PCA space.",
        "- Positive-vs-negative score-change solids indicate whether improved leaves occupy different regions.",
        "",
        "Raw-coordinate hull plots:",
        "- threshold / beta / mu",
        "- threshold / gamma / diffusion_rate",
        "These preserve interpretability of dominant physical parameters.",
    ]

    out_txt.write_text("\n".join(lines) + "\n")
    write_pdf_from_text("\n".join(lines), out_pdf)


def build_manifold_report(emb: pd.DataFrame, pca_var: np.ndarray, umap_info: dict, out_txt: Path, out_pdf: Path):
    n_by_kind = emb["point_kind"].value_counts().to_dict()
    n_by_phase = emb["phase"].value_counts().to_dict()

    lines = [
        "Manifold Report",
        "===============",
        "",
        "Embeddings are computed from 7-D parameter vectors (global, rerun-final, and traced candidates).",
        f"Point counts by kind: {n_by_kind}",
        f"Point counts by phase: {n_by_phase}",
        f"PCA explained variance ratio: {[round(x, 4) for x in pca_var.tolist()]}",
        "",
        "Generated plots:",
        "- PCA 2D and PCA 3D",
        "- t-SNE 2D",
        "- UMAP 2D/3D when library availability allowed",
        "",
    ]
    if umap_info.get("generated_2d", False):
        lines.append("UMAP status: generated 2D embedding.")
    else:
        lines.append("UMAP status: not generated (library unavailable or insufficient support).")
    if umap_info.get("generated_3d", False):
        lines.append("UMAP 3D status: generated.")
    else:
        lines.append("UMAP 3D status: not generated.")

    lines.extend([
        "",
        "Interpretation guidance:",
        "- PCA reveals dominant linear organization of parameter shifts.",
        "- t-SNE highlights local neighborhood structure among sampled candidates.",
        "- If available, UMAP gives an alternative global/local topology view.",
    ])

    out_txt.write_text("\n".join(lines) + "\n")
    write_pdf_from_text("\n".join(lines), out_pdf)


def build_master_summary(
    root: Path,
    selected: pd.DataFrame,
    final_summary: pd.DataFrame,
    trace_summary: pd.DataFrame,
    surface_case_summary_path: Path,
    umap_info: dict,
):
    if surface_case_summary_path.exists():
        surf = pd.read_csv(surface_case_summary_path)
    else:
        surf = pd.DataFrame()

    total_runtime = float(trace_summary["runtime_seconds_trace"].sum()) if "runtime_seconds_trace" in trace_summary.columns else np.nan
    mean_runtime = float(trace_summary["runtime_seconds_trace"].mean()) if "runtime_seconds_trace" in trace_summary.columns else np.nan

    high = final_summary[final_summary["group"] == "high"]
    low = final_summary[final_summary["group"] == "low"]

    lines = [
        "Master Summary: Representative Full-Budget Plus Study",
        "====================================================",
        "",
        "This study reruns a representative subset with original optimizer settings while adding trace logging.",
        "Only number of leaves was reduced; optimization logic remained unchanged.",
        "",
        f"Selected leaves: {len(selected)}",
        "Selected cases:",
    ]
    for _, r in selected.iterrows():
        lines.append(f"- {r['group']}: {r['filename']} (baseline score={float(r['score']):.6f})")

    lines.extend([
        "",
        f"Runtime per leaf (trace max elapsed) mean: {mean_runtime:.2f} s" if not np.isnan(mean_runtime) else "Runtime per leaf unavailable",
        f"Total representative runtime (sum): {total_runtime:.2f} s" if not np.isnan(total_runtime) else "Total runtime unavailable",
        "",
        "Score-landscape interpretation (true trace-based):",
    ])

    if len(surf):
        gain = float(surf["improve_from_random"].mean()) if "improve_from_random" in surf.columns else np.nan
        sharp = surf["sharpness"].value_counts().to_dict() if "sharpness" in surf.columns else {}
        lines.append(f"- Mean refine gain from random best: {gain:+.4f}" if not np.isnan(gain) else "- Mean refine gain unavailable")
        lines.append(f"- Surface sharpness distribution: {sharp}")
        if "multi_basin_hint" in surf.columns:
            lines.append(f"- Multi-basin hints: {int(surf['multi_basin_hint'].sum())}/{len(surf)} cases")

    lines.extend([
        "",
        "Geometry/manifold interpretation:",
        "- Solid 3D plots characterize parameter-support geometry (hulls, ellipsoids, density).",
        "- Manifold plots characterize embedding organization of global/final/candidate parameter vectors.",
        f"- UMAP generated: 2D={umap_info.get('generated_2d', False)}, 3D={umap_info.get('generated_3d', False)}",
        "",
    ])

    if len(high) and len(low):
        lines.append(f"High-score leaves mean score diff (rerun-global): {high['score_diff_rerun_minus_global'].mean():+.4f}")
        lines.append(f"Low-score leaves mean score diff (rerun-global): {low['score_diff_rerun_minus_global'].mean():+.4f}")

    lines.extend([
        "",
        "Recommended figures for paper:",
        "- surface_plots/*energy_threshold_vs_beta* (trace-based score structure)",
        "- trajectory_plots/*trajectory_best_score_so_far* (refinement contribution)",
        "- solid_3d_plots/raw3d_threshold_beta_mu_hulls.* (interpretable global-vs-final shift)",
        "",
        "Recommended appendix figures:",
        "- manifold_plots/tSNE and UMAP families",
        "- solid_3d_plots/pca3d_density_isosurfaces_*",
        "",
        "Method distinction:",
        "- Score surfaces/trajectories are true trace-based reconstructions from logged candidate evaluations.",
        "- Solid 3D and manifold plots are geometric analyses derived from traced parameters and final optima.",
    ])

    reports = root / "reports"
    txt_path = reports / "master_summary.txt"
    pdf_path = reports / "master_summary.pdf"
    txt_path.write_text("\n".join(lines) + "\n")
    write_pdf_from_text("\n".join(lines), pdf_path)


def main():
    args = parse_args()
    root = Path(args.root)
    reports = root / "reports"
    solid_dir = root / "solid_3d_plots"
    mani_dir = root / "manifold_plots"
    reports.mkdir(parents=True, exist_ok=True)
    solid_dir.mkdir(parents=True, exist_ok=True)
    mani_dir.mkdir(parents=True, exist_ok=True)

    selected, global_df, rerun_df, trace_df = load_inputs(root, Path(args.opt_csv))

    final_summary = build_final_parameter_summary(
        selected=selected,
        global_df=global_df,
        rerun_df=rerun_df,
        out_csv=reports / "final_parameter_summary.csv",
    )

    trace_summary = build_candidate_trace_summary(
        selected=selected,
        trace_df=trace_df,
        out_csv=reports / "candidate_trace_summary.csv",
    )

    emb, pca_var, umap_info = make_manifold_embeddings(final_summary, trace_df)
    emb.to_csv(reports / "manifold_embedding_summary.csv", index=False)

    # solid 3D plots
    solid_plot_convex_hulls(emb, solid_dir)
    solid_plot_cov_ellipsoids(emb, solid_dir)
    solid_plot_density_isosurfaces(emb, solid_dir)
    solid_plot_positive_negative_surfaces(emb, final_summary, solid_dir)
    solid_plot_raw_hulls(final_summary, solid_dir, PAIR_A, "raw3d_threshold_beta_mu_hulls")
    solid_plot_raw_hulls(final_summary, solid_dir, PAIR_B, "raw3d_threshold_gamma_diffusion_hulls")

    # manifold plots
    build_manifold_plots(emb, mani_dir, umap_info)

    # reports
    build_solid_report(final_summary, emb, reports / "solid_3d_report.txt", reports / "solid_3d_report.pdf")
    build_manifold_report(emb, pca_var, umap_info, reports / "manifold_report.txt", reports / "manifold_report.pdf")
    build_master_summary(
        root=root,
        selected=selected,
        final_summary=final_summary,
        trace_summary=trace_summary,
        surface_case_summary_path=reports / "surface_case_summary.csv",
        umap_info=umap_info,
    )

    print(f"Wrote: {reports / 'candidate_trace_summary.csv'}")
    print(f"Wrote: {reports / 'final_parameter_summary.csv'}")
    print(f"Wrote: {reports / 'manifold_embedding_summary.csv'}")
    print(f"Wrote reports and plots under: {root}")


if __name__ == "__main__":
    main()
