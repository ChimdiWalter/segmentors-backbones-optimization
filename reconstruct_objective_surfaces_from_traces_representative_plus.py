#!/usr/bin/env python3
"""Reconstruct trace-based objective surfaces, trajectories, and density plots."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde


PAIR_SPECS = [
    ("energy_threshold", "beta"),
    ("energy_threshold", "mu"),
    ("beta", "gamma"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="experiments/exp_v1/outputs/objective_surface_traces_representative_fullbudget_plus",
        help="Output root for the representative-plus trace study",
    )
    return p.parse_args()


def load_selected_cases(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df


def read_trace_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["iteration", "candidate_index", "is_refine", "energy_threshold", "area_px", "n_contours"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "score", "best_score_so_far", "elapsed_seconds", "area_frac"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("candidate_index")


def make_surface_plot(df: pd.DataFrame, x: str, y: str, case_id: str, final_row: pd.Series | None, outdir: Path):
    z = df["score"].to_numpy()
    xv = df[x].to_numpy()
    yv = df[y].to_numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xv, yv, z, c=z, cmap="viridis", s=45, alpha=0.9)

    # Fit simple trisurface only when enough unique support exists.
    uniq = pd.DataFrame({x: xv, y: yv, "score": z}).groupby([x, y], as_index=False)["score"].mean()
    if len(uniq) >= 10 and uniq[x].nunique() >= 4 and uniq[y].nunique() >= 4:
        try:
            ax.plot_trisurf(uniq[x], uniq[y], uniq["score"], color="lightgray", alpha=0.35, linewidth=0.2)
        except Exception:
            pass

    rand = df[df["phase"] == "random"]
    refi = df[df["phase"] == "refine"]
    if len(rand):
        br = rand.loc[rand["score"].idxmax()]
        ax.scatter([br[x]], [br[y]], [br["score"]], marker="^", s=140, c="orange", edgecolors="black", label="Best random")
    if len(refi):
        bf = refi.loc[refi["score"].idxmax()]
        ax.scatter([bf[x]], [bf[y]], [bf["score"]], marker="s", s=120, c="cyan", edgecolors="black", label="Best refine")

    if final_row is not None:
        ax.scatter(
            [float(final_row[x])], [float(final_row[y])], [float(final_row["score"])],
            marker="*", s=220, c="red", edgecolors="black", label="Final optimum"
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel("score")
    ax.set_title(f"{case_id}: trace-based objective points ({x} vs {y})")
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08, label="score")
    ax.legend(loc="best", fontsize=8)

    stem = f"{case_id}__surface_{x}_vs_{y}"
    fig.savefig(outdir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_density_plot(df: pd.DataFrame, x: str, y: str, case_id: str, final_row: pd.Series | None, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = 18

    axes[0].hist2d(df[x], df[y], bins=bins, cmap="magma")
    axes[0].set_xlabel(x)
    axes[0].set_ylabel(y)
    axes[0].set_title("All candidate density")
    if final_row is not None:
        axes[0].plot(float(final_row[x]), float(final_row[y]), "w*", markersize=12, markeredgecolor="black")

    rand = df[df["phase"] == "random"]
    refi = df[df["phase"] == "refine"]
    axes[1].scatter(rand[x], rand[y], c="tab:blue", s=28, alpha=0.8, label="random")
    axes[1].scatter(refi[x], refi[y], c="tab:orange", s=35, alpha=0.8, label="refine")
    axes[1].set_xlabel(x)
    axes[1].set_ylabel(y)
    axes[1].set_title("Phase split")
    if len(df) >= 12:
        try:
            xy = np.vstack([df[x].to_numpy(), df[y].to_numpy()])
            kde = gaussian_kde(xy)
            gx = np.linspace(df[x].min(), df[x].max(), 60)
            gy = np.linspace(df[y].min(), df[y].max(), 60)
            GX, GY = np.meshgrid(gx, gy)
            ZZ = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
            axes[1].contour(GX, GY, ZZ, colors="k", linewidths=0.5, alpha=0.5)
        except Exception:
            pass
    if final_row is not None:
        axes[1].plot(float(final_row[x]), float(final_row[y]), "r*", markersize=12, markeredgecolor="black", label="final optimum")
    axes[1].legend(loc="best", fontsize=8)

    fig.suptitle(f"{case_id}: explored-region density ({x} vs {y})")
    fig.tight_layout()
    stem = f"{case_id}__density_{x}_vs_{y}"
    fig.savefig(outdir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_trajectory_plots(df: pd.DataFrame, case_id: str, outdir: Path):
    # score vs iteration
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df["candidate_index"], df["score"], color="0.65", lw=1.2)
    rand = df[df["phase"] == "random"]
    refi = df[df["phase"] == "refine"]
    ax.scatter(rand["candidate_index"], rand["score"], c="tab:blue", s=35, label="random")
    ax.scatter(refi["candidate_index"], refi["score"], c="tab:orange", s=40, label="refine")
    ax.set_xlabel("candidate_index")
    ax.set_ylabel("score")
    ax.set_title(f"{case_id}: score vs candidate index")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.savefig(outdir / f"{case_id}__trajectory_score_vs_iteration.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / f"{case_id}__trajectory_score_vs_iteration.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # best score so far
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df["candidate_index"], df["best_score_so_far"], color="tab:green", lw=2)
    ax.set_xlabel("candidate_index")
    ax.set_ylabel("best_score_so_far")
    ax.set_title(f"{case_id}: best score so far vs candidate index")
    ax.grid(alpha=0.3)
    fig.savefig(outdir / f"{case_id}__trajectory_best_score_so_far.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / f"{case_id}__trajectory_best_score_so_far.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # 3D parameter path
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(df["energy_threshold"], df["beta"], df["mu"], color="0.6", lw=1.0)
    ax.scatter(rand["energy_threshold"], rand["beta"], rand["mu"], c="tab:blue", s=28, label="random")
    ax.scatter(refi["energy_threshold"], refi["beta"], refi["mu"], c="tab:orange", s=32, label="refine")
    ax.set_xlabel("energy_threshold")
    ax.set_ylabel("beta")
    ax.set_zlabel("mu")
    ax.set_title(f"{case_id}: 3D parameter trajectory (threshold, beta, mu)")
    ax.legend(loc="best")
    fig.savefig(outdir / f"{case_id}__trajectory_3d_threshold_beta_mu.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / f"{case_id}__trajectory_3d_threshold_beta_mu.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig)


def pair_r2(df: pd.DataFrame, x: str, y: str) -> float:
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(), df[y].to_numpy()])
    z = df["score"].to_numpy()
    if len(df) < 4:
        return float("nan")
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    zhat = X @ beta
    ssr = np.sum((z - zhat) ** 2)
    sst = np.sum((z - z.mean()) ** 2) + 1e-12
    return float(max(0.0, 1.0 - ssr / sst))


def has_multiple_basin_hint(df: pd.DataFrame) -> bool:
    # Heuristic: top quartile spread very wide in any key 2D pair.
    q = df["score"].quantile(0.75)
    top = df[df["score"] >= q]
    if len(top) < 4:
        return False
    for x, y in PAIR_SPECS:
        vals = top[[x, y]].to_numpy(dtype=float)
        for j in range(2):
            rng = vals[:, j].max() - vals[:, j].min()
            if rng > 0:
                vals[:, j] = (vals[:, j] - vals[:, j].min()) / rng
        if len(vals) >= 4:
            dmax = float(pdist(vals).max())
            if dmax > 1.0:
                return True
    return False


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


def main():
    args = parse_args()
    root = Path(args.root)
    trace_dir = root / "candidate_traces"
    merged_csv = root / "candidate_traces_merged.csv"
    reports_dir = root / "reports"
    selected_txt = reports_dir / "selected_cases.txt"
    rerun_opt = root / "rerun_outputs" / "opt_summary.csv"

    surface_dir = root / "surface_plots"
    traj_dir = root / "trajectory_plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    surface_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    selected = load_selected_cases(selected_txt)
    rerun_df = pd.read_csv(rerun_opt)
    rerun_df.columns = [c.strip() for c in rerun_df.columns]
    rerun_df["filename"] = rerun_df["filename"].astype(str).str.strip()

    case_summaries = []
    for _, row in selected.iterrows():
        filename = str(row["filename"]).strip()
        case_id = Path(filename).stem
        tpath = trace_dir / f"{case_id}.csv"
        if not tpath.exists():
            # fallback: extract from merged
            if merged_csv.exists():
                mdf = pd.read_csv(merged_csv)
                cdf = mdf[mdf["case_id"] == case_id].copy()
                if len(cdf):
                    cdf.to_csv(tpath, index=False)
            if not tpath.exists():
                case_summaries.append({"case_id": case_id, "filename": filename, "missing_trace": True})
                continue

        cdf = read_trace_csv(tpath)
        final_row = None
        m = rerun_df[rerun_df["filename"] == filename]
        if len(m):
            final_row = m.iloc[0]

        # A + C: surfaces and density per pair
        for x, y in PAIR_SPECS:
            make_surface_plot(cdf, x, y, case_id, final_row, surface_dir)
            make_density_plot(cdf, x, y, case_id, final_row, traj_dir)

        # B trajectories
        make_trajectory_plots(cdf, case_id, traj_dir)

        rand = cdf[cdf["phase"] == "random"]
        refi = cdf[cdf["phase"] == "refine"]
        best_random = float(rand["score"].max()) if len(rand) else float("nan")
        best_refine = float(refi["score"].max()) if len(refi) else float("nan")
        best_final = float(cdf["score"].max())
        improve = best_final - best_random if not math.isnan(best_random) else float("nan")
        pair_scores = {f"{x} vs {y}": pair_r2(cdf, x, y) for x, y in PAIR_SPECS}
        dominant_pair = sorted(pair_scores.items(), key=lambda kv: -np.nan_to_num(kv[1], nan=-1))[0][0]
        span = float(cdf["score"].max() - cdf["score"].min())
        sharpness = "flat" if span < 0.8 else ("moderate" if span < 2.0 else "sharp")
        basins = has_multiple_basin_hint(cdf)
        case_summaries.append(
            {
                "case_id": case_id,
                "filename": filename,
                "group": str(row["group"]),
                "n_candidates": int(len(cdf)),
                "n_random": int(len(rand)),
                "n_refine": int(len(refi)),
                "best_random": best_random,
                "best_refine": best_refine,
                "best_final": best_final,
                "improve_from_random": improve,
                "runtime_seconds_trace": float(cdf["elapsed_seconds"].max()) if "elapsed_seconds" in cdf.columns else float("nan"),
                "dominant_pair": dominant_pair,
                "sharpness": sharpness,
                "multi_basin_hint": basins,
            }
        )

    summary_df = pd.DataFrame(case_summaries)
    summary_df.to_csv(reports_dir / "surface_case_summary.csv", index=False)
    summary_df.to_csv(reports_dir / "candidate_trace_summary.csv", index=False)

    # Global interpretation text
    if "missing_trace" in summary_df.columns:
        valid = summary_df[~summary_df["missing_trace"].fillna(False)].copy()
    else:
        valid = summary_df.copy()
    hi = valid[valid["group"] == "high"]
    lo = valid[valid["group"] == "low"]
    med = valid[valid["group"] == "median"]

    lines = []
    lines.append("True Trace-Based Objective-Surface Study Report")
    lines.append("====================================================")
    lines.append("")
    lines.append("This study reconstructs objective geometry from true candidate traces logged during optimization.")
    lines.append(f"Output root: {root}")
    lines.append(f"Selected leaves (target): {len(selected)}")
    lines.append(f"Selected leaves with traces: {len(valid)}")
    lines.append("")
    lines.append("Selected leaves:")
    for _, r in selected.iterrows():
        lines.append(f"- {r['group']}: {r['filename']} (original score={float(r['score']):.6f})")
    lines.append("")
    lines.append("Surface interpretation (from trace samples):")
    if len(valid):
        sharp_counts = valid["sharpness"].value_counts().to_dict()
        lines.append(f"- Surface sharpness distribution: {sharp_counts}")
        lines.append(
            "- Random vs refine contribution: mean(final-best_random) = "
            f"{valid['improve_from_random'].mean():+.4f}"
        )
        near = (valid["improve_from_random"].abs() < 0.15).sum()
        lines.append(f"- Cases where random search already near final optimum (|gain|<0.15): {near}/{len(valid)}")
        lines.append(
            "- Dominant parameter-pair frequency: "
            + str(valid["dominant_pair"].value_counts().to_dict())
        )
        lines.append(
            "- Multiple-basin hints (heuristic): "
            f"{int(valid['multi_basin_hint'].sum())}/{len(valid)} cases"
        )
    lines.append("")
    lines.append("High vs low score geometry:")
    if len(hi) and len(lo):
        lines.append(
            f"- High-score mean candidates={hi['n_candidates'].mean():.1f}, "
            f"mean refine gain={hi['improve_from_random'].mean():+.4f}"
        )
        lines.append(
            f"- Low-score mean candidates={lo['n_candidates'].mean():.1f}, "
            f"mean refine gain={lo['improve_from_random'].mean():+.4f}"
        )
        lines.append(
            "- If low-score leaves show flatter spans and weaker refine gains, this suggests less exploitable local curvature."
        )
    lines.append("")
    lines.append("Generated plot sets:")
    lines.append("- Surface plots (trace points + optional fitted trisurface): surface_plots/")
    lines.append("- Trajectories (score path, best-so-far path, 3D parameter path): trajectory_plots/")
    lines.append("- Density plots by parameter pair and phase split: trajectory_plots/")
    lines.append("")
    lines.append("Method note:")
    lines.append("- Surfaces are truly reconstructed from logged candidate evaluations (random + refine phases).")
    lines.append("- With sparse traces per case, plots are primarily scatter-surfaces with interpolation only when data support it.")
    lines.append("")
    lines.append("Per-case summary:")
    for _, r in valid.iterrows():
        lines.append(
            f"- {r['case_id']} [{r['group']}]: n={int(r['n_candidates'])}, "
            f"best_random={r['best_random']:+.4f}, best_refine={r['best_refine']:+.4f}, "
            f"best_final={r['best_final']:+.4f}, gain={r['improve_from_random']:+.4f}, "
            f"dominant_pair={r['dominant_pair']}, sharpness={r['sharpness']}, "
            f"multi_basin_hint={bool(r['multi_basin_hint'])}"
        )

    report_txt = root / "surface_report.txt"
    report_pdf = root / "surface_report.pdf"
    report_txt.write_text("\n".join(lines) + "\n")
    write_pdf_from_text("\n".join(lines), report_pdf)
    # Mirror reports in reports/ for the combined-study layout.
    (reports_dir / "surface_report.txt").write_text("\n".join(lines) + "\n")
    write_pdf_from_text("\n".join(lines), reports_dir / "surface_report.pdf")
    print(f"Wrote {report_txt}")
    print(f"Wrote {report_pdf}")


if __name__ == "__main__":
    main()
