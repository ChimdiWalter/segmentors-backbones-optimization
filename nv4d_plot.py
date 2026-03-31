#!/usr/bin/env python3
import argparse, os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def infer_mode_from_path(path: str) -> str:
    p = path.lower()
    if "bestch" in p: return "bestch"
    if "fused"  in p: return "fused"
    return "unknown"

def load_logs(dirs):
    frames = []
    for d in dirs:
        # pick up any results_log.csv under the directory
        candidates = glob.glob(os.path.join(d, "**", "results_log.csv"), recursive=True)
        if not candidates:
            # also allow passing the csv directly
            if d.lower().endswith(".csv") and os.path.isfile(d):
                candidates = [d]
        for csv_path in candidates:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[skip] {csv_path}: {e}")
                continue
            df["source_dir"] = os.path.dirname(csv_path)
            df["mode"] = infer_mode_from_path(df["source_dir"].iloc[0])
            # normalize expected columns if needed
            for col in ["mu","lambda","diffusion_rate"]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' missing in {csv_path}")
            frames.append(df)
    if not frames:
        raise SystemExit("No results_log.csv files found.")
    all_df = pd.concat(frames, ignore_index=True)
    return all_df

def plot_3d(df, by, title_prefix, out_png=None):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(df["mu"], df["lambda"], df["diffusion_rate"],
                    c=df[by], cmap="plasma", s=50, depthshade=True)
    ax.set_xlabel("mu")
    ax.set_ylabel("lambda")
    ax.set_zlabel("diffusion_rate")
    cbar = plt.colorbar(sc, ax=ax, pad=0.12)
    cbar.set_label(by)
    ax.set_title(f"{title_prefix}: {by}")
    if out_png:
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        print(f"[saved] {out_png}")
    else:
        plt.show()
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="4D scatter of Navier params (color by metric).")
    ap.add_argument("paths", nargs="+",
                    help="Folders that contain results_log.csv (or a CSV path)")
    ap.add_argument("--by", default="n_contours",
                    choices=["n_contours","total_area_px"],
                    help="Metric for color (default: n_contours)")
    ap.add_argument("--raw", action="store_true",
                    help="Plot raw rows (no aggregation).")
    ap.add_argument("--split-by-mode", action="store_true",
                    help="Make one plot per mode (fused/bestch/unknown).")
    ap.add_argument("--save", metavar="OUT_DIR",
                    help="If set, save PNGs into this directory instead of showing.")
    args = ap.parse_args()

    df = load_logs(args.paths)

    # keep only columns we need
    need = {"mu","lambda","diffusion_rate","n_contours","total_area_px","mode"}
    missing = [c for c in ["mu","lambda","diffusion_rate",args.by] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    # optionally aggregate to mean per (mu,lambda,dr,mode)
    if args.raw:
        plot_df = df.copy()
        title_suffix = "(raw)"
    else:
        gb = df.groupby(["mode","mu","lambda","diffusion_rate"], as_index=False)\
               .agg({args.by:"mean"})
        gb = gb.rename(columns={args.by: args.by})
        plot_df = gb
        title_suffix = "(mean per μ,λ,dr)"

    if args.split_by_mode:
        for mode, sub in plot_df.groupby("mode"):
            # guard against empty
            if sub.empty: 
                continue
            out_png = None
            if args.save:
                os.makedirs(args.save, exist_ok=True)
                out_png = os.path.join(args.save, f"{mode}_{args.by}.png")
            plot_3d(sub, args.by, f"{mode}: {title_suffix}", out_png=out_png)
    else:
        out_png = None
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            out_png = os.path.join(args.save, f"all_{args.by}.png")
        plot_3d(plot_df, args.by, f"all modes: {title_suffix}", out_png=out_png)

if __name__ == "__main__":
    main()
