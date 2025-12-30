#!/usr/bin/env python3
import argparse, os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
plt.rcParams["figure.dpi"] = 120

REQUIRED_COLS = {
    "mu","lambda","diffusion_rate",
    "n_contours","total_area_px",
    "overlay16_path","mask16_path","filename"
}

def infer_mode_from_dir(d):
    # crude but works for your naming
    base = os.path.basename(os.path.normpath(d)).lower()
    for m in ["bestch","fused","auto","tiled","unknown"]:
        if m in base:
            return m if m in ["bestch","fused"] else "unknown"
    return "unknown"

def load_one_dir(d):
    # find any results_log.csv in the tree (handles tiled/auto)
    cands = glob.glob(os.path.join(d, "**", "results_log.csv"), recursive=True)
    if not cands:
        cand = os.path.join(d, "results_log.csv")
        if os.path.isfile(cand):
            cands = [cand]
    frames = []
    for csvp in cands:
        try:
            df = pd.read_csv(csvp)
            missing = REQUIRED_COLS - set(df.columns)
            if missing:
                # try to be lenient (some variants may not have modes/cols)
                common = REQUIRED_COLS & set(df.columns)
                if len(common) < 6:
                    print(f"[skip] {csvp}: missing many cols {missing}")
                    continue
            df["mode"] = infer_mode_from_dir(d)
            frames.append(df)
        except Exception as e:
            print(f"[skip] {csvp}: {e}")
    if frames:
        out = pd.concat(frames, ignore_index=True)
        # clean footprints
        for c in ["mu","lambda","diffusion_rate"]:
            out[c] = out[c].astype(float)
        for c in ["n_contours","total_area_px"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
        return out
    return pd.DataFrame()

def scatter_3d(df, metric, title, savepath, averaged=False):
    if df.empty:
        return
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if averaged:
        g = (df.groupby(["mu","lambda","diffusion_rate"], as_index=False)
               .agg({metric:"mean"}))
        x, y, z = g["mu"].values, g["lambda"].values, g["diffusion_rate"].values
        c = g[metric].values
    else:
        x, y, z = df["mu"].values, df["lambda"].values, df["diffusion_rate"].values
        c = df[metric].values

    sc = ax.scatter(x, y, z, c=c, cmap="viridis", s=40, depthshade=True)
    ax.set_xlabel("mu"); ax.set_ylabel("lambda"); ax.set_zlabel("diffusion_rate")
    fig.colorbar(sc, ax=ax, pad=0.1, label=metric)
    ax.set_title(title)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

def summarize_top(df, by_metric="n_contours"):
    cols = ["mode","mu","lambda","diffusion_rate","n_runs",
            "mean_contours","median_contours","mean_area"]
    if df.empty: 
        return pd.DataFrame(columns=cols)
    grp = (df.groupby(["mode","mu","lambda","diffusion_rate"])
             .agg(n_runs=("filename","count"),
                  mean_contours=("n_contours","mean"),
                  median_contours=("n_contours","median"),
                  mean_area=("total_area_px","mean"))
             .reset_index())
    # nicer order
    grp = grp.sort_values(["mode","mean_contours"], ascending=[True, False])
    # make numbers nicer
    grp["mean_contours"] = grp["mean_contours"].round(3)
    grp["median_contours"] = grp["median_contours"].round(1)
    grp["mean_area"] = grp["mean_area"].round(3)
    return grp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="One or more output dirs that contain results_log.csv")
    ap.add_argument("--out", required=True, help="Where to save figures + summary")
    ap.add_argument("--metric", default="n_contours", choices=["n_contours","total_area_px"])
    ap.add_argument("--also-area", action="store_true", help="Additionally plot total_area_px")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # load + concatenate
    frames = [load_one_dir(d) for d in args.dirs]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        print("No rows found.")
        return

    # global summary to CSV
    desc_cnt = df["n_contours"].describe()
    desc_area = df["total_area_px"].describe()
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write("[summary] rows: %d\n" % len(df))
        fh.write(str(desc_cnt) + "\n")
        fh.write(str(desc_area) + "\n")

    # top combos table
    top = summarize_top(df, by_metric=args.metric)
    top.to_csv(os.path.join(args.out, "top_combos_by_mean_contours.csv"), index=False)

    # per-mode plots (raw + averaged) for the chosen metric
    metrics = [args.metric]
    if args.also_area and args.metric != "total_area_px":
        metrics.append("total_area_px")

    for metric in metrics:
        for mode, dmode in df.groupby("mode"):
            # raw
            scatter_3d(
                dmode, metric,
                title=f"{mode}: (raw) {metric}",
                savepath=os.path.join(args.out, f"{mode}__raw__{metric}.png"),
                averaged=False
            )
            # averaged per (mu,lambda,dr)
            scatter_3d(
                dmode, metric,
                title=f"{mode}: (avg over images) {metric}",
                savepath=os.path.join(args.out, f"{mode}__avg__{metric}.png"),
                averaged=True
            )

    print(f"Saved figures and tables to: {args.out}")

if __name__ == "__main__":
    main()
