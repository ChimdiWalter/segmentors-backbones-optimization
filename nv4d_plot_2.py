#!/usr/bin/env python3
import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

AREA_ALIASES = ["total_area_px", "total_area", "area_px", "area"]

def infer_mode_from_path(path: str) -> str:
    p = path.lower()
    if "bestch" in p: return "bestch"
    if "fused"  in p: return "fused"
    return "unknown"

def resolve_column(df: pd.DataFrame, candidates, must_exist=False):
    for c in candidates:
        if c in df.columns:
            return c
    if must_exist:
        raise KeyError(f"Missing required column; tried: {candidates}")
    return None

def load_logs(dirs, verbose=True):
    frames = []
    for d in dirs:
        cands = []
        if os.path.isdir(d):
            cands = glob.glob(os.path.join(d, "**", "results_log.csv"), recursive=True)
        elif d.lower().endswith(".csv") and os.path.isfile(d):
            cands = [d]
        if not cands and verbose:
            print(f"[warn] no CSVs in {d}")

        for csv_path in cands:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[skip] {csv_path}: {e}")
                continue

            df["source_dir"] = os.path.dirname(csv_path)
            df["mode"] = infer_mode_from_path(df["source_dir"].iloc[0])

            # Require the 3 NV params
            for col in ["mu","lambda","diffusion_rate"]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' missing in {csv_path}")

            # Normalize area column name (if any)
            area_col = resolve_column(df, AREA_ALIASES, must_exist=False)
            if area_col and area_col != "total_area_px":
                if "total_area_px" in df.columns and area_col != "total_area_px":
                    pass  # already present
                else:
                    df = df.rename(columns={area_col: "total_area_px"})

            if verbose:
                present = ", ".join(sorted(df.columns))
                print(f"[load] {csv_path}  mode={df['mode'].iloc[0]}  cols=[{present}]")

            frames.append(df)
    if not frames:
        raise SystemExit("No results_log.csv files found.")
    return pd.concat(frames, ignore_index=True)

def build_normalizer(values, kind="quantile", clip=None):
    vals = np.asarray(values, dtype=float)
    v = vals.copy()
    if clip and len(clip) == 2:
        lo, hi = np.percentile(v, [clip[0], clip[1]])
        v = np.clip(v, lo, hi)
        vmin, vmax = lo, hi
    else:
        vmin, vmax = np.nanmin(v), np.nanmax(v)
    if vmin == vmax: vmax = vmin + 1e-9

    if kind == "log":
        shift = 0.0
        if np.any(vals <= 0):
            shift = np.nanmin(vals) - 1e-9
            vals = vals - shift + 1e-9
            vmin = vmin - shift + 1e-9
            vmax = vmax - shift + 1e-9
        return mcolors.LogNorm(vmin=max(vmin, 1e-9), vmax=vmax)

    if kind == "zscore":
        mu, sd = np.nanmean(vals), np.nanstd(vals) or 1.0
        class ZNorm(mcolors.Normalize):
            def __call__(self, val, clip=False):
                z = (np.asarray(val) - mu) / sd
                z01 = 0.5*(np.tanh(z/2.0)+1.0)
                return np.ma.masked_array(z01)
        return ZNorm()

    if kind == "linear":
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    # quantile: map to ECDF rank
    xs = np.sort(vals)
    ps = np.linspace(0, 1, len(xs), endpoint=True)
    class QNorm(mcolors.Normalize):
        def __call__(self, val, clip=False):
            arr = np.asarray(val, dtype=float)
            prc = np.interp(arr, xs, ps, left=0.0, right=1.0)
            return np.ma.masked_array(prc)
    return QNorm()

def size_mapping(values, s_min=30, s_max=140, clip=None):
    v = np.asarray(values, dtype=float)
    if clip and len(clip) == 2:
        lo, hi = np.percentile(v, [clip[0], clip[1]])
        v = np.clip(v, lo, hi)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if vmin == vmax: vmax = vmin + 1e-9
    return s_min + (s_max - s_min) * ((values - vmin) / (vmax - vmin))

def plot_3d(df, by, title_prefix, out_png=None, cmap_name="viridis",
            norm_kind="quantile", clip=None, size_by=None, alpha=0.9):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap(cmap_name)
    norm = build_normalizer(df[by].to_numpy(), kind=norm_kind, clip=clip)

    if size_by and size_by in df.columns:
        sizes = size_mapping(df[size_by].to_numpy(), s_min=30, s_max=140, clip=clip)
    else:
        sizes = 70
        if size_by and size_by not in df.columns:
            print(f"[warn] size-by column '{size_by}' not found; using constant marker size.")

    sc = ax.scatter(
        df["mu"], df["lambda"], df["diffusion_rate"],
        c=df[by], cmap=cmap, norm=norm, s=sizes,
        depthshade=False, alpha=alpha, edgecolors="k", linewidths=0.25
    )

    ax.set_xlabel("mu")
    ax.set_ylabel("lambda")
    ax.set_zlabel("diffusion_rate")
    ax.set_title(f"{title_prefix}: color={by} (norm={norm_kind})"
                 + (f", size={size_by}" if size_by and size_by in df.columns else ""))

    cbar = plt.colorbar(sc, ax=ax, pad=0.12)
    cbar.set_label(by)

    # tick at unique param values
    for axis, col in zip([ax.xaxis, ax.yaxis, ax.zaxis], ["mu","lambda","diffusion_rate"]):
        vals = np.sort(df[col].unique())
        axis.set_ticks(vals)

    ax.view_init(elev=20, azim=35)

    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
        print(f"[saved] {out_png}")
    else:
        plt.show()
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="4D scatter of Navier params (robust colors + optional size).")
    ap.add_argument("paths", nargs="+", help="Folders containing results_log.csv (or CSV files)")
    ap.add_argument("--by", default="n_contours", choices=["n_contours","total_area_px","total_area","area","area_px"],
                    help="Metric for color (default: n_contours)")
    ap.add_argument("--size-by", help="Optional metric to scale point size (e.g., total_area_px).")
    ap.add_argument("--raw", action="store_true", help="Plot raw rows (no aggregation).")
    ap.add_argument("--split-by-mode", action="store_true", help="One plot per mode (fused/bestch/unknown).")
    ap.add_argument("--save", metavar="OUT_DIR", help="Save PNGs to this directory.")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default: viridis).")
    ap.add_argument("--norm", default="quantile", choices=["quantile","linear","log","zscore"],
                    help="Color normalization (default: quantile).")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("LO","HI"),
                    help="Percentile clip for colors (e.g., --clip 2 98).")
    ap.add_argument("--alpha", type=float, default=0.9, help="Point alpha (default: 0.9).")
    args = ap.parse_args()

    df = load_logs(args.paths, verbose=True)

    # unify color metric column name if user passed an alias
    color_by = args.by
    if color_by not in df.columns:
        alias = resolve_column(df, [color_by] + AREA_ALIASES, must_exist=False)
        if alias and alias in df.columns:
            color_by = alias
        else:
            raise SystemExit(f"Color metric '{args.by}' not found in data.")

    # unify size metric column name
    size_by = args.size_by
    if size_by:
        if size_by not in df.columns:
            alias = resolve_column(df, [size_by] + AREA_ALIASES, must_exist=False)
            if alias and alias in df.columns:
                size_by = alias
            else:
                print(f"[warn] size-by metric '{args.size_by}' not found in any known aliases; disabling size scaling.")
                size_by = None

    # aggregate or raw
    if args.raw:
        plot_df = df.copy()
        suffix = "raw"
    else:
        gb = df.groupby(["mode","mu","lambda","diffusion_rate"], as_index=False)\
               .agg({color_by:"mean", **({size_by:"mean"} if size_by else {})})
        plot_df = gb
        suffix = "avg"

    def do_plot(sub, tag):
        out_png = None
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            out_png = os.path.join(args.save, f"{tag}__{suffix}__{color_by}.png")
        plot_3d(sub, color_by, tag + f" ({suffix})",
                out_png=out_png, cmap_name=args.cmap,
                norm_kind=args.norm, clip=args.clip, size_by=size_by, alpha=args.alpha)

    if args.split_by_mode:
        for mode, sub in plot_df.groupby("mode"):
            if not sub.empty:
                do_plot(sub, mode)
    else:
        do_plot(plot_df, "all_modes")

if __name__ == "__main__":
    main()
