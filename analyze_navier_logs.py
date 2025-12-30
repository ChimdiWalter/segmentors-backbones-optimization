#!/usr/bin/env python3
import os, re, argparse, glob
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---- helpers ---------------------------------------------------------------
def load_logs(paths):
    dfs = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            # accept dir containing results_log.csv
            csvs = glob.glob(os.path.join(p, "**", "results_log.csv"), recursive=True)
        else:
            csvs = [p]
        for c in csvs:
            if os.path.exists(c):
                try:
                    df = pd.read_csv(c)
                    df["__log_path__"] = c
                    dfs.append(df)
                except Exception as e:
                    print(f"[warn] failed to read {c}: {e}")
    if not dfs:
        raise FileNotFoundError("No results_log.csv found in provided paths.")
    df = pd.concat(dfs, ignore_index=True)
    # coerce expected numeric cols if present
    for col in ["mu","lambda","diffusion_rate","alpha","beta","gamma",
                "Eth_q","Eth_seed","n_contours","total_area_px"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # derive a “mode” tag
    def tag(row):
        if "bestch" in row["__log_path__"]:
            return "bestch"
        if "fused" in row["__log_path__"]:
            return "fused"
        return "unknown"
    df["mode"] = df.apply(tag, axis=1)
    return df

def unique_param_key(row):
    keys = ["mu","lambda","diffusion_rate","alpha","beta","gamma"]
    return tuple([row[k] if k in row and pd.notna(row[k]) else None for k in keys])

def quick_table(df):
    piv = (df
           .groupby(["mode","mu","lambda","diffusion_rate"], dropna=False)
           .agg(n_runs=("filename","count"),
                mean_contours=("n_contours","mean"),
                median_contours=("n_contours","median"),
                mean_area=("total_area_px","mean"))
           .reset_index()
           .sort_values(["mode","mean_contours"], ascending=[True, False]))
    return piv

# ---- plotting --------------------------------------------------------------
def plot_3d(df, color_by="n_contours", title=None, save=None):
    required = {"mu","lambda","diffusion_rate",color_by}
    if not required.issubset(df.columns):
        print(f"[skip] missing columns for 3D plot: need {required}")
        return
    sub = df.dropna(subset=["mu","lambda","diffusion_rate",color_by]).copy()
    if sub.empty:
        print("[skip] no rows to plot.")
        return
    # normalizing color range for legibility
    cvals = sub[color_by].values
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(sub["mu"], sub["lambda"], sub["diffusion_rate"],
                    c=cvals, s=36, alpha=0.85)
    ax.set_xlabel("mu")
    ax.set_ylabel("lambda")
    ax.set_zlabel("diffusion_rate")
    ttl = title or f"3D: (mu,lambda,dr) colored by {color_by}"
    ax.set_title(ttl)
    cb = plt.colorbar(sc, pad=0.12, shrink=0.7)
    cb.set_label(color_by)
    ax.view_init(elev=22, azim=40)
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=160)
        print(f"[saved] {save}")
    else:
        plt.show()
    plt.close(fig)

# ---- overlay launcher (optional) ------------------------------------------
def show_examples(df, k_mu, k_lam, k_dr, maxn=6):
    sub = df[(np.isclose(df["mu"], k_mu)) &
             (np.isclose(df["lambda"], k_lam)) &
             (np.isclose(df["diffusion_rate"], k_dr))]
    if sub.empty:
        print("[info] no rows for that (mu,lambda,dr).")
        return
    # prefer rows that have overlay paths
    if "overlay16_path" in sub.columns:
        sub = sub[sub["overlay16_path"].notna()]
    sub = sub.sort_values("n_contours", ascending=False).head(maxn)
    for _, r in sub.iterrows():
        p = r.get("overlay16_path", None)
        if p and os.path.exists(p):
            try:
                img = plt.imread(p)
                plt.figure()
                plt.imshow(img)
                plt.title(f"{os.path.basename(p)}\n"
                          f"n_contours={int(r.get('n_contours',-1))}, "
                          f"area={int(r.get('total_area_px',-1))}")
                plt.axis("off")
                plt.show()
            except Exception as e:
                print(f"[warn] could not display {p}: {e}")
        else:
            print(f"[miss] overlay file not found: {p}")

# --- file helpers -----------------------------------------------------------
def save_text(path, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _describe_to_df(series: pd.Series, metric_name: str) -> pd.DataFrame:
    """Turn series.describe() into a 1-row DataFrame with a 'metric' column."""
    d = series.describe()
    out = d.to_frame().T
    out.insert(0, "metric", metric_name)
    return out

# ---- main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze Navier/Navier+Snake CSV logs.")
    ap.add_argument("paths", nargs="+",
                    help="Path(s) to directories or CSV files (results_log.csv).")
    ap.add_argument("--by", default="n_contours",
                    choices=["n_contours","total_area_px"],
                    help="Color 3D scatter by this metric.")
    ap.add_argument("--split-by-mode", action="store_true",
                    help="Plot separate 3D figures for fused vs bestch.")
    ap.add_argument("--save-dir", default=None,
                    help="If provided, save PNGs and numeric tables here.")
    ap.add_argument("--examples", nargs=3, type=float, default=None,
                    metavar=("MU","LAMBDA","DR"),
                    help="Show example overlays for a specific (mu, lambda, dr).")
    args = ap.parse_args()

    df = load_logs(args.paths)
    print("\n[summary] rows:", len(df))
    if "n_contours" in df.columns:
        print(df["n_contours"].describe())
    if "total_area_px" in df.columns:
        print(df["total_area_px"].describe())

    tbl = quick_table(df)
    with pd.option_context('display.max_rows', 40,'display.width',120):
        print("\n[top combos (mean_contours)]:")
        print(tbl.head(20))

    # ---------- SAVE CSVs & TEXT if requested ----------
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

        # 1) Human-readable summary.txt
        buf = io.StringIO()
        buf.write(f"[summary] rows: {len(df)}\n")
        if "n_contours" in df.columns:
            buf.write(df["n_contours"].describe().to_string() + "\n")
        if "total_area_px" in df.columns:
            buf.write(df["total_area_px"].describe().to_string() + "\n")
        save_text(os.path.join(args.save_dir, "summary.txt"), buf.getvalue())

        # 2) Top combos table (what you print) -> CSV
        tbl.to_csv(os.path.join(args.save_dir, "top_combos_by_mean_contours.csv"), index=False)

        # 3) Full per-(mode, μ, λ, d) summary -> CSV
        combo_stats = (df
            .groupby(["mode","mu","lambda","diffusion_rate"], dropna=False)
            .agg(
                runs=("filename","count"),
                mean_n_contours=("n_contours","mean"),
                median_n_contours=("n_contours","median"),
                mean_total_area=("total_area_px","mean"),
                median_total_area=("total_area_px","median"),
            )
            .reset_index()
            .sort_values(["mode","mean_n_contours"], ascending=[True, False])
        )
        combo_stats.to_csv(os.path.join(args.save_dir, "combo_stats_all.csv"), index=False)

        # 4) Global describe() for each metric -> CSV (tidy)
        desc_rows = []
        if "n_contours" in df.columns:
            desc_rows.append(_describe_to_df(df["n_contours"], "n_contours"))
        if "total_area_px" in df.columns:
            desc_rows.append(_describe_to_df(df["total_area_px"], "total_area_px"))
        if desc_rows:
            pd.concat(desc_rows, ignore_index=True)\
              .to_csv(os.path.join(args.save_dir, "global_describe.csv"), index=False)

        # 5) Per-mode describe() for each metric -> CSV
        if "mode" in df.columns:
            per_mode_desc = []
            for metric in ["n_contours", "total_area_px"]:
                if metric in df.columns:
                    d = (df.groupby("mode")[metric]
                           .describe()
                           .reset_index())
                    # add metric column to keep both in one CSV
                    d.insert(1, "metric", metric)
                    per_mode_desc.append(d)
            if per_mode_desc:
                pd.concat(per_mode_desc, ignore_index=True)\
                  .to_csv(os.path.join(args.save_dir, "per_mode_describe.csv"), index=False)

    # ---------- 3D plots ----------
    if args.split_by_mode and "mode" in df.columns:
        for m in sorted(df["mode"].unique()):
            sub = df[df["mode"]==m]
            out = os.path.join(args.save_dir, f"scatter3d_{m}_{args.by}.png") if args.save_dir else None
            plot_3d(sub, color_by=args.by, title=f"{m}: {args.by}", save=out)
    else:
        out = os.path.join(args.save_dir, f"scatter3d_all_{args.by}.png") if args.save_dir else None
        plot_3d(df, color_by=args.by, title=f"all modes: {args.by}", save=out)

    # ---------- optional overlay previews ----------
    if args.examples is not None:
        mu, lam, dr = args.examples
        show_examples(df, mu, lam, dr)

if __name__ == "__main__":
    main()
