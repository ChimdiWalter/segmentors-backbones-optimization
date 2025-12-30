#!/usr/bin/env python3
import os, glob, argparse, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

# ------------------------ I/O helpers ------------------------
def load_logs(paths):
    frames = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            cands = glob.glob(os.path.join(p, "**", "results_log.csv"), recursive=True)
        else:
            cands = [p]
        for c in cands:
            if not os.path.exists(c):
                continue
            try:
                df = pd.read_csv(c)
                df["__log_path__"] = c
                frames.append(df)
            except Exception as e:
                print(f"[warn] failed {c}: {e}")
    if not frames:
        raise SystemExit("No results_log.csv found in provided paths.")
    df = pd.concat(frames, ignore_index=True)

    # Coerce types
    for col in ["mu","lambda","diffusion_rate","alpha","beta","gamma","energy_threshold",
                "n_contours","total_area_px"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ensure these exist (some older logs may not have them)
    for col in ["alpha","beta","gamma","energy_threshold"]:
        if col not in df.columns:
            df[col] = np.nan

    # Derive mode label from path, if missing
    if "mode" not in df.columns:
        def tag_mode(p):
            p = p.lower()
            if "bestch" in p: return "bestch"
            if "fused"  in p: return "fused"
            return "unknown"
        df["mode"] = df["__log_path__"].apply(tag_mode)

    # Derive channel label from overlay/mask path or filename if "channel" missing
    if "channel" not in df.columns:
        def tag_channel(row):
            for col in ["overlay16_path","mask16_path","filename","__log_path__"]:
                v = str(row.get(col, "")).lower()
                if "ch_r" in v or "_chR_".lower() in v or "__chr__" in v: return "R"
                if "ch_g" in v or "_chG_".lower() in v or "__chg__" in v: return "G"
                if "ch_b" in v or "_chB_".lower() in v or "__chb__" in v: return "B"
            return ""
        df["channel"] = df.apply(tag_channel, axis=1)

    return df

def binarize_mask(path):
    # accept tif/png/etc, return boolean numpy array
    with Image.open(path) as im:
        arr = np.array(im)
    # any non-zero => True
    return (arr.astype(np.uint64) > 0)

def safe_iou(a_path, b_path):
    if not (isinstance(a_path, str) and isinstance(b_path, str)):
        return np.nan
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        return np.nan
    try:
        A = binarize_mask(a_path)
        B = binarize_mask(b_path)
        if A.shape != B.shape:
            return np.nan
        inter = np.logical_and(A, B).sum()
        union = np.logical_or(A, B).sum()
        return float(inter) / float(union) if union > 0 else 1.0
    except Exception:
        return np.nan

# ------------------------ core compare ------------------------
KEY_COLS = ["filename","mu","lambda","diffusion_rate","alpha","beta","gamma","energy_threshold"]

def prepare_subset(df, label_col, label_value):
    sub = df.copy()
    sub = sub[sub[label_col] == label_value]
    # Keep necessary cols
    cols = KEY_COLS + ["n_contours","total_area_px","overlay16_path","mask16_path"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].rename(columns={
        "n_contours": f"n_contours__{label_value}",
        "total_area_px": f"total_area_px__{label_value}",
        "overlay16_path": f"overlay16_path__{label_value}",
        "mask16_path": f"mask16_path__{label_value}",
    })

def pairwise_compare(df, label_col, A, B, compute_iou=False):
    a = prepare_subset(df, label_col, A)
    b = prepare_subset(df, label_col, B)
    merged = pd.merge(a, b, on=[c for c in KEY_COLS if c in a.columns and c in b.columns], how="inner")

    # Deltas and % deltas
    for metric in ["n_contours", "total_area_px"]:
        ma = f"{metric}__{A}"
        mb = f"{metric}__{B}"
        if ma in merged.columns and mb in merged.columns:
            merged[f"abs_d_{metric}"] = (merged[ma] - merged[mb]).abs()
            denom = np.maximum(1.0, 0.5*(merged[ma].astype(float) + merged[mb].astype(float)))
            merged[f"pct_d_{metric}"] = merged[f"abs_d_{metric}"] / denom
        else:
            merged[f"abs_d_{metric}"] = np.nan
            merged[f"pct_d_{metric}"] = np.nan

    if compute_iou:
        merged["IoU_masks"] = merged.apply(
            lambda r: safe_iou(r.get(f"mask16_path__{A}"), r.get(f"mask16_path__{B}")),
            axis=1
        )
    return merged

def convergence_flags(df_pair, pct_tol_contours=0.15, pct_tol_area=0.15, iou_tol=0.6, use_iou=False):
    ok_cont = df_pair["pct_d_n_contours"] <= pct_tol_contours
    ok_area = df_pair["pct_d_total_area_px"] <= pct_tol_area
    if use_iou and "IoU_masks" in df_pair.columns:
        ok_iou = df_pair["IoU_masks"] >= iou_tol
        return ok_cont & ok_area & ok_iou
    return ok_cont & ok_area

def aggregate_by_combo(df_pair, conv_flag_col, label=(None,None)):
    cols = [c for c in KEY_COLS if c in df_pair.columns]
    agg = (df_pair.groupby(cols, dropna=False)
           .agg(
               n_pairs=("filename", "count"),
               conv_rate=(conv_flag_col, "mean"),
               mean_IoU=("IoU_masks", "mean"),
               mean_abs_d_contours=("abs_d_n_contours","mean"),
               mean_abs_d_area=("abs_d_total_area_px","mean")
           )
           .reset_index())
    if label[0] and label[1]:
        agg.insert(0, "pair", f"{label[0]}_vs_{label[1]}")
    return agg

def aggregate_by_param_grid(df_pair, conv_flag_col):
    # μ×λ per diffusion_rate
    have = all(c in df_pair.columns for c in ["mu","lambda","diffusion_rate"])
    if not have: 
        return pd.DataFrame()
    g = (df_pair.groupby(["diffusion_rate","mu","lambda"], dropna=False)
                 .agg(conv_rate=(conv_flag_col,"mean"),
                      n_pairs=("filename","count"))
                 .reset_index())
    return g

# ------------------------ plotting ------------------------
def plot_3d_conv_rate(df_grid, outdir, title_prefix):
    if df_grid.empty: return
    os.makedirs(outdir, exist_ok=True)
    for dr, sub in df_grid.groupby("diffusion_rate"):
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(sub["mu"], sub["lambda"], np.full(len(sub), dr),
                        c=sub["conv_rate"], s=70, cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("mu"); ax.set_ylabel("lambda"); ax.set_zlabel("diffusion_rate")
        ax.set_title(f"{title_prefix} — conv_rate @ d={dr}")
        cb = plt.colorbar(sc, ax=ax, pad=0.12); cb.set_label("convergence rate")
        fig.tight_layout()
        path = os.path.join(outdir, f"conv3d_dr{dr}.png")
        fig.savefig(path, dpi=160); plt.close(fig)

def plot_heatmaps(df_grid, outdir, title_prefix):
    if df_grid.empty: return
    os.makedirs(outdir, exist_ok=True)
    for dr, sub in df_grid.groupby("diffusion_rate"):
        pivot = sub.pivot(index="mu", columns="lambda", values="conv_rate")
        fig, ax = plt.subplots(figsize=(6.2,5.0))
        im = ax.imshow(pivot.values, origin="lower", vmin=0, vmax=1)
        ax.set_title(f"{title_prefix} — conv_rate heatmap @ d={dr}")
        ax.set_xlabel("lambda"); ax.set_ylabel("mu")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{v:g}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels([f"{v:g}" for v in pivot.index])
        cb = fig.colorbar(im, ax=ax); cb.set_label("convergence rate")
        fig.tight_layout()
        path = os.path.join(outdir, f"heatmap_dr{dr}.png")
        fig.savefig(path, dpi=160); plt.close(fig)

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser(description="Convergence across variants/channels using Navier logs.")
    ap.add_argument("paths", nargs="+", help="Dirs or CSVs to read (recursive search for results_log.csv).")
    ap.add_argument("--compare-col", default="mode", choices=["mode","channel"],
                    help="Column to compare on (default: mode).")
    ap.add_argument("--A", required=True, help="Label A (e.g., bestch or G).")
    ap.add_argument("--B", required=True, help="Label B (e.g., fused or R).")
    ap.add_argument("--out", required=True, help="Output folder.")
    ap.add_argument("--pct-tol-contours", type=float, default=0.15, help="Pct tol for contours (default 0.15).")
    ap.add_argument("--pct-tol-area", type=float, default=0.15, help="Pct tol for area (default 0.15).")
    ap.add_argument("--iou", action="store_true", help="Also compute IoU from mask16 (slower).")
    ap.add_argument("--iou-tol", type=float, default=0.6, help="IoU tolerance (default 0.6).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_logs(args.paths)
    if args.compare_col not in df.columns:
        raise SystemExit(f"'{args.compare_col}' column not present in data.")

    # Make pairwise table
    pair = pairwise_compare(df, args.compare_col, args.A, args.B, compute_iou=args.iou)

    # Convergence flags
    pair["converged"] = convergence_flags(
        pair,
        pct_tol_contours=args.pct_tol_contours,
        pct_tol_area=args.pct_tol_area,
        iou_tol=args.iou_tol,
        use_iou=args.iou
    )

    # Save row-wise table
    pair_path = os.path.join(args.out, "pairwise_matches.csv")
    pair.to_csv(pair_path, index=False)

    # Aggregate summaries
    by_combo = aggregate_by_combo(pair, "converged", label=(args.A,args.B))
    by_combo_path = os.path.join(args.out, "convergence_by_combo.csv")
    by_combo.to_csv(by_combo_path, index=False)

    by_grid = aggregate_by_param_grid(pair, "converged")
    by_grid_path = os.path.join(args.out, "convergence_by_param_grid.csv")
    by_grid.to_csv(by_grid_path, index=False)

    # Plots
    plot_3d_conv_rate(by_grid, os.path.join(args.out,"plots3d"),
                      f"{args.compare_col}: {args.A} vs {args.B}")
    plot_heatmaps(by_grid, os.path.join(args.out,"heatmaps"),
                  f"{args.compare_col}: {args.A} vs {args.B}")

    # Console summary
    overall = float(pair["converged"].mean()) if len(pair) else float("nan")
    print(f"[done] pairs: {len(pair)}  | overall convergence={overall:.3f}")
    print("Saved:")
    print(" ", pair_path)
    print(" ", by_combo_path)
    print(" ", by_grid_path)
    print(" ", os.path.join(args.out, "plots3d/"))
    print(" ", os.path.join(args.out, "heatmaps/"))

if __name__ == "__main__":
    main()
