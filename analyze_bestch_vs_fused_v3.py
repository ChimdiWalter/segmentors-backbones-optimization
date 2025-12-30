#!/usr/bin/env python3
import os, argparse, glob
import numpy as np
import pandas as pd
import cv2

# ---------------- utils ----------------
def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_log_dir(root):
    cands = glob.glob(os.path.join(root, "**", "results_log.csv"), recursive=True)
    if not cands:
        cand = os.path.join(root, "results_log.csv")
        if os.path.isfile(cand):
            cands = [cand]
    frames = []
    for path in cands:
        try:
            df = pd.read_csv(path)
            df["__log_path__"] = path
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")
    if not frames:
        raise FileNotFoundError(f"No results_log.csv in {root}")
    df = pd.concat(frames, ignore_index=True)
    need = ["filename","mu","lambda","diffusion_rate","energy_threshold",
            "mask16_path","overlay16_path","n_contours","total_area_px"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"[warn] missing columns in logs: {missing}")
    coerce_numeric(df, ["mu","lambda","diffusion_rate","energy_threshold","n_contours","total_area_px"])
    return df

def apply_rewrites(path, rewrites):
    p = path if isinstance(path, str) else path
    for old, new in rewrites:
        if old and new and isinstance(p, str) and old in p:
            p = p.replace(old, new, 1)
    return p

def read_mask_binary(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return (im.astype(np.uint64) > 0)

def iou(a, b):
    if a is None or b is None: return np.nan
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (inter/union) if union>0 else 1.0

def dice(a, b):
    if a is None or b is None: return np.nan
    inter = np.logical_and(a, b).sum()
    s = a.sum()+b.sum()
    return (2*inter/s) if s>0 else 1.0

def make_pair_key_row(row):
    def f(x):
        try: return f"{float(x):.6g}"
        except: return str(x)
    return "|".join([
        row.get("filename",""),
        f(row.get("mu","")),
        f(row.get("lambda","")),
        f(row.get("diffusion_rate","")),
        f(row.get("energy_threshold","")),
    ])

# ---------- NEW: robust finders ----------
def try_variant_ext(path):
    if not isinstance(path, str): return None
    base, ext = os.path.splitext(path)
    if ext.lower() == ".tif":
        alt = base + ".tiff"
    elif ext.lower() == ".tiff":
        alt = base + ".tif"
    else:
        return None
    return alt if os.path.isfile(alt) else None

def try_same_tail_under_root(path_after_rewrite, side_root):
    """
    If path_after_rewrite still doesn't exist, try to map its tail onto side_root.
    E.g. /old/root/A/B/file__mask16.tif -> join(side_root, A/B/file__mask16.tif) if A/B exists.
    """
    if not isinstance(path_after_rewrite, str): return None
    tail = None
    # find the first segment that contains the output folder name
    for marker in ("navier_output_bestch","navier_output_fused","navier_output","navier_output_2"):
        idx = path_after_rewrite.find(marker + os.sep)
        if idx != -1:
            tail = path_after_rewrite[idx+len(marker)+1:]
            break
    if tail is None:
        # fallback: use just the filename
        tail = os.path.basename(path_after_rewrite)
    cand = os.path.join(side_root, tail)
    return cand if os.path.isfile(cand) else None

def try_glob_by_stem(side_root, filename_stem):
    pattern = os.path.join(side_root, "**", f"{filename_stem}__*__mask16.tif")
    hits = glob.glob(pattern, recursive=True)
    if hits: return hits[0]
    # try .tiff
    pattern2 = os.path.join(side_root, "**", f"{filename_stem}__*__mask16.tiff")
    hits2 = glob.glob(pattern2, recursive=True)
    return hits2[0] if hits2 else None

def resolve_mask_path(original_path, side_root, rewrites):
    """Apply rewrites; if missing, try same-tail, alt extension, and stem-based glob."""
    p1 = apply_rewrites(original_path, rewrites)
    if isinstance(p1, str) and os.path.isfile(p1):
        return p1
    # alt extension for p1
    alt = try_variant_ext(p1) if isinstance(p1, str) else None
    if alt: return alt
    # same tail under side_root
    p2 = try_same_tail_under_root(p1, side_root)
    if p2: return p2
    # stem-based search
    stem = os.path.splitext(os.path.basename(original_path or ""))[0]
    p3 = try_glob_by_stem(side_root, stem)
    return p3

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Compare bestch vs fused masks with robust path recovery + IoU.")
    ap.add_argument("--bestch", required=True, help="Folder for bestch results (has results_log.csv)")
    ap.add_argument("--fused",  required=True, help="Folder for fused results (has results_log.csv)")
    ap.add_argument("--out",    required=True, help="Output directory")
    ap.add_argument("--rewrite-root", nargs="*", default=[],
                    help="Pairs OLD=NEW to rewrite recorded paths to actual disk locations.")
    ap.add_argument("--save-montages", action="store_true")
    ap.add_argument("--montage-max", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rewrites = []
    for pair in args.rewrite_root:
        if "=" in pair:
            old, new = pair.split("=", 1)
            rewrites.append((old, new))
        else:
            print(f"[warn] bad rewrite '{pair}', expected OLD=NEW")

    # Load logs
    dfb = load_log_dir(args.bestch)
    dff = load_log_dir(args.fused)
    dfb["side"] = "bestch"
    dff["side"] = "fused"

    # Resolve paths robustly
    miss = []
    def resolve_df(df, side_root):
        out = []
        for r in df.itertuples(index=False):
            orig = getattr(r, "mask16_path", None)
            resolved = resolve_mask_path(orig, side_root, rewrites)
            ok = isinstance(resolved, str) and os.path.isfile(resolved)
            if not ok:
                miss.append({
                    "side": getattr(r, "side", ""),
                    "filename": getattr(r, "filename",""),
                    "mu": getattr(r, "mu", np.nan),
                    "lambda": getattr(r, "lambda", np.nan),
                    "diffusion_rate": getattr(r, "diffusion_rate", np.nan),
                    "energy_threshold": getattr(r, "energy_threshold", np.nan),
                    "mask16_path_orig": orig,
                    "mask16_path_resolved": resolved,
                    "exists": ok
                })
            d = r._asdict()
            d["mask16_path_resolved"] = resolved
            out.append(d)
        return pd.DataFrame(out)

    dfb2 = resolve_df(dfb, args.bestch)
    dff2 = resolve_df(dff, args.fused)

    diag_path = os.path.join(args.out, "missing_files_diagnostics.csv")
    pd.DataFrame(miss).to_csv(diag_path, index=False)
    if miss:
        print(f"[warn] Some mask paths are missing. See: {diag_path}")

    # keep rows with masks resolved
    dfb_ok = dfb2[dfb2["mask16_path_resolved"].apply(lambda p: isinstance(p,str) and os.path.isfile(p))].copy()
    dff_ok = dff2[dff2["mask16_path_resolved"].apply(lambda p: isinstance(p,str) and os.path.isfile(p))].copy()

    if dfb_ok.empty or dff_ok.empty:
        print("[fatal] No rows with existing masks after robust resolution.")
        print(f"Check {diag_path} and verify the roots passed for --bestch / --fused.")
        return

    # pair by robust key
    for df in (dfb_ok, dff_ok):
        df["pair_key"] = df.apply(make_pair_key_row, axis=1)

    left, right = dfb_ok.set_index("pair_key"), dff_ok.set_index("pair_key")
    keys = left.index.intersection(right.index)
    if len(keys) == 0:
        print("[fatal] No matching pairs (by filename, mu, lambda, d, E).")
        return

    # compute IoU/Dice
    rows = []
    for k in keys:
        lb, rf = left.loc[k], right.loc[k]
        if isinstance(lb, pd.DataFrame): lb = lb.iloc[0]
        if isinstance(rf, pd.DataFrame): rf = rf.iloc[0]
        mb = read_mask_binary(lb["mask16_path_resolved"])
        mf = read_mask_binary(rf["mask16_path_resolved"])
        J = iou(mb, mf)
        D = dice(mb, mf)
        rows.append({
            "pair_key": k,
            "filename": lb.get("filename",""),
            "mu": lb.get("mu", np.nan),
            "lambda": lb.get("lambda", np.nan),
            "diffusion_rate": lb.get("diffusion_rate", np.nan),
            "energy_threshold": lb.get("energy_threshold", np.nan),
            "iou": J, "dice": D,
            "bestch_mask": lb["mask16_path_resolved"],
            "fused_mask":  rf["mask16_path_resolved"],
        })
    pairwise = pd.DataFrame(rows)
    out_pair = os.path.join(args.out, "pairwise_matches.csv")
    pairwise.to_csv(out_pair, index=False)

    # summaries
    lvls = [0.5, 0.7, 0.9]
    def agg_block(g):
        out = {
            "n_pairs": g["iou"].notna().sum(),
            "iou_mean": g["iou"].mean(),
            "iou_median": g["iou"].median(),
            "dice_mean": g["dice"].mean(),
            "dice_median": g["dice"].median(),
        }
        for t in lvls:
            out[f"share_iou_ge_{t}"] = float((g["iou"] >= t).mean())
        return pd.Series(out)

    by_combo = (pairwise.groupby(["mu","lambda","diffusion_rate"], dropna=False)
        .apply(agg_block).reset_index().sort_values("iou_mean", ascending=False))
    by_grid  = (pairwise.groupby(["mu","lambda","diffusion_rate","energy_threshold"], dropna=False)
        .apply(agg_block).reset_index().sort_values("iou_mean", ascending=False))

    by_combo.to_csv(os.path.join(args.out, "convergence_by_combo.csv"), index=False)
    by_grid.to_csv(os.path.join(args.out, "convergence_by_param_grid.csv"), index=False)
    print("[ok] wrote pairwise + summary CSVs.")

    # optional montages
    if args.save_montages:
        samp = pairwise.dropna(subset=["iou"]).copy()
        samp["bucket"] = pd.cut(samp["iou"], bins=[-0.01,0.3,0.6,0.8,1.01], labels=["very_low","low","mid","high"])
        picks = []
        for b in ["very_low","low","mid","high"]:
            sub = samp[samp["bucket"]==b].head(max(1, args.montage_max//4))
            if len(sub): picks.append(sub)
        if picks:
            picks = pd.concat(picks)
            mdir = os.path.join(args.out, "montages")
            os.makedirs(mdir, exist_ok=True)
            for i, r in enumerate(picks.itertuples(index=False), 1):
                mb = read_mask_binary(r.bestch_mask)
                mf = read_mask_binary(r.fused_mask)
                if mb is None or mf is None: continue
                vb = (mb.astype(np.uint8))*255
                vf = (mf.astype(np.uint8))*255
                H = max(vb.shape[0], vf.shape[0])
                W = vb.shape[1] + vf.shape[1] + 10
                canvas = np.zeros((H, W), np.uint8)
                canvas[:vb.shape[0], :vb.shape[1]] = vb
                canvas[:vf.shape[0], vb.shape[1]+10:vb.shape[1]+10+vf.shape[1]] = vf
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
                cv2.putText(canvas, f"IoU={r.iou:.3f}  Dice={r.dice:.3f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(mdir, f"montage_{i:03d}.png"), canvas)
            print(f"[ok] montages saved at: {mdir}")

if __name__ == "__main__":
    main()
