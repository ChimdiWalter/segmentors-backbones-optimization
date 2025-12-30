#!/usr/bin/env python3
import os, glob, argparse, textwrap
import numpy as np
import pandas as pd
import cv2

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_csvs(root: str) -> pd.DataFrame:
    cands = glob.glob(os.path.join(root, "**", "results_log.csv"), recursive=True)
    if not cands:
        cand = os.path.join(root, "results_log.csv")
        if os.path.isfile(cand): cands = [cand]
    frames = []
    for c in cands:
        try:
            df = pd.read_csv(c)
            df["__src_csv__"] = c
            frames.append(df)
        except Exception as e:
            print(f"[skip] {c}: {e}")
    if not frames:
        raise SystemExit(f"No results_log.csv found under: {root}")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    for col in ["mu","lambda","diffusion_rate","alpha","beta","gamma",
                "energy_threshold","n_contours","total_area_px"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def pick_mask_col(df: pd.DataFrame) -> str:
    for c in ["mask16_path","mask_path","mask_file","mask"]:
        if c in df.columns: return c
    raise SystemExit("No mask path column found (tried: mask16_path, mask_path, mask_file, mask).")

def pick_overlay_col(df: pd.DataFrame) -> str:
    for c in ["overlay16_path","overlay_path","overlay_file","overlay"]:
        if c in df.columns: return c
    # overlay is optional, but nice-to-have for montages
    return None

def mk_combo_tag(row, keys):
    parts=[]
    for k in keys:
        v=row.get(k, None)
        if pd.isna(v): continue
        if isinstance(v, float): parts.append(f"{k}{str(v).replace('.','p')}")
        else: parts.append(f"{k}{v}")
    return "__".join(parts) if parts else "combo"

def to_bool_mask(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img.astype(np.uint16) > 0)

def iou_dice(a, b):
    inter = np.logical_and(a,b).sum()
    union = np.logical_or(a,b).sum()
    iou = inter/union if union>0 else (1.0 if a.sum()==0 and b.sum()==0 else 0.0)
    denom = a.sum()+b.sum()
    dice = (2*inter/denom) if denom>0 else (1.0 if a.sum()==0 and b.sum()==0 else 0.0)
    return float(iou), float(dice)

def pct_diff(a, b):
    if (a+b) <= 0: return 0.0
    return abs(a-b)/(0.5*(a+b))

def main():
    ap = argparse.ArgumentParser(description="bestch vs fused: robust convergence analysis")
    ap.add_argument("--bestch", required=True)
    ap.add_argument("--fused",  required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--k-top",  type=int, default=12)
    ap.add_argument("--strictness", type=str, default="0.30,0.50,0.70")
    args = ap.parse_args()

    thr_loose, thr_med, thr_strict = [float(x) for x in args.strictness.split(",")]
    ensure_dir(args.out)

    df_b = read_csvs(args.bestch)
    df_f = read_csvs(args.fused)

    # choose defining keys (intersection)
    keys = [k for k in ["mu","lambda","diffusion_rate","alpha","beta","gamma","energy_threshold"]
            if k in df_b.columns and k in df_f.columns]
    if not {"mu","lambda","diffusion_rate"}.issubset(keys):
        raise SystemExit("Both logs must contain mu, lambda, diffusion_rate.")

    mask_b = pick_mask_col(df_b); mask_f = pick_mask_col(df_f)
    over_b = pick_overlay_col(df_b); over_f = pick_overlay_col(df_f)

    need_b = ["filename", mask_b, "n_contours", "total_area_px"] + keys
    need_f = ["filename", mask_f, "n_contours", "total_area_px"] + keys
    for miss, name in [(set(need_b)-set(df_b.columns), "bestch"),
                       (set(need_f)-set(df_f.columns), "fused")]:
        if miss:
            raise SystemExit(f"{name} CSV missing columns: {sorted(miss)}")

    on_cols = ["filename"] + keys
    merged = pd.merge(
        df_f[on_cols + [mask_f, over_f, "n_contours","total_area_px"]],
        df_b[on_cols + [mask_b, over_b, "n_contours","total_area_px"]],
        on=on_cols, suffixes=("_fused","_bestch")
    )
    if merged.empty:
        raise SystemExit("No intersecting (filename + combo) pairs between fused and bestch.")

    # diagnostics for missing files
    merged["exists_mask_fused"]  = merged[f"{mask_f}_fused"].map(lambda p: os.path.exists(str(p)))
    merged["exists_mask_bestch"] = merged[f"{mask_b}_bestch"].map(lambda p: os.path.exists(str(p)))
    diag = merged[~(merged["exists_mask_fused"] & merged["exists_mask_bestch"])][
        on_cols + [f"{mask_f}_fused", f"{mask_b}_bestch", "exists_mask_fused", "exists_mask_bestch"]
    ]
    if not diag.empty:
        diag_path = os.path.join(args.out, "missing_files_diagnostics.csv")
        diag.to_csv(diag_path, index=False)
        print(f"[warn] Some mask paths are missing. See: {diag_path}")

    rows=[]
    bad=0; used=0
    for _, r in merged.iterrows():
        p_f = r[f"{mask_f}_fused"]; p_b = r[f"{mask_b}_bestch"]
        if not (isinstance(p_f,str) and os.path.exists(p_f) and isinstance(p_b,str) and os.path.exists(p_b)):
            bad+=1; continue
        mf = to_bool_mask(p_f); mb = to_bool_mask(p_b)
        if mf is None or mb is None: bad+=1; continue
        if mf.shape != mb.shape:
            mb = cv2.resize(mb.astype(np.uint8), (mf.shape[1], mf.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        iou, dice = iou_dice(mf, mb)
        rows.append({
            **{k:r[k] for k in keys},
            "filename": r["filename"],
            "overlay_fused": r.get(f"{over_f}_fused") if over_f else "",
            "overlay_bestch": r.get(f"{over_b}_bestch") if over_b else "",
            "mask_fused": p_f,
            "mask_bestch": p_b,
            "n_contours_fused": r["n_contours_fused"],
            "n_contours_bestch": r["n_contours_bestch"],
            "total_area_px_fused": r["total_area_px_fused"],
            "total_area_px_bestch": r["total_area_px_bestch"],
            "IoU": iou,
            "Dice": dice,
            "count_diff_pct": pct_diff(float(r["n_contours_fused"]), float(r["n_contours_bestch"])),
            "area_diff_pct": pct_diff(float(r["total_area_px_fused"]), float(r["total_area_px_bestch"])),
        })
        used+=1

    pairwise = pd.DataFrame(rows)
    pair_csv = os.path.join(args.out, "pairwise_metrics.csv")
    pairwise.to_csv(pair_csv, index=False)

    if pairwise.empty:
        msg = "No valid pairs after reading masks. Check missing_files_diagnostics.csv; verify your directory args point to the folders that actually contain the TIFFs."
        print("[fatal] " + msg)
        with open(os.path.join(args.out, "README.txt"), "w") as fh:
            fh.write(msg + "\n")
        return

    # safe groupby on present key columns
    group_keys = [k for k in keys if k in pairwise.columns]
    def conv_rate(vals, thr): 
        v = np.asarray(vals, float)
        return float(np.mean(v >= thr)) if v.size else 0.0

    summary = (pairwise.groupby(group_keys, as_index=False)
        .agg(
            n_pairs=("IoU","size"),
            conv_loose=("IoU", lambda s: conv_rate(s, thr_loose)),
            conv_medium=("IoU", lambda s: conv_rate(s, thr_med)),
            conv_strict=("IoU", lambda s: conv_rate(s, thr_strict)),
            mean_IoU=("IoU","mean"),
            mean_Dice=("Dice","mean"),
            mean_count_diff_pct=("count_diff_pct","mean"),
            mean_area_diff_pct=("area_diff_pct","mean"),
        )
        .sort_values(["conv_medium","mean_IoU"], ascending=[False, False])
    )
    summary["combo_tag"] = summary.apply(lambda r: mk_combo_tag(r, group_keys), axis=1)
    sum_csv = os.path.join(args.out, "summary_convergence.csv")
    summary.to_csv(sum_csv, index=False)

    with open(os.path.join(args.out, "README.txt"), "w") as fh:
        fh.write(textwrap.dedent(f"""
        bestch vs fused — robust convergence summary
        ============================================
        Inputs:
          bestch: {args.bestch}
          fused : {args.fused}

        Pairs merged:   {len(merged)}
        Masks read OK:  {used}
        Masks failed:   {bad}

        Outputs:
          - {os.path.basename(sum_csv)}     (per-combo rates + means)
          - {os.path.basename(pair_csv)}    (per-image IoU/Dice + deltas)
          - missing_files_diagnostics.csv   (if any) — helps fix bad paths
        """).strip()+"\n")

    print("[done]")
    print("  ", sum_csv)
    print("  ", pair_csv)

if __name__ == "__main__":
    main()
