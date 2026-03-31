#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import cv2

def read_prob_u8(path: str) -> np.ndarray:
    p = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if p is None:
        raise RuntimeError(f"Failed to read prob: {path}")
    if p.ndim == 3:
        p = p[..., 0]
    return p.astype(np.uint8)

def binarize_teacher(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read teacher mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0)

def filter_components(mask_u8, min_area):
    if min_area <= 0:
        return mask_u8
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def dice_iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    ap = a.sum(); bp = b.sum()
    union = np.logical_or(a, b).sum()
    if ap == 0 and bp == 0:
        return 1.0, 1.0
    dice = (2.0 * inter) / (ap + bp + 1e-8)
    iou  = inter / (union + 1e-8)
    return float(dice), float(iou)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt_csv", required=True)
    ap.add_argument("--student_dir", required=True)
    ap.add_argument("--prob_suffix", default="_prob.png")
    ap.add_argument("--thr", type=float, default=0.30)
    ap.add_argument("--out_csv", default="experiments/exp_v1/outputs/postprocess_sweep.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.opt_csv)
    df.columns = df.columns.str.strip()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().str.lower() == "ok"].copy()

    # sweep grids
    min_areas = [0, 100, 200, 500, 1000, 2000, 5000]
    max_area_fracs = [1.0, 0.8, 0.6, 0.4]   # 1.0 = no cap

    rows = []

    for min_area in min_areas:
        for cap in max_area_fracs:
            dices_all, ious_all = [], []
            dices_ne,  ious_ne  = [], []
            halluc = 0
            miss = 0

            for _, r in df.iterrows():
                fname = str(r["filename"]).strip()
                base = os.path.splitext(os.path.basename(fname))[0]

                teacher_path = str(r["mask16_path"]).strip()
                if not os.path.exists(teacher_path):
                    raise RuntimeError(f"Missing teacher: {teacher_path}")

                prob_path = os.path.join(args.student_dir, base + args.prob_suffix)
                if not os.path.exists(prob_path):
                    raise RuntimeError(f"Missing prob: {prob_path}")

                tmask = binarize_teacher(teacher_path)

                prob = read_prob_u8(prob_path).astype(np.float32) / 255.0
                smask_u8 = ((prob > args.thr).astype(np.uint8) * 255)

                # component filtering
                smask_u8 = filter_components(smask_u8, min_area=min_area)

                # cap by area fraction (relative to whole image)
                area_frac = (smask_u8 > 0).mean()
                if area_frac > cap:
                    smask_u8[:] = 0

                smask = (smask_u8 > 0)

                dice, iou = dice_iou(tmask, smask)
                dices_all.append(dice); ious_all.append(iou)

                t_empty = (tmask.sum() == 0)
                s_empty = (smask.sum() == 0)
                if t_empty and (not s_empty):
                    halluc += 1
                if (not t_empty) and s_empty:
                    miss += 1

                if not t_empty:
                    dices_ne.append(dice); ious_ne.append(iou)

            dices_all = np.array(dices_all); ious_all = np.array(ious_all)
            dices_ne  = np.array(dices_ne) if len(dices_ne) else np.array([np.nan])
            ious_ne   = np.array(ious_ne)  if len(ious_ne)  else np.array([np.nan])

            rows.append({
                "thr": args.thr,
                "min_area": min_area,
                "max_area_frac": cap,
                "dice_mean_all": float(np.nanmean(dices_all)),
                "iou_mean_all": float(np.nanmean(ious_all)),
                "dice_mean_teacher_nonempty": float(np.nanmean(dices_ne)),
                "iou_mean_teacher_nonempty": float(np.nanmean(ious_ne)),
                "hallucinations_teacher_empty_student_nonempty": int(halluc),
                "misses_teacher_nonempty_student_empty": int(miss),
            })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # pick best: maximize nonempty dice, penalize hallucinations
    out2 = out.copy()
    out2["rank_score"] = out2["dice_mean_teacher_nonempty"] - 0.002*out2["hallucinations_teacher_empty_student_nonempty"] - 0.002*out2["misses_teacher_nonempty_student_empty"]
    best = out2.sort_values("rank_score", ascending=False).iloc[0]

    print("Saved:", args.out_csv)
    print("\nBest setting (heuristic):")
    print(best.to_string(index=False))

if __name__ == "__main__":
    main()