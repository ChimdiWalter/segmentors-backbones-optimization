#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def binarize_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0)

def read_prob_u8(path: str) -> np.ndarray:
    p = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if p is None:
        raise RuntimeError(f"Failed to read prob map: {path}")
    if p.ndim == 3:
        p = p[..., 0]
    # saved as 0..255 uint8
    if p.dtype != np.uint8:
        p = p.astype(np.uint8)
    return p

def dice_iou(a: np.ndarray, b: np.ndarray):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    ap = a.sum()
    bp = b.sum()
    union = np.logical_or(a, b).sum()
    if ap == 0 and bp == 0:
        return 1.0, 1.0
    dice = (2.0 * inter) / (ap + bp + 1e-8)
    iou  = inter / (union + 1e-8)
    return float(dice), float(iou)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt_csv", required=True, help="opt_summary_local.csv with valid mask16_path")
    ap.add_argument("--student_dir", required=True, help="Folder containing <base>_prob.png")
    ap.add_argument("--prob_suffix", default="_prob.png")
    ap.add_argument("--out_csv", default="experiments/exp_v1/outputs/threshold_sweep.csv")
    ap.add_argument("--plot_path", default="experiments/exp_v1/outputs/threshold_sweep_plot.png")
    args = ap.parse_args()

    df = pd.read_csv(args.opt_csv)
    df.columns = df.columns.str.strip()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().str.lower() == "ok"].copy()

    for c in ["filename", "mask16_path"]:
        if c not in df.columns:
            raise ValueError(f"Missing column in opt_csv: {c}")

    # thresholds to test
    thresholds = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80]

    rows = []
    missing_prob = 0

    for thr in thresholds:
        dices_all, ious_all = [], []
        dices_ne,  ious_ne  = [], []
        halluc = 0  # teacher empty, student nonempty
        miss   = 0  # teacher nonempty, student empty

        for _, r in df.iterrows():
            fname = str(r["filename"]).strip()
            base = os.path.splitext(os.path.basename(fname))[0]

            teacher_path = str(r["mask16_path"]).strip()
            if not os.path.exists(teacher_path):
                raise RuntimeError(f"Teacher mask missing: {teacher_path}")

            prob_path = os.path.join(args.student_dir, base + args.prob_suffix)
            if not os.path.exists(prob_path):
                missing_prob += 1
                continue

            tmask = binarize_mask(teacher_path)
            prob_u8 = read_prob_u8(prob_path)
            if prob_u8.shape != tmask.shape:
                # Should not happen; if it does, resize prob to teacher shape
                prob_u8 = cv2.resize(prob_u8, (tmask.shape[1], tmask.shape[0]), interpolation=cv2.INTER_LINEAR)

            prob = prob_u8.astype(np.float32) / 255.0
            smask = (prob > thr)

            dice, iou = dice_iou(tmask, smask)
            dices_all.append(dice); ious_all.append(iou)

            t_empty = (tmask.sum() == 0)
            s_empty = (smask.sum() == 0)

            if (t_empty and (not s_empty)):
                halluc += 1
            if ((not t_empty) and s_empty):
                miss += 1

            if not t_empty:
                dices_ne.append(dice); ious_ne.append(iou)

        if len(dices_all) == 0:
            raise RuntimeError("No matches found. Are <base>_prob.png files present in student_dir?")

        rows.append({
            "threshold": thr,
            "n_total": len(dices_all),
            "dice_mean_all": float(np.mean(dices_all)),
            "iou_mean_all":  float(np.mean(ious_all)),
            "dice_median_all": float(np.median(dices_all)),
            "iou_median_all":  float(np.median(ious_all)),
            "n_teacher_nonempty": len(dices_ne),
            "dice_mean_teacher_nonempty": float(np.mean(dices_ne)) if dices_ne else np.nan,
            "iou_mean_teacher_nonempty":  float(np.mean(ious_ne)) if ious_ne else np.nan,
            "hallucinations_teacher_empty_student_nonempty": halluc,
            "misses_teacher_nonempty_student_empty": miss
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # pick best threshold by maximizing non-empty mean Dice, tie-break by fewer hallucinations
    out2 = out.copy()
    out2["rank_score"] = out2["dice_mean_teacher_nonempty"] - 0.001*out2["hallucinations_teacher_empty_student_nonempty"]
    best = out2.sort_values("rank_score", ascending=False).iloc[0]
    print("\nSaved:", args.out_csv)
    print("\nBest threshold (heuristic):", float(best["threshold"]))
    print(best.to_string())

    # plot
    plt.figure()
    plt.plot(out["threshold"], out["dice_mean_all"], marker="o", label="Dice mean (all)")
    plt.plot(out["threshold"], out["dice_mean_teacher_nonempty"], marker="o", label="Dice mean (teacher non-empty)")
    plt.xlabel("Threshold")
    plt.ylabel("Dice")
    plt.title("Threshold sweep (from saved prob maps)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.plot_path) or ".", exist_ok=True)
    plt.savefig(args.plot_path, dpi=200)
    plt.close()
    print("Saved plot:", args.plot_path)

    if missing_prob > 0:
        print("\nWARNING: missing prob maps for", missing_prob, "images.")
        print("Make sure inference on leaves saved <base>_prob.png into student_dir.")

if __name__ == "__main__":
    main()