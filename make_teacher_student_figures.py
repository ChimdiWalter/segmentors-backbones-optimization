#!/usr/bin/env python3
import os
import shutil
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

METRICS_CSV = "experiments/exp_v1/outputs/teacher_student_metrics.csv"
TEACHER_DIR = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/cpu_output_3"
STUDENT_DIR = "experiments/exp_v1/outputs/unet_infer_on_leaves"
OUT_DIR     = "experiments/exp_v1/outputs/teacher_student_figures"

def ensure(p): os.makedirs(p, exist_ok=True)

def find_teacher_overlay(base: str):
    # overlays are parameter-tagged; find by prefix
    prefix = base + "__"
    for f in os.listdir(TEACHER_DIR):
        if f.startswith(prefix) and f.endswith("__overlay16.tif"):
            return os.path.join(TEACHER_DIR, f)
    return None

def read_overlay16_to_8(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # if 16-bit, scale down for viewing
    if img.dtype == np.uint16:
        img8 = (img / 257).astype(np.uint8)
    else:
        img8 = img.astype(np.uint8)
    return img8

def main():
    ensure(OUT_DIR)
    ensure(os.path.join(OUT_DIR, "plots"))
    ensure(os.path.join(OUT_DIR, "examples", "best"))
    ensure(os.path.join(OUT_DIR, "examples", "median"))
    ensure(os.path.join(OUT_DIR, "examples", "worst"))

    df = pd.read_csv(METRICS_CSV).sort_values("dice").reset_index(drop=True)

    # ---- Summary text ----
    dice = df["dice"].to_numpy()
    iou  = df["iou"].to_numpy()

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Matched pairs: {len(df)}\n")
        f.write(f"Dice mean/std: {dice.mean():.6f} {dice.std(ddof=1):.6f}\n")
        f.write(f"Dice median:   {np.median(dice):.6f}\n")
        f.write(f"Dice p10/p25/p75/p90: {np.quantile(dice,0.10):.6f} {np.quantile(dice,0.25):.6f} {np.quantile(dice,0.75):.6f} {np.quantile(dice,0.90):.6f}\n")
        f.write(f"IoU  mean/std: {iou.mean():.6f} {iou.std(ddof=1):.6f}\n")
        f.write(f"IoU  median:   {np.median(iou):.6f}\n")
        f.write(f"IoU  p10/p25/p75/p90: {np.quantile(iou,0.10):.6f} {np.quantile(iou,0.25):.6f} {np.quantile(iou,0.75):.6f} {np.quantile(iou,0.90):.6f}\n")
        f.write(f"Dice>=0.7 fraction: {(dice>=0.7).mean():.6f}\n")
        f.write(f"Dice<0.3 count: {(dice<0.3).sum()}\n")

    # ---- Plots: histograms ----
    plt.figure()
    plt.hist(dice, bins=30)
    plt.title("Teacher vs Student Dice distribution")
    plt.xlabel("Dice")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plots", "dice_hist.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.hist(iou, bins=30)
    plt.title("Teacher vs Student IoU distribution")
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plots", "iou_hist.png"), dpi=200)
    plt.close()

    # ---- Plot: Dice vs teacher area (diagnostic) ----
    plt.figure()
    plt.scatter(df["teacher_area_px"], df["dice"], s=10)
    plt.title("Dice vs Teacher Mask Area")
    plt.xlabel("Teacher area (px)")
    plt.ylabel("Dice")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plots", "dice_vs_teacher_area.png"), dpi=200)
    plt.close()

    # ---- Save example overlays (best/median/worst) ----
    def copy_examples(subdf, out_subdir):
        for _, r in subdf.iterrows():
            base = r["base"]
            # student overlay
            s_overlay = os.path.join(STUDENT_DIR, base + "_overlay.jpg")
            # teacher overlay16 (search)
            t_overlay = find_teacher_overlay(base)

            # copy student overlay
            if os.path.exists(s_overlay):
                shutil.copy2(s_overlay, os.path.join(out_subdir, base + "__student_overlay.jpg"))

            # convert teacher overlay16 to 8-bit png for easy viewing
            if t_overlay and os.path.exists(t_overlay):
                img8 = read_overlay16_to_8(t_overlay)
                if img8 is not None:
                    cv2.imwrite(os.path.join(out_subdir, base + "__teacher_overlay.png"), img8)

            # write a small txt with metrics
            with open(os.path.join(out_subdir, base + "__metrics.txt"), "w") as f:
                f.write(f"base={base}\n")
                f.write(f"dice={r['dice']:.6f}\n")
                f.write(f"iou={r['iou']:.6f}\n")
                f.write(f"teacher_area_px={int(r['teacher_area_px'])}\n")
                f.write(f"student_area_px={int(r['student_area_px'])}\n")

    worst = df.head(5)
    mid = df.iloc[len(df)//2 - 2 : len(df)//2 + 3]
    best = df.tail(5)

    copy_examples(worst, os.path.join(OUT_DIR, "examples", "worst"))
    copy_examples(mid,   os.path.join(OUT_DIR, "examples", "median"))
    copy_examples(best,  os.path.join(OUT_DIR, "examples", "best"))

    print("Wrote:", OUT_DIR)
    print("Summary:", summary_path)
    print("Plots:", os.path.join(OUT_DIR, "plots"))
    print("Examples:", os.path.join(OUT_DIR, "examples"))

if __name__ == "__main__":
    main()