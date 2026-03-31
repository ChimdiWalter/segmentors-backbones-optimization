#!/usr/bin/env python3
import os
import argparse
import csv
import numpy as np
import cv2

def binarize_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0)

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

def _read_opt_rows(path: str):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], []
        fields = [c.strip() for c in reader.fieldnames]
        rows = []
        for row in reader:
            clean = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            rows.append(clean)
    return rows, fields

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt_csv", required=True, help="Path to opt_summary.csv")
    ap.add_argument("--student_dir", required=True, help="Dir containing <base>_mask.png from UNet inference")
    ap.add_argument("--student_suffix", default="_mask.png", help="Suffix used by student masks")
    ap.add_argument("--out_csv", default="teacher_student_metrics.csv", help="Where to save per-image metrics CSV")
    args = ap.parse_args()

    rows_in, fields = _read_opt_rows(args.opt_csv)

    required = ["filename", "mask16_path"]
    for c in required:
        if c not in fields:
            raise ValueError(f"opt_summary.csv missing required column: {c}")

    # Keep successful rows if status exists
    if "status" in fields:
        rows_in = [r for r in rows_in if str(r.get("status", "")).strip().lower() == "ok"]

    rows = []
    missing_student = 0
    missing_teacher = 0

    for r in rows_in:
        fname = str(r.get("filename", "")).strip()
        base = os.path.splitext(os.path.basename(fname))[0]

        teacher = str(r.get("mask16_path", "")).strip()
        student = os.path.join(args.student_dir, base + args.student_suffix)

        if not teacher or not os.path.exists(teacher):
            missing_teacher += 1
            continue
        if not os.path.exists(student):
            missing_student += 1
            continue

        tmask = binarize_mask(teacher)
        smask = binarize_mask(student)

        # Ensure same shape (if something went wrong)
        if tmask.shape != smask.shape:
            # try resizing student to teacher size (should not be needed if stitch is correct)
            smask_u8 = smask.astype(np.uint8) * 255
            smask_u8 = cv2.resize(smask_u8, (tmask.shape[1], tmask.shape[0]), interpolation=cv2.INTER_NEAREST)
            smask = (smask_u8 > 0)

        dice, iou = dice_iou(tmask, smask)

        rows.append({
            "base": base,
            "dice": dice,
            "iou": iou,
            "teacher_path": teacher,
            "student_path": student,
            "teacher_area_px": int(tmask.sum()),
            "student_area_px": int(smask.sum())
        })

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_fields = [
        "base", "dice", "iou", "teacher_path", "student_path", "teacher_area_px", "student_area_px"
    ]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    if len(rows) == 0:
        print("No matched teacher/student pairs found.")
        print("Missing teacher:", missing_teacher)
        print("Missing student:", missing_student)
        return

    dice_arr = np.array([float(r["dice"]) for r in rows], dtype=np.float64)
    iou_arr = np.array([float(r["iou"]) for r in rows], dtype=np.float64)
    print("Matched pairs:", len(rows))
    print("Missing teacher:", missing_teacher)
    print("Missing student:", missing_student)
    print("\nSummary:")
    print("Dice  mean/std:", float(dice_arr.mean()), float(dice_arr.std(ddof=1)) if len(dice_arr) > 1 else 0.0)
    print("IoU   mean/std:", float(iou_arr.mean()), float(iou_arr.std(ddof=1)) if len(iou_arr) > 1 else 0.0)
    print("Dice  median  :", float(np.median(dice_arr)))
    print("IoU   median  :", float(np.median(iou_arr)))
    print("\nSaved:", args.out_csv)

if __name__ == "__main__":
    main()
